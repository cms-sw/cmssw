// Contract tests for the RNTuple behaviour NanoAODRNTupleOutputModule is built on.
//
// The output module (PhysicsTools/NanoAOD/plugins/rntuple/) writes its grouped fields as untyped
// records and untyped vectors of records, gives every member a flat TTree-style name with a
// projected field, and grows a trigger record while the ntuple is being written. None of that is
// ordinary RNTuple usage, the API behind it has moved repeatedly across ROOT 6.3x, and the release
// this work targets (CMSSW_20_1_0_pre2_ROOT640, ROOT 6.40) cannot be built on every developer
// machine. These tests pin the behaviour down so a ROOT upgrade that changes it fails here rather
// than silently in the output.
//
// They exercise ROOT directly, in the same shapes and the same order as the module, rather than the
// module's own classes: those live in a plugin library, which a test binary cannot link against.
//
// The late-extension section is where the ROOT version matters. Adding a subfield to an untyped
// record of a model that is already being written needs RUpdater::AddField(field, parentName),
// which is 6.40 and newer; before that the module writes the path as a top-level field instead, and
// this test follows the same split, so each build tests the path it actually takes.

#include "catch2/catch_all.hpp"

#include <ROOT/REntry.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RField/RFieldRecord.hxx>
#include <ROOT/RField/RFieldSequenceContainer.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RVersion.hxx>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

static constexpr auto s_tag = "[NanoAODRNTupleFields]";

namespace {

  using ROOT::RFieldBase;
  using ROOT::RNTupleModel;
  using ROOT::RNTupleReader;
  using ROOT::RNTupleWriter;
  using ROOT::RRecordField;
  using ROOT::RVectorField;

  // Every type nanoaod::FlatTable::ColumnType can hold, as RNTuple spells them. A column of any of
  // these can end up as a member of a collection or a singleton record, so all of them must project.
  const std::vector<std::string> kColumnTypes = {"bool",
                                                 "std::uint8_t",
                                                 "std::int8_t",
                                                 "std::uint16_t",
                                                 "std::int16_t",
                                                 "std::uint32_t",
                                                 "std::int32_t",
                                                 "std::uint64_t",
                                                 "std::int64_t",
                                                 "float",
                                                 "double"};

  // A file that deletes itself, so a failing assertion does not leave one behind.
  class ScopedFile {
  public:
    explicit ScopedFile(const std::string& name) : m_path(std::filesystem::temp_directory_path() / name) {
      std::filesystem::remove(m_path);
    }
    ~ScopedFile() { std::filesystem::remove(m_path); }
    std::string path() const { return m_path.string(); }

  private:
    std::filesystem::path m_path;
  };

  // An untyped record, as TableCollectionSet builds one for a singleton table or a trigger group.
  std::unique_ptr<RRecordField> makeRecord(const std::string& name, const std::vector<std::string>& memberTypes) {
    std::vector<std::unique_ptr<RFieldBase>> members;
    members.reserve(memberTypes.size());
    for (const auto& type : memberTypes) {
      members.push_back(RFieldBase::Create("m_" + type, type).Unwrap());
    }
    return std::make_unique<RRecordField>(name, std::move(members));
  }

  // The projection of one member, as rntupleprojection::add makes it: a scalar for a member of a
  // plain record, a vector for a member of a record inside a vector, and a mapping for each level.
  // Returns whether ROOT accepted it -- an RResult cannot be handed to REQUIRE directly, and it
  // throws when destroyed unchecked, so it is inspected here.
  bool addProjection(RNTupleModel& model,
                     const std::string& fieldName,
                     const std::string& memberName,
                     const std::string& memberTypeName,
                     bool inCollection) {
    const std::string name = fieldName + "_" + memberName;
    auto field =
        RFieldBase::Create(name, inCollection ? "std::vector<" + memberTypeName + ">" : memberTypeName).Unwrap();
    const std::string member = fieldName + (inCollection ? "._0." : ".") + memberName;
    auto result = model.AddProjectedField(std::move(field),
                                          [name, fieldName, member, inCollection](const std::string& projected) {
                                            return (inCollection && projected == name) ? fieldName : member;
                                          });
    return static_cast<bool>(result);
  }

  const RRecordField* recordOf(const RNTupleModel& model, const std::string& fieldName) {
    return dynamic_cast<const RRecordField*>(&model.GetConstField(fieldName));
  }

}  // anonymous namespace

TEST_CASE("Grouped fields carry flat projections of every column type", s_tag) {
  ScopedFile file{"testNanoAODRNTupleProjections.root"};

  // The Events model is bare: ROOT refuses to add a subfield to an untyped record of a model that
  // has a default entry, which is what the trigger backfill needs.
  auto model = RNTupleModel::CreateBare();

  // A named collection: one untyped vector of an untyped record, one member per column type.
  auto item = makeRecord("_0", kColumnTypes);
  const auto collOffsets = item->GetOffsets();
  const auto collRecordSize = item->GetValueSize();
  model->AddField(RVectorField::CreateUntyped("Coll", std::move(item)));

  // A named singleton table: the untyped record on its own, no vector around it.
  auto singleton = makeRecord("Sing", {"float"});
  const auto singOffsets = singleton->GetOffsets();
  const auto singRecordSize = singleton->GetValueSize();
  model->AddField(std::move(singleton));

  SECTION("every column type projects, in a collection and on its own") {
    for (const auto& type : kColumnTypes) {
      REQUIRE(addProjection(*model, "Coll", "m_" + type, type, true));
    }
    REQUIRE(addProjection(*model, "Sing", "m_float", "float", false));
  }

  SECTION("a projection onto a name the model already holds throws") {
    // rntupleprojection::add checks GetFieldNames() before adding for exactly this reason: the
    // clash comes back as an exception, not in the RResult, so it cannot simply be inspected.
    model->MakeField<float>("Sing_m_float");
    REQUIRE_THROWS(addProjection(*model, "Sing", "m_float", "float", false));
  }

  SECTION("a projection of a member the field does not have is refused, not thrown") {
    REQUIRE_FALSE(addProjection(*model, "Sing", "m_absent", "float", false));
  }

  SECTION("projections read back the values of the members they project") {
    for (const auto& type : kColumnTypes) {
      REQUIRE(addProjection(*model, "Coll", "m_" + type, type, true));
    }
    REQUIRE(addProjection(*model, "Sing", "m_float", "float", false));
    model->Freeze();

    auto writer = RNTupleWriter::Recreate(std::move(model), "Events", file.path());
    auto entry = writer->GetModel().CreateBareEntry();

    // Bound the way the module binds them: a vector field's value is the std::vector object, a
    // record's value is the memory holding it, so one gets the buffer and the other its contents.
    std::vector<unsigned char> collBuffer;
    std::vector<unsigned char> singBuffer(singRecordSize, 0);
    entry->BindRawPtr<void>("Coll", &collBuffer);
    entry->BindRawPtr<void>("Sing", singBuffer.data());

    constexpr std::size_t kEntries = 3;
    constexpr std::size_t kRows = 2;
    for (std::size_t i = 0; i < kEntries; i++) {
      collBuffer.assign(collRecordSize * kRows, 0);
      for (std::size_t row = 0; row < kRows; row++) {
        // One byte per member is enough to tell the columns apart -- every type here is at least a
        // byte wide, and on a little-endian machine that byte is the low one -- and it is the byte
        // the projection has to come back with. The exception is bool, whose column keeps only the
        // low bit, so it gets a value that is a valid bool to begin with.
        for (std::size_t m = 0; m < kColumnTypes.size(); m++) {
          const auto value = static_cast<unsigned char>(1 + i + row + m);
          collBuffer[row * collRecordSize + collOffsets[m]] =
              (kColumnTypes[m] == "bool") ? static_cast<unsigned char>(value & 1u) : value;
        }
      }
      const float value = 0.5f * static_cast<float>(i);
      std::memcpy(singBuffer.data() + singOffsets[0], &value, sizeof(value));
      writer->Fill(*entry);
    }
    writer.reset();

    auto reader = RNTupleReader::Open("Events", file.path());
    REQUIRE(reader->GetNEntries() == kEntries);

    // The members are read under their flat names, which is the point: each is its own column of
    // the type the member has, wrapped in a vector because the record sits inside one.
    auto bools = reader->GetView<std::vector<bool>>("Coll_m_bool");
    auto bytes = reader->GetView<std::vector<std::uint8_t>>("Coll_m_std::uint8_t");
    auto ints = reader->GetView<std::vector<std::int32_t>>("Coll_m_std::int32_t");
    auto binvar = reader->GetView<float>("Sing_m_float");
    for (std::size_t i = 0; i < kEntries; i++) {
      REQUIRE(bools(i).size() == kRows);
      REQUIRE(bytes(i).size() == kRows);
      REQUIRE(ints(i).size() == kRows);
      REQUIRE(binvar(i) == 0.5f * static_cast<float>(i));
      for (std::size_t row = 0; row < kRows; row++) {
        // Members 0, 1 and 6 of the record, each read back with what was written into it.
        REQUIRE(bools(i)[row] == (((1 + i + row) & 1u) != 0));
        REQUIRE(bytes(i)[row] == static_cast<std::uint8_t>(1 + i + row + 1));
        REQUIRE(ints(i)[row] == static_cast<std::int32_t>(1 + i + row + 6));
      }
    }
  }
}

TEST_CASE("A record grows while the ntuple is being written", s_tag) {
  ScopedFile file{"testNanoAODRNTupleBackfill.root"};

  auto model = RNTupleModel::CreateBare();
  model->AddField(makeRecord("HLT", {"bool"}));
  REQUIRE(addProjection(*model, "HLT", "m_bool", "bool", false));
  model->Freeze();

  auto writer = RNTupleWriter::Recreate(std::move(model), "Events", file.path());

  // What the late path leaves behind differs by version, and so does what has to be bound.
  //
  // On 6.40 the member goes into the record and HLT_m_late is a *projection* of it: a projection is
  // read-only, derived from the field it projects, and has no value in an entry at all -- binding
  // one throws "invalid field name". The value reaches it through the record buffer, so there is
  // nothing extra to bind. That is why TriggerRecordFields only ever fills m_looseFlatNames, the
  // list bind() walks, on the pre-6.40 branch.
  //
  // Before 6.40 HLT_m_late is a real top-level field and does have to be bound; the projection
  // there is HLT_m_late_alias, onto that field.
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 40, 0)
  constexpr bool kBindsLateField = false;
  const std::string kProjectionName = "HLT_m_late";
#else
  constexpr bool kBindsLateField = true;
  const std::string kProjectionName = "HLT_m_late_alias";
#endif

  // A bare model has no default entry, and every schema change invalidates the entries made against
  // the previous schema, so the module rebuilds and rebinds its entry after each one. Same here.
  std::vector<unsigned char> buffer;
  auto lateValue = std::make_shared<bool>(false);
  auto bindEntry = [&writer, &buffer, &lateValue](bool haveLateField) {
    const auto& model = writer->GetModel();
    auto entry = model.CreateBareEntry();
    const auto* record = recordOf(model, "HLT");
    REQUIRE(record != nullptr);
    buffer.assign(record->GetValueSize(), 0);
    entry->BindRawPtr<void>("HLT", buffer.data());
    if (haveLateField) {
      entry->BindValue<bool>("HLT_m_late", lateValue);
    }
    return entry;
  };

  auto entry = bindEntry(false);
  REQUIRE(recordOf(writer->GetModel(), "HLT") != nullptr);
  const auto offsets = recordOf(writer->GetModel(), "HLT")->GetOffsets();
  for (std::size_t i = 0; i < 2; i++) {
    buffer[offsets[0]] = 1;
    writer->Fill(*entry);
  }

  {
    // The updater commits in its destructor. Committing explicitly as well commits twice, which
    // leaves the model id at 0 and makes the next Fill() throw; the module documents this and so
    // does not call CommitUpdate() either, which is what the Fill() below guards.
    auto updater = writer->CreateModelUpdater();
    updater->BeginUpdate();
    auto field = RFieldBase::Create("m_late", "bool").Unwrap();
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 40, 0)
    // 6.40 and newer: the member goes into the record, and its projection is added in the same
    // transaction, resolving against the member added just above.
    updater->AddField(std::move(field), "HLT");
    auto projection = RFieldBase::Create("HLT_m_late", "bool").Unwrap();
    auto added = updater->AddProjectedField(std::move(projection),
                                            [](const std::string&) -> std::string { return "HLT.m_late"; });
    REQUIRE(static_cast<bool>(added));
#else
    // Before 6.40 a record cannot grow, and the module writes the path as a top-level field under
    // the flat name the projection would have had. A projection onto that late field still works,
    // which is what says the two operations may share one transaction.
    field = RFieldBase::Create("HLT_m_late", "bool").Unwrap();
    updater->AddField(std::move(field));
    auto projection = RFieldBase::Create("HLT_m_late_alias", "bool").Unwrap();
    auto added = updater->AddProjectedField(std::move(projection),
                                            [](const std::string&) -> std::string { return "HLT_m_late"; });
    REQUIRE(static_cast<bool>(added));
#endif
  }

  // A projection is not part of an entry on either version, whichever field it is a projection of.
  REQUIRE_THROWS(writer->GetModel().CreateBareEntry()->BindValue<bool>(kProjectionName, lateValue));

  entry = bindEntry(kBindsLateField);
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 40, 0)
  // The record grew, so its offsets moved and the buffer was resized: re-read them, as bind() does.
  REQUIRE(recordOf(writer->GetModel(), "HLT") != nullptr);
  const auto grownOffsets = recordOf(writer->GetModel(), "HLT")->GetOffsets();
  REQUIRE(grownOffsets.size() == 2);
#endif
  for (std::size_t i = 0; i < 2; i++) {
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 40, 0)
    buffer[grownOffsets[0]] = 1;
    buffer[grownOffsets[1]] = 1;
#else
    buffer[offsets[0]] = 1;
    *lateValue = true;
#endif
    REQUIRE_NOTHROW(writer->Fill(*entry));
  }
  writer.reset();

  auto reader = RNTupleReader::Open("Events", file.path());
  REQUIRE(reader->GetNEntries() == 4);

  auto early = reader->GetView<bool>("HLT_m_bool");
  auto late = reader->GetView<bool>("HLT_m_late");
  for (std::size_t i = 0; i < 4; i++) {
    // The member written from the start is true throughout, projection and all.
    REQUIRE(early(i));
    // The one added later is a deferred column: the entries written before it read back false,
    // which is what makes a trigger path first seen in a later run safe to add.
    REQUIRE(late(i) == (i >= 2));
  }

#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 40, 0)
  // Only on 6.40 is the late member inside the record; before that it is top-level and there is no
  // HLT.m_late to read.
  auto grouped = reader->GetView<bool>("HLT.m_late");
  for (std::size_t i = 0; i < 4; i++) {
    REQUIRE(grouped(i) == (i >= 2));
  }
#endif
}
