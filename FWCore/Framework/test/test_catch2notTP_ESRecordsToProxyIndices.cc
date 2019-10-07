#include "catch.hpp"

#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/Framework/interface/ComponentDescription.h"

#include <memory>
#include <string>
#include <vector>

namespace {
  struct Data1 {};
  struct Data2 {};
  struct Data3 {};
  struct Rcd1 {};
  struct Rcd2 {};
  struct Rcd3 {};
  struct MissingRcd {};
}  // namespace

namespace edm::test {
  class ESTagGetterTester {
  public:
    static std::vector<ESTagGetter::Info> const& info(ESTagGetter const& iGet) { return iGet.lookup_; }
  };
}  // namespace edm::test
using edm::test::ESTagGetterTester;

TYPELOOKUP_DATA_REG(Rcd1);
TYPELOOKUP_DATA_REG(Rcd2);
TYPELOOKUP_DATA_REG(Rcd3);
TYPELOOKUP_DATA_REG(MissingRcd);

TYPELOOKUP_DATA_REG(Data1);
TYPELOOKUP_DATA_REG(Data2);
TYPELOOKUP_DATA_REG(Data3);

using namespace edm::eventsetup;
TEST_CASE("test ESRecordsToProxyIndices", "[ESRecordsToProxyIndices]") {
  DataKey const data1Key{DataKey::makeTypeTag<Data1>(), ""};
  DataKey const data2Key{DataKey::makeTypeTag<Data2>(), "foo"};
  DataKey const data3Key{DataKey::makeTypeTag<Data3>(), ""};
  auto const rcd1Key = EventSetupRecordKey::makeKey<Rcd1>();
  auto const rcd2Key = EventSetupRecordKey::makeKey<Rcd2>();
  auto const rcd3Key = EventSetupRecordKey::makeKey<Rcd3>();
  auto const missingRcdKey = EventSetupRecordKey::makeKey<MissingRcd>();

  auto constexpr kMissingKey = ESRecordsToProxyIndices::missingProxyIndex();

  SECTION("test empty") {
    ESRecordsToProxyIndices empty{{}};

    REQUIRE(kMissingKey == empty.indexInRecord(rcd1Key, data1Key));
    REQUIRE(nullptr == empty.component(rcd1Key, data1Key));
    REQUIRE(empty.makeTagGetter(rcd1Key, data1Key.type()).hasNothingToGet());
  }

  SECTION(" test full") {
    std::vector<EventSetupRecordKey> records = {rcd1Key, rcd2Key, rcd3Key};
    std::sort(records.begin(), records.end());
    ESRecordsToProxyIndices r2pi{records};

    std::vector<DataKey> dataKeys = {data1Key, data2Key, data3Key};
    std::sort(dataKeys.begin(), dataKeys.end());
    //Now fill
    ComponentDescription const p[5] = {
        {"zero", "zero", false, false},
        {"one", "one", false, false},
        {"two", "two", false, false},
        {"three", "three", false, false},
        {"four", "four", false, false},
    };
    std::vector<
        std::pair<EventSetupRecordKey, std::pair<std::vector<DataKey>, std::vector<ComponentDescription const*> > > >
        orderedOfKeys = {{records[0], {{dataKeys[0], dataKeys[1], dataKeys[2]}, {p + 1, p + 2, p + 3}}},
                         {records[1], {{}, {}}},
                         {records[2], {{dataKeys[1]}, {p + 4}}}};

    unsigned int index = 0;
    for (auto const& pr : orderedOfKeys) {
      REQUIRE(index + 1 == r2pi.dataKeysInRecord(index, pr.first, pr.second.first, pr.second.second));
      ++index;
    }
    for (auto const& pr : orderedOfKeys) {
      index = 0;
      auto it = pr.second.second.begin();
      for (auto const& dk : pr.second.first) {
        REQUIRE(index == r2pi.indexInRecord(pr.first, dk).value());
        REQUIRE(*it == r2pi.component(pr.first, dk));
        ++index;
        ++it;
      }
    }
    {
      auto v = ESTagGetterTester::info(r2pi.makeTagGetter(records[0], dataKeys[0].type()));
      REQUIRE(v.size() == 1);
      REQUIRE(v.front().index_ == r2pi.indexInRecord(records[0], dataKeys[0]));
      REQUIRE(v.front().productLabel_ == dataKeys[0].name().value());
      REQUIRE(v.front().moduleLabel_ == (p + 1)->label_);
    }
    {
      auto v = ESTagGetterTester::info(r2pi.makeTagGetter(records[0], dataKeys[1].type()));
      REQUIRE(v.size() == 1);
      REQUIRE(v.front().index_ == r2pi.indexInRecord(records[0], dataKeys[1]));
      REQUIRE(v.front().productLabel_ == dataKeys[1].name().value());
      REQUIRE(v.front().moduleLabel_ == (p + 2)->label_);
    }
    {
      auto v = ESTagGetterTester::info(r2pi.makeTagGetter(records[0], dataKeys[2].type()));
      REQUIRE(v.size() == 1);
      REQUIRE(v.front().index_ == r2pi.indexInRecord(records[0], dataKeys[2]));
      REQUIRE(v.front().productLabel_ == dataKeys[2].name().value());
      REQUIRE(v.front().moduleLabel_ == (p + 3)->label_);
    }

    REQUIRE(kMissingKey == r2pi.indexInRecord(missingRcdKey, dataKeys[0]));
    REQUIRE(kMissingKey == r2pi.indexInRecord(records[1], dataKeys[0]));
    REQUIRE(kMissingKey == r2pi.indexInRecord(records[2], dataKeys[0]));

    REQUIRE(r2pi.makeTagGetter(missingRcdKey, dataKeys[0].type()).hasNothingToGet());
    REQUIRE(r2pi.makeTagGetter(records[1], dataKeys[0].type()).hasNothingToGet());
    REQUIRE(r2pi.makeTagGetter(records[2], dataKeys[0].type()).hasNothingToGet());
  }
}
