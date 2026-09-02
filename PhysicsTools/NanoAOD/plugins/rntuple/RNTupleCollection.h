#ifndef PhysicsTools_NanoAOD_RNTupleCollection_h
#define PhysicsTools_NanoAOD_RNTupleCollection_h

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Common/interface/Handle.h"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/REntry.hxx>

// A named FlatTable and its extensions written as one field: an untyped std::vector<record> whose
// members are the columns, or -- for a singleton table, which carries exactly one row per entry --
// the untyped record on its own. The TTree module makes the same distinction: singleton tables
// become scalar branches, Generator_binvar and not nGenerator plus an array.
class RNTupleCollection {
public:
  RNTupleCollection() = delete;
  RNTupleCollection(const std::string& name,
                    const std::string& desc,
                    std::vector<edm::Handle<nanoaod::FlatTable>>& tables,
                    ROOT::RNTupleModel& model,
                    bool singleton);

  const std::string& getFieldName() const { return m_name; }

  // Give every column the flat name the TTree module uses for it as well: GenJet_pt beside
  // GenJet.pt. See rntupleprojection.
  void addProjections(ROOT::RNTupleModel& model) const;

  void bindBuffer(ROOT::REntry& entry);
  void fill(std::vector<edm::Handle<nanoaod::FlatTable>>& tables);

private:
  std::string m_name;
  bool m_singleton;
  std::size_t m_record_size;
  std::vector<std::size_t> m_record_offsets;
  std::vector<unsigned char> m_buffer;
};

#endif
