#ifndef PhysicsTools_NanoAOD_RNTupleCollection_h
#define PhysicsTools_NanoAOD_RNTupleCollection_h

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Common/interface/Handle.h"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/REntry.hxx>

class RNTupleCollection {
public:
  RNTupleCollection() = delete;
  RNTupleCollection(const std::string& name,
                    const std::string& desc,
                    std::vector<edm::Handle<nanoaod::FlatTable>>& tables,
                    ROOT::RNTupleModel& model);

  const std::string& getFieldName() const { return m_name; }

  void bindBuffer(ROOT::RNTupleModel& model);
  void fill(std::vector<edm::Handle<nanoaod::FlatTable>>& tables);

private:
  std::string m_name;
  std::size_t m_record_size;
  std::vector<std::size_t> m_record_offsets;
  std::vector<unsigned char> m_buffer;
};

#endif
