#ifndef PhysicsTools_NanoAOD_RNTupleCollection_h
#define PhysicsTools_NanoAOD_RNTupleCollection_h

#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/REntry.hxx>

struct RNTupleSubfieldDescription {
  std::string m_name;
  std::string m_desc;
  nanoaod::FlatTable::ColumnType m_type;
};

class RNTupleCollection {
public:
  RNTupleCollection() = delete;
  RNTupleCollection(const std::string& name,
                    const std::string& desc,
                    std::vector<RNTupleSubfieldDescription>& subfields_desc,
                    ROOT::RNTupleModel& model);

  void bind_entry(ROOT::REntry& entry);
  const std::string& getFieldName() const { return m_name; }

private:
  std::string m_name;
  std::size_t m_record_size;
  std::vector<std::size_t> m_record_offsets;
  std::vector<unsigned char> m_buffer;
};

#endif
