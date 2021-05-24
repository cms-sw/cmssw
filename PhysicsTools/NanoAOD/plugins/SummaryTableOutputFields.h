#ifndef PhysicsTools_NanoAOD_SummaryTableOutputFields_h
#define PhysicsTools_NanoAOD_SummaryTableOutputFields_h

#include "DataFormats/NanoAOD/interface/MergeableCounterTable.h"

#include "RNTupleFieldPtr.h"

class SummaryTableOutputFields {
public:
  SummaryTableOutputFields() = default;
  SummaryTableOutputFields(const nanoaod::MergeableCounterTable &tab, RNTupleModel &model);
  void fill(const nanoaod::MergeableCounterTable &tab);

private:
  template <typename T, typename Col>
  std::vector<RNTupleFieldPtr<T>> makeFields(const std::vector<Col> &tabcols, RNTupleModel &model);
  template <typename T, typename Col>
  static void fillScalarFields(const std::vector<Col> &tabcols, std::vector<RNTupleFieldPtr<T>> fields);
  template <typename T, typename Col>
  static void fillVectorFields(const std::vector<Col> &tabcols, std::vector<RNTupleFieldPtr<T>> fields);

  std::vector<RNTupleFieldPtr<std::uint64_t>> m_intFields;
  std::vector<RNTupleFieldPtr<double>> m_floatFields;
  std::vector<RNTupleFieldPtr<double>> m_floatWithNormFields;
  std::vector<RNTupleFieldPtr<std::vector<double>>> m_vfloatFields;
  std::vector<RNTupleFieldPtr<std::vector<double>>> m_vfloatWithNormFields;
  std::vector<RNTupleFieldPtr<std::vector<std::uint64_t>>> m_vintFields;
};

#endif
