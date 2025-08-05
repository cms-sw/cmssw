#ifndef PhysicsTools_NanoAOD_SummaryTableOutputFields_h
#define PhysicsTools_NanoAOD_SummaryTableOutputFields_h

#include "DataFormats/NanoAOD/interface/MergeableCounterTable.h"

#include "RNTupleFieldPtr.h"

class SummaryTableOutputFields {
public:
  SummaryTableOutputFields() = default;
  SummaryTableOutputFields(const nanoaod::MergeableCounterTable &tab, ROOT::RNTupleModel &model);
  void fill(const nanoaod::MergeableCounterTable &tab);

private:
  template <typename T, typename Col>
  std::vector<RNTupleFieldPtr<T>> makeFields(const std::vector<Col> &tabcols, ROOT::RNTupleModel &model);
  template <typename T, typename Col>
  static void fillScalarFields(const std::vector<Col> &tabcols, std::vector<RNTupleFieldPtr<T>> fields);
  template <typename T, typename Col>
  static void fillVectorFields(const std::vector<Col> &tabcols, std::vector<RNTupleFieldPtr<T>> fields);

  using int_accumulator = nanoaod::MergeableCounterTable::int_accumulator;
  using float_accumulator = nanoaod::MergeableCounterTable::float_accumulator;

  std::vector<RNTupleFieldPtr<int_accumulator>> m_intFields;
  std::vector<RNTupleFieldPtr<float_accumulator>> m_floatFields;
  std::vector<RNTupleFieldPtr<float_accumulator>> m_floatWithNormFields;
  std::vector<RNTupleFieldPtr<std::vector<float_accumulator>>> m_vfloatFields;
  std::vector<RNTupleFieldPtr<std::vector<float_accumulator>>> m_vfloatWithNormFields;
  std::vector<RNTupleFieldPtr<std::vector<int_accumulator>>> m_vintFields;
};

#endif
