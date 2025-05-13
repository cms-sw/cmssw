#include "SummaryTableOutputFields.h"

using ROOT::RNTupleModel;

template <typename T, typename Col>
std::vector<RNTupleFieldPtr<T>> SummaryTableOutputFields::makeFields(const std::vector<Col> &tabcols,
                                                                     RNTupleModel &model) {
  std::vector<RNTupleFieldPtr<T>> fields;
  fields.reserve(tabcols.size());
  for (const auto &col : tabcols) {
    fields.emplace_back(RNTupleFieldPtr<T>(col.name, col.doc, model));
  }
  return fields;
}

template <typename T, typename Col>
void SummaryTableOutputFields::fillScalarFields(const std::vector<Col> &tabcols,
                                                std::vector<RNTupleFieldPtr<T>> fields) {
  if (tabcols.size() != fields.size()) {
    throw cms::Exception("LogicError", "Mismatch in table columns");
  }
  for (std::size_t i = 0; i < tabcols.size(); ++i) {
    if (tabcols[i].name != fields[i].getFieldName()) {
      throw cms::Exception("LogicError", "Mismatch in table columns");
    }
    fields[i].fill(tabcols[i].value);
  }
}

// TODO: maybe we can unify with the function above since now it's the same
template <typename T, typename Col>
void SummaryTableOutputFields::fillVectorFields(const std::vector<Col> &tabcols,
                                                std::vector<RNTupleFieldPtr<T>> fields) {
  if (tabcols.size() != fields.size()) {
    throw cms::Exception("LogicError", "Mismatch in table columns");
  }
  for (std::size_t i = 0; i < tabcols.size(); ++i) {
    if (tabcols[i].name != fields[i].getFieldName()) {
      throw cms::Exception("LogicError", "Mismatch in table columns");
    }
    fields[i].fill(tabcols[i].values);
  }
}

SummaryTableOutputFields::SummaryTableOutputFields(const nanoaod::MergeableCounterTable &tab, RNTupleModel &model) {
  m_intFields = makeFields<int_accumulator>(tab.intCols(), model);
  m_floatFields = makeFields<float_accumulator>(tab.floatCols(), model);
  m_floatWithNormFields = makeFields<float_accumulator>(tab.floatWithNormCols(), model);
  m_vintFields = makeFields<std::vector<int_accumulator>>(tab.vintCols(), model);
  m_vfloatFields = makeFields<std::vector<float_accumulator>>(tab.vfloatCols(), model);
  m_vfloatWithNormFields = makeFields<std::vector<float_accumulator>>(tab.vfloatWithNormCols(), model);
}

void SummaryTableOutputFields::fill(const nanoaod::MergeableCounterTable &tab) {
  fillScalarFields(tab.intCols(), m_intFields);
  fillScalarFields(tab.floatCols(), m_floatFields);
  fillScalarFields(tab.floatWithNormCols(), m_floatWithNormFields);
  fillVectorFields(tab.vintCols(), m_vintFields);
  fillVectorFields(tab.vfloatCols(), m_vfloatFields);
  fillVectorFields(tab.vfloatWithNormCols(), m_vfloatWithNormFields);
}
