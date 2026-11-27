#include "SummaryTableOutputFields.h"

using ROOT::RNTupleModel;

template <typename T, typename Col>
std::vector<RNTupleFieldPtr<T>> SummaryTableOutputFields::makeFields(const std::vector<Col> &tabcols,
                                                                     RNTupleModel &model) {
  std::vector<RNTupleFieldPtr<T>> fields;
  fields.reserve(tabcols.size());
  for (const auto &col : tabcols) {
    fields.emplace_back(col.name, col.doc, model);
  }
  return fields;
}

template <typename T, typename Col>
void SummaryTableOutputFields::fillFields(const std::vector<Col> &tabcols, std::vector<RNTupleFieldPtr<T>> &fields) {
  if (tabcols.size() != fields.size()) {
    throw cms::Exception("LogicError", "Mismatch in table columns");
  }
  for (std::size_t i = 0; i < tabcols.size(); ++i) {
    if (tabcols[i].name != fields[i].getFieldName()) {
      throw cms::Exception("LogicError", "Mismatch in table columns");
    }
    // MergeableCounterTable spells a scalar column's payload `value` and a vector column's `values`
    if constexpr (requires { tabcols[i].value; }) {
      fields[i].fill(tabcols[i].value);
    } else {
      fields[i].fill(tabcols[i].values);
    }
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
  fillFields(tab.intCols(), m_intFields);
  fillFields(tab.floatCols(), m_floatFields);
  fillFields(tab.floatWithNormCols(), m_floatWithNormFields);
  fillFields(tab.vintCols(), m_vintFields);
  fillFields(tab.vfloatCols(), m_vfloatFields);
  fillFields(tab.vfloatWithNormCols(), m_vfloatWithNormFields);
}
