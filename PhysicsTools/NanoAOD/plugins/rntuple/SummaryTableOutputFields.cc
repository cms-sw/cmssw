#include "SummaryTableOutputFields.h"

template <typename T, typename Col>
std::vector<RNTupleFieldPtr<T>> SummaryTableOutputFields::makeFields(const std::vector<Col> &tabcols,
                                                                     RNTupleModel &model) {
  std::vector<RNTupleFieldPtr<T>> fields;
  fields.reserve(tabcols.size());
  for (const auto &col : tabcols) {
    // TODO field description
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
    auto data = tabcols[i].values;
    // TODO remove this awful hack when std::int64_t is supported
    // -- turns std::vector<int64_t> into std::vector<uint64_t>
    T casted_data(data.begin(), data.end());
    fields[i].fill(casted_data);
  }
}

SummaryTableOutputFields::SummaryTableOutputFields(const nanoaod::MergeableCounterTable &tab, RNTupleModel &model) {
  // TODO use std::int64_t when supported
  m_intFields = makeFields<std::uint64_t>(tab.intCols(), model);
  m_floatFields = makeFields<double>(tab.floatCols(), model);
  m_floatWithNormFields = makeFields<double>(tab.floatWithNormCols(), model);
  m_vintFields = makeFields<std::vector<std::uint64_t>>(tab.vintCols(), model);
  m_vfloatFields = makeFields<std::vector<double>>(tab.vfloatCols(), model);
  m_vfloatWithNormFields = makeFields<std::vector<double>>(tab.vfloatWithNormCols(), model);
}

void SummaryTableOutputFields::fill(const nanoaod::MergeableCounterTable &tab) {
  fillScalarFields(tab.intCols(), m_intFields);
  fillScalarFields(tab.floatCols(), m_floatFields);
  fillScalarFields(tab.floatWithNormCols(), m_floatWithNormFields);
  fillVectorFields(tab.vintCols(), m_vintFields);
  fillVectorFields(tab.vfloatCols(), m_vfloatFields);
  fillVectorFields(tab.vfloatWithNormCols(), m_vfloatWithNormFields);
}
