#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalESProducerTools_h
#define RecoLocalCalo_HGCalRecAlgos_HGCalESProducerTools_h

#include "FWCore/Utilities/interface/Exception.h"
#include <string>
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;  // ordered_json preserves key insertion order

namespace hgcal {

  std::string search_modkey(const std::string& module, const json& data, const std::string& name);
  std::string search_fedkey(const int& fedid, const json& data, const std::string& name);
  bool check_keys(const json& data,
                  const std::string& firstkey,
                  const std::vector<std::string>& keys,
                  const std::string& fname);
  bool check_keys(const json& data, const std::vector<std::string>& keys, const std::string& fname);

  // @short fill full SoA column with single value
  template <typename T>
  void fill_SoA_column_single(T* column_SoA, const float& value, const int offset, const int nrows) {
    std::fill(column_SoA + offset, column_SoA + offset + nrows, value);
  }

  // @short fill SoA column with data from vector for any type with some offset
  template <typename T>
  void fill_SoA_column(
      std::span<T> column_SoA, const std::vector<T>& values, const int offset, const int nrows, int arr_offset = 0) {
    const int nrows_vals = values.size();
    if (arr_offset < 0) {
      arr_offset = 0;
      if (nrows_vals != arr_offset + nrows) {
        cms::Exception ex("InvalidData");
        ex << " Expected " << nrows << " rows, but got " << nrows_vals << "!";
        ex.addContext("Calling hgcal::fill_SoA_eigen_row()");
        throw ex;
      }
    } else if (nrows_vals < arr_offset + nrows) {
      cms::Exception ex("InvalidData");
      ex << " Tried to copy " << nrows << " rows to SoA with offset " << arr_offset << ", but only have " << nrows_vals
         << " values in JSON!";
      ex.addContext("Calling hgcal::fill_SoA_eigen_row()");
      throw ex;
    }
    auto begin = values.begin() + arr_offset;
    auto end = (begin + nrows > values.end()) ? values.end() : begin + nrows;
    std::copy(begin, end, &column_SoA[offset]);
  }

  // @short fill full SoA column with data from vector for any type
  template <typename T, typename P>
  void fill_SoA_eigen_row(P& soa, const std::vector<std::vector<T>>& values, const size_t row) {
    if (row >= values.size()) {
      cms::Exception ex("InvalidData");
      ex << " Tried to copy row " << row << " to SoA, but only have " << values.size() << " values in JSON!";
      ex.addContext("Calling hgcal::fill_SoA_eigen_row()");
      throw ex;
    }
    if (!values.empty() && int(values[row].size()) != soa.size()) {
      cms::Exception ex("InvalidData");
      ex << " Expected " << soa.size() << " elements in Eigen vector, but got " << values[row].size() << "!";
      ex.addContext("Calling hgcal::fill_SoA_eigen_row()");
      throw ex;
    }
    for (int i = 0; i < soa.size(); i++)
      soa(i) = values[row][i];
  }

}  // namespace hgcal

#endif
