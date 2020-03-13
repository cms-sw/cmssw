#ifndef DataFormats_NanoAOD_FlatTable_h
#define DataFormats_NanoAOD_FlatTable_h

#include "DataFormats/Math/interface/libminifloat.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <boost/range/sub_range.hpp>

#include <cstdint>
#include <vector>
#include <string>

#include <type_traits>

namespace nanoaod {

  namespace flatTableHelper {
    template <typename T>
    struct MaybeMantissaReduce {
      MaybeMantissaReduce(int mantissaBits) {}
      inline T one(const T &val) const { return val; }
      inline void bulk(boost::sub_range<std::vector<T>> data) const {}
    };
    template <>
    struct MaybeMantissaReduce<float> {
      int bits_;
      MaybeMantissaReduce(int mantissaBits) : bits_(mantissaBits) {}
      inline float one(const float &val) const {
        return (bits_ > 0 ? MiniFloatConverter::reduceMantissaToNbitsRounding(val, bits_) : val);
      }
      inline void bulk(boost::sub_range<std::vector<float>> data) const {
        if (bits_ > 0)
          MiniFloatConverter::reduceMantissaToNbitsRounding(bits_, data.begin(), data.end(), data.begin());
      }
    };
  }  // namespace flatTableHelper

  class FlatTable {
  public:
    enum class ColumnType {
      Float,
      Int,
      UInt8,
      Bool
    };  // We could have other Float types with reduced mantissa, and similar

    FlatTable() : size_(0) {}
    FlatTable(unsigned int size, const std::string &name, bool singleton, bool extension = false)
        : size_(size), name_(name), singleton_(singleton), extension_(extension) {}
    ~FlatTable() {}

    unsigned int nColumns() const { return columns_.size(); };
    unsigned int nRows() const { return size_; };
    unsigned int size() const { return size_; }
    bool singleton() const { return singleton_; }
    bool extension() const { return extension_; }
    const std::string &name() const { return name_; }

    const std::string &columnName(unsigned int col) const { return columns_[col].name; }
    int columnIndex(const std::string &name) const;

    ColumnType columnType(unsigned int col) const { return columns_[col].type; }

    void setDoc(const std::string &doc) { doc_ = doc; }
    const std::string &doc() const { return doc_; }
    const std::string &columnDoc(unsigned int col) const { return columns_[col].doc; }

    /// get a column by index (const)
    template <typename T>
    boost::sub_range<const std::vector<T>> columnData(unsigned int column) const {
      auto begin = beginData<T>(column);
      return boost::sub_range<const std::vector<T>>(begin, begin + size_);
    }

    /// get a column by index (non-const)
    template <typename T>
    boost::sub_range<std::vector<T>> columnData(unsigned int column) {
      auto begin = beginData<T>(column);
      return boost::sub_range<std::vector<T>>(begin, begin + size_);
    }

    /// get a column value for singleton (const)
    template <typename T>
    const T &columValue(unsigned int column) const {
      if (!singleton())
        throw cms::Exception("LogicError", "columnValue works only for singleton tables");
      return *beginData<T>(column);
    }

    double getAnyValue(unsigned int row, unsigned int column) const;

    class RowView {
    public:
      RowView() {}
      RowView(const FlatTable &table, unsigned int row) : table_(&table), row_(row) {}
      double getAnyValue(unsigned int column) const { return table_->getAnyValue(row_, column); }
      double getAnyValue(const std::string &column) const {
        return table_->getAnyValue(row_, table_->columnIndex(column));
      }
      const FlatTable &table() const { return *table_; }
      unsigned int row() const { return row_; }

    private:
      const FlatTable *table_;
      unsigned int row_;
    };
    RowView row(unsigned int row) const { return RowView(*this, row); }

    template <typename T, typename C>
    void addColumn(const std::string &name, const C &values, const std::string &docString, int mantissaBits = -1) {
      if (columnIndex(name) != -1)
        throw cms::Exception("LogicError", "Duplicated column: " + name);
      if (values.size() != size())
        throw cms::Exception("LogicError", "Mismatched size for " + name);
      if constexpr (std::is_same<T, bool>()) {
        columns_.emplace_back(name, docString, ColumnType::Bool, uint8s_.size());
        uint8s_.insert(uint8s_.end(), values.begin(), values.end());
      } else if constexpr (std::is_same<T, float>()) {
        columns_.emplace_back(name, docString, ColumnType::Float, floats_.size());
        floats_.insert(floats_.end(), values.begin(), values.end());
        flatTableHelper::MaybeMantissaReduce<float>(mantissaBits).bulk(columnData<float>(columns_.size() - 1));
      } else {
        ColumnType type = defaultColumnType<T>();
        auto &vec = bigVector<T>();
        columns_.emplace_back(name, docString, type, vec.size());
        vec.insert(vec.end(), values.begin(), values.end());
      }
    }

    template <typename T, typename C>
    void addColumnValue(const std::string &name, const C &value, const std::string &docString, int mantissaBits = -1) {
      if (!singleton())
        throw cms::Exception("LogicError", "addColumnValue works only for singleton tables");
      if (columnIndex(name) != -1)
        throw cms::Exception("LogicError", "Duplicated column: " + name);
      if constexpr (std::is_same<T, bool>()) {
        columns_.emplace_back(name, docString, ColumnType::Bool, uint8s_.size());
        uint8s_.push_back(value);
      } else if constexpr (std::is_same<T, float>()) {
        columns_.emplace_back(name, docString, ColumnType::Float, floats_.size());
        floats_.push_back(flatTableHelper::MaybeMantissaReduce<float>(mantissaBits).one(value));
      } else {
        ColumnType type = defaultColumnType<T>();
        auto &vec = bigVector<T>();
        columns_.emplace_back(name, docString, type, vec.size());
        vec.push_back(value);
      }
    }

    void addExtension(const FlatTable &extension);

    template <typename T>
    static ColumnType defaultColumnType() {
      if constexpr (std::is_same<T, int>())
        return ColumnType::Int;
      if constexpr (std::is_same<T, uint8_t>())
        return ColumnType::UInt8;
      throw cms::Exception("unsupported type");
    }

    // this below needs to be public for ROOT, but it is to be considered private otherwise
    struct Column {
      std::string name, doc;
      ColumnType type;
      unsigned int firstIndex;
      Column() {}  // for ROOT
      Column(const std::string &aname, const std::string &docString, ColumnType atype, unsigned int anIndex)
          : name(aname), doc(docString), type(atype), firstIndex(anIndex) {}
    };

  private:
    template <typename T>
    typename std::vector<T>::const_iterator beginData(unsigned int column) const {
      const Column &col = columns_[column];
      return bigVector<T>().begin() + col.firstIndex;
    }
    template <typename T>
    typename std::vector<T>::iterator beginData(unsigned int column) {
      const Column &col = columns_[column];
      return bigVector<T>().begin() + col.firstIndex;
    }

    template <typename T>
    const std::vector<T> &bigVector() const {
      throw cms::Exception("unsupported type");
    }
    template <typename T>
    std::vector<T> &bigVector() {
      throw cms::Exception("unsupported type");
    }

    unsigned int size_;
    std::string name_, doc_;
    bool singleton_, extension_;
    std::vector<Column> columns_;
    std::vector<float> floats_;
    std::vector<int> ints_;
    std::vector<uint8_t> uint8s_;
  };

  template <>
  inline const std::vector<float> &FlatTable::bigVector<float>() const {
    return floats_;
  }
  template <>
  inline const std::vector<int> &FlatTable::bigVector<int>() const {
    return ints_;
  }
  template <>
  inline const std::vector<uint8_t> &FlatTable::bigVector<uint8_t>() const {
    return uint8s_;
  }
  template <>
  inline std::vector<float> &FlatTable::bigVector<float>() {
    return floats_;
  }
  template <>
  inline std::vector<int> &FlatTable::bigVector<int>() {
    return ints_;
  }
  template <>
  inline std::vector<uint8_t> &FlatTable::bigVector<uint8_t>() {
    return uint8s_;
  }

}  // namespace nanoaod

#endif
