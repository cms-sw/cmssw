#ifndef DataFormats_NanoAOD_FlatTable_h
#define DataFormats_NanoAOD_FlatTable_h

#include "DataFormats/Math/interface/libminifloat.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Span.h"

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
      template <typename Span>
      inline void bulk(Span const &data) const {}
    };
    template <>
    struct MaybeMantissaReduce<float> {
      int bits_;
      MaybeMantissaReduce(int mantissaBits) : bits_(mantissaBits) {}
      inline float one(const float &val) const {
        return (bits_ > 0 ? MiniFloatConverter::reduceMantissaToNbitsRounding(val, bits_) : val);
      }
      template <typename Span>
      inline void bulk(Span &&data) const {
        if (bits_ > 0)
          MiniFloatConverter::reduceMantissaToNbitsRounding(bits_, data.begin(), data.end(), data.begin());
      }
    };
  }  // namespace flatTableHelper

  class FlatTable {
  public:
    enum class ColumnType {
      Int8,
      UInt8,
      Int16,
      UInt16,
      Int32,
      UInt32,
      Bool,
      Float,
      Double,
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
    auto columnData(unsigned int column) const {
      auto begin = beginData<T>(column);
      return edm::Span(begin, begin + size_);
    }

    /// get a column by index (non-const)
    template <typename T>
    auto columnData(unsigned int column) {
      auto begin = beginData<T>(column);
      return edm::Span(begin, begin + size_);
    }

    /// get a column value for singleton (const)
    template <typename T>
    const auto &columValue(unsigned int column) const {
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
      auto &vec = bigVector<T>();
      columns_.emplace_back(name, docString, defaultColumnType<T>(), vec.size());
      vec.insert(vec.end(), values.begin(), values.end());
      flatTableHelper::MaybeMantissaReduce<T>(mantissaBits).bulk(columnData<T>(columns_.size() - 1));
    }

    template <typename T, typename C>
    void addColumnValue(const std::string &name, const C &value, const std::string &docString, int mantissaBits = -1) {
      if (!singleton())
        throw cms::Exception("LogicError", "addColumnValue works only for singleton tables");
      if (columnIndex(name) != -1)
        throw cms::Exception("LogicError", "Duplicated column: " + name);
      auto &vec = bigVector<T>();
      columns_.emplace_back(name, docString, defaultColumnType<T>(), vec.size());
      vec.push_back(flatTableHelper::MaybeMantissaReduce<T>(mantissaBits).one(value));
    }

    void addExtension(const FlatTable &extension);

    template <class T>
    struct dependent_false : std::false_type {};
    template <typename T>
    static ColumnType defaultColumnType() {
      if constexpr (std::is_same<T, int8_t>())
        return ColumnType::Int8;
      else if constexpr (std::is_same<T, uint8_t>())
        return ColumnType::UInt8;
      else if constexpr (std::is_same<T, int16_t>())
        return ColumnType::Int16;
      else if constexpr (std::is_same<T, uint16_t>())
        return ColumnType::UInt16;
      else if constexpr (std::is_same<T, int32_t>())
        return ColumnType::Int32;
      else if constexpr (std::is_same<T, uint32_t>())
        return ColumnType::UInt32;
      else if constexpr (std::is_same<T, bool>())
        return ColumnType::Bool;
      else if constexpr (std::is_same<T, float>())
        return ColumnType::Float;
      else if constexpr (std::is_same<T, double>())
        return ColumnType::Double;
      else
        static_assert(dependent_false<T>::value, "unsupported type");
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
    auto beginData(unsigned int column) const {
      return bigVector<T>().cbegin() + columns_[column].firstIndex;
    }
    template <typename T>
    auto beginData(unsigned int column) {
      return bigVector<T>().begin() + columns_[column].firstIndex;
    }

    template <typename T>
    auto const &bigVector() const {
      return bigVectorImpl<T>(*this);
    }
    template <typename T>
    auto &bigVector() {
      return bigVectorImpl<T>(*this);
    }

    template <typename T, class This>
    static auto &bigVectorImpl(This &table) {
      // helper function to avoid code duplication, for the two accessor functions that differ only in const-ness
      if constexpr (std::is_same<T, int8_t>())
        return table.int8s_;
      else if constexpr (std::is_same<T, uint8_t>())
        return table.uint8s_;
      else if constexpr (std::is_same<T, int16_t>())
        return table.int16s_;
      else if constexpr (std::is_same<T, uint16_t>())
        return table.uint16s_;
      else if constexpr (std::is_same<T, int32_t>())
        return table.int32s_;
      else if constexpr (std::is_same<T, uint32_t>())
        return table.uint32s_;
      else if constexpr (std::is_same<T, bool>())
        return table.uint8s_;  // special case: bool stored as vector of uint8
      else if constexpr (std::is_same<T, float>())
        return table.floats_;
      else if constexpr (std::is_same<T, double>())
        return table.doubles_;
      else
        static_assert(dependent_false<T>::value, "unsupported type");
    }

    unsigned int size_;
    std::string name_, doc_;
    bool singleton_, extension_;
    std::vector<Column> columns_;
    std::vector<int8_t> int8s_;
    std::vector<uint8_t> uint8s_;
    std::vector<int16_t> int16s_;
    std::vector<uint16_t> uint16s_;
    std::vector<int32_t> int32s_;
    std::vector<uint32_t> uint32s_;
    std::vector<float> floats_;
    std::vector<double> doubles_;
  };

}  // namespace nanoaod

#endif
