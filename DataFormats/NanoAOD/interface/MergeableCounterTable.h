#ifndef DataFormats_NanoAOD_MergeableCounterTable_h
#define DataFormats_NanoAOD_MergeableCounterTable_h

#include "FWCore/Utilities/interface/Exception.h"
#include <vector>
#include <string>
#include <algorithm>

namespace nanoaod {

  class MergeableCounterTable {
  public:
    MergeableCounterTable() {}
    typedef long long int_accumulator;  // we accumulate in long long int, to avoid overflow
    typedef double float_accumulator;   // we accumulate in double, to preserve precision

    template <typename T>
    struct SingleColumn {
      typedef T value_type;
      SingleColumn() {}
      SingleColumn(const std::string& aname, const std::string& adoc, T avalue = T())
          : name(aname), doc(adoc), value(avalue) {}
      std::string name, doc;
      T value;
      void operator+=(const SingleColumn<T>& other) {
        if (!compatible(other))
          throw cms::Exception("LogicError",
                               "Trying to merge " + name + " with " + other.name + " failed compatibility test.\n");
        value += other.value;
      }
      bool compatible(const SingleColumn<T>& other) {
        return name == other.name;  // we don't check the doc, not needed
      }
    };
    typedef SingleColumn<float_accumulator> FloatColumn;
    typedef SingleColumn<int_accumulator> IntColumn;

    template <typename T>
    struct SingleWithNormColumn : SingleColumn<T> {
      SingleWithNormColumn() { norm = 0; }
      SingleWithNormColumn(const std::string& aname, const std::string& adoc, T avalue = T(), const double anorm = 0)
          : SingleColumn<T>(aname, adoc, avalue), norm(anorm) {}
      double norm;
      void operator+=(const SingleWithNormColumn<T>& other) {
        if (!this->compatible(other))
          throw cms::Exception(
              "LogicError", "Trying to merge " + this->name + " with " + other.name + " failed compatibility test.\n");
        auto newNorm = norm + other.norm;
        this->value = (newNorm != 0) ? (this->value * norm + other.value * other.norm) / newNorm : 0;
        norm = newNorm;
      }
    };
    typedef SingleWithNormColumn<float_accumulator> FloatWithNormColumn;

    template <typename T>
    struct VectorColumn {
      typedef T element_type;
      VectorColumn() {}
      VectorColumn(const std::string& aname, const std::string& adoc, unsigned int size)
          : name(aname), doc(adoc), values(size, T()) {}
      VectorColumn(const std::string& aname, const std::string& adoc, const std::vector<T>& somevalues)
          : name(aname), doc(adoc), values(somevalues) {}
      std::string name, doc;
      std::vector<T> values;
      void operator+=(const VectorColumn<T>& other) {
        if (!compatible(other))
          throw cms::Exception("LogicError",
                               "Trying to merge " + name + " with " + other.name + " failed compatibility test.\n");
        for (unsigned int i = 0, n = values.size(); i < n; ++i) {
          values[i] += other.values[i];
        }
      }
      bool compatible(const VectorColumn<T>& other) {
        return name == other.name && values.size() == other.values.size();  // we don't check the doc, not needed
      }
    };
    typedef VectorColumn<float_accumulator> VFloatColumn;
    typedef VectorColumn<int_accumulator> VIntColumn;

    template <typename T>
    struct VectorWithNormColumn : VectorColumn<T> {
      double norm;
      VectorWithNormColumn() { norm = 0; }
      VectorWithNormColumn(const std::string& aname, const std::string& adoc, unsigned int size, double anorm = 0)
          : VectorColumn<T>(aname, adoc, size), norm(anorm) {}
      VectorWithNormColumn(const std::string& aname,
                           const std::string& adoc,
                           const std::vector<T>& somevalues,
                           double anorm = 0)
          : VectorColumn<T>(aname, adoc, somevalues), norm(anorm) {}
      void operator+=(const VectorWithNormColumn<T>& other) {
        if (!this->compatible(other))
          throw cms::Exception(
              "LogicError", "Trying to merge " + this->name + " with " + other.name + " failed compatibility test.\n");
        auto newNorm = norm + other.norm;
        for (unsigned int i = 0, n = this->values.size(); i < n; ++i) {
          this->values[i] =
              (newNorm != 0) ? (this->values[i] * norm + other.values[i] * other.norm) / (norm + other.norm) : 0;
        }
        norm = newNorm;
      }
    };
    typedef VectorWithNormColumn<float_accumulator> VFloatWithNormColumn;

    const std::vector<FloatColumn>& floatCols() const { return floatCols_; }
    const std::vector<VFloatColumn>& vfloatCols() const { return vfloatCols_; }
    const std::vector<FloatWithNormColumn>& floatWithNormCols() const { return floatWithNormCols_; }
    const std::vector<VFloatWithNormColumn>& vfloatWithNormCols() const { return vfloatWithNormCols_; }
    const std::vector<IntColumn>& intCols() const { return intCols_; }
    const std::vector<VIntColumn>& vintCols() const { return vintCols_; }

    template <typename F>
    void addFloat(const std::string& name, const std::string& doc, F value) {
      floatCols_.push_back(FloatColumn(name, doc, value));
    }

    template <typename F>
    void addFloatWithNorm(const std::string& name, const std::string& doc, F value, double norm) {
      floatWithNormCols_.push_back(FloatWithNormColumn(name, doc, value, norm));
    }

    template <typename I>
    void addInt(const std::string& name, const std::string& doc, I value) {
      intCols_.push_back(IntColumn(name, doc, value));
    }

    template <typename F>
    void addVFloat(const std::string& name, const std::string& doc, const std::vector<F> values) {
      vfloatCols_.push_back(VFloatColumn(name, doc, values.size()));
      std::copy(values.begin(), values.end(), vfloatCols_.back().values.begin());
    }

    template <typename F>
    void addVFloatWithNorm(const std::string& name, const std::string& doc, const std::vector<F> values, double norm) {
      vfloatWithNormCols_.push_back(VFloatWithNormColumn(name, doc, values.size(), norm));
      std::copy(values.begin(), values.end(), vfloatWithNormCols_.back().values.begin());
    }

    template <typename I>
    void addVInt(const std::string& name, const std::string& doc, const std::vector<I> values) {
      vintCols_.push_back(VIntColumn(name, doc, values.size()));
      std::copy(values.begin(), values.end(), vintCols_.back().values.begin());
    }

    bool mergeProduct(const MergeableCounterTable& other) {
      if (!tryMerge(intCols_, other.intCols_))
        return false;
      if (!tryMerge(vintCols_, other.vintCols_))
        return false;
      if (!tryMerge(floatCols_, other.floatCols_))
        return false;
      if (!tryMerge(vfloatCols_, other.vfloatCols_))
        return false;
      if (!tryMerge(floatWithNormCols_, other.floatWithNormCols_))
        return false;
      if (!tryMerge(vfloatWithNormCols_, other.vfloatWithNormCols_))
        return false;
      return true;
    }

    void swap(MergeableCounterTable& iOther) {
      floatCols_.swap(iOther.floatCols_);
      vfloatCols_.swap(iOther.vfloatCols_);
      floatWithNormCols_.swap(iOther.floatWithNormCols_);
      vfloatWithNormCols_.swap(iOther.vfloatWithNormCols_);
      intCols_.swap(iOther.intCols_);
      vintCols_.swap(iOther.vintCols_);
    }

  private:
    std::vector<FloatColumn> floatCols_;
    std::vector<VFloatColumn> vfloatCols_;
    std::vector<FloatWithNormColumn> floatWithNormCols_;
    std::vector<VFloatWithNormColumn> vfloatWithNormCols_;
    std::vector<IntColumn> intCols_;
    std::vector<VIntColumn> vintCols_;

    template <typename T>
    bool tryMerge(std::vector<T>& one, const std::vector<T>& two) {
      for (auto y : two) {
        auto x = std::find_if(one.begin(), one.end(), [&y](const T& x) { return x.name == y.name; });
        if (x == one.end())
          one.push_back(y);
        else
          (*x) += y;
      }
      return true;
    }
  };

}  // namespace nanoaod

#endif
