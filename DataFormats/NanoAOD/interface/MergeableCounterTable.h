#ifndef DataFormats_NanoAOD_MergeableCounterTable_h
#define DataFormats_NanoAOD_MergeableCounterTable_h

#include "FWCore/Utilities/interface/Exception.h"
#include <vector>
#include <string>

namespace nanoaod {

class MergeableCounterTable {
    public:
        MergeableCounterTable() {}
        typedef long long int_accumulator; // we accumulate in long long int, to avoid overflow
        typedef double float_accumulator; // we accumulate in double, to preserve precision

        template<typename T>
        struct SingleColumn {
            typedef T value_type;
            SingleColumn() {}
            SingleColumn(const std::string & aname, const std::string & adoc, T avalue = T()) : name(aname), doc(adoc), value(avalue) {}
            std::string name, doc;
            T value;
            void operator+=(const SingleColumn<T> & other) {
                //// if one arrives here from tryMerge the checks are already done in the compatible() function before.
                //// you may however want to enable these and remove the 'return false' in tryMerge in order to see what's incompatible between the tables.
                //if (name != other.name) throw cms::Exception("LogicError", "Trying to merge "+name+" with "+other.name+"\n");
                value += other.value;
            }
            bool compatible(const SingleColumn<T> & other) {
                return name == other.name;  // we don't check the doc, not needed
            }
        };
        typedef SingleColumn<float_accumulator> FloatColumn; 
        typedef SingleColumn<int_accumulator> IntColumn;  
       
        template<typename T>
        struct VectorColumn {
            typedef T element_type;
            VectorColumn() {}
            VectorColumn(const std::string & aname, const std::string & adoc, unsigned int size) : name(aname), doc(adoc), values(size, T()) {}
            VectorColumn(const std::string & aname, const std::string & adoc, const std::vector<T> & somevalues) : name(aname), doc(adoc), values(somevalues) {}
            std::string name, doc;
            std::vector<T> values;
            void operator+=(const VectorColumn<T> & other) {
                //// if one arrives here from tryMerge the checks are already done in the compatible() function before.
                //// you may however want to enable these and remove the 'return false' in tryMerge in order to see what's incompatible between the tables.
                //if (name != other.name) throw cms::Exception("LogicError", "Trying to merge "+name+" with "+other.name+"\n");
                //if (values.size() != other.values.size()) throw cms::Exception("LogicError", "Trying to merge "+name+" with different number of values!\n");
                for (unsigned int i = 0, n = values.size(); i < n; ++i) {
                    values[i] += other.values[i];
                }
            }
            bool compatible(const VectorColumn<T> & other) {
                return name == other.name && values.size() == other.values.size(); // we don't check the doc, not needed
            }
        };
        typedef VectorColumn<float_accumulator> VFloatColumn; 
        typedef VectorColumn<int_accumulator> VIntColumn;
 
        const std::vector<FloatColumn> & floatCols() const { return floatCols_; }
        const std::vector<VFloatColumn> & vfloatCols() const { return vfloatCols_; }
        const std::vector<IntColumn> & intCols() const { return intCols_; }
        const std::vector<VIntColumn> & vintCols() const { return vintCols_; }
       
        template<typename F>
        void addFloat(const std::string & name, const std::string & doc, F value) { floatCols_.push_back(FloatColumn(name, doc, value)); } 

        template<typename I>
        void addInt(const std::string & name, const std::string & doc, I value) { intCols_.push_back(IntColumn(name, doc, value)); } 

        template<typename F>
        void addVFloat(const std::string & name, const std::string & doc, const std::vector<F> values) { 
            vfloatCols_.push_back(VFloatColumn(name, doc, values.size())); 
            std::copy(values.begin(), values.end(), vfloatCols_.back().values.begin());
        } 

        template<typename I>
        void addVInt(const std::string & name, const std::string & doc, const std::vector<I> values) { 
            vintCols_.push_back(VIntColumn(name, doc, values.size())); 
            std::copy(values.begin(), values.end(), vintCols_.back().values.begin());
        } 


        bool mergeProduct(const MergeableCounterTable & other) {
            if (!tryMerge(intCols_, other.intCols_)) return false;
            if (!tryMerge(vintCols_, other.vintCols_)) return false;
            if (!tryMerge(floatCols_, other.floatCols_)) return false;
            if (!tryMerge(vfloatCols_, other.vfloatCols_)) return false;
            return true; 
        }

        void swap(MergeableCounterTable& iOther) {
          floatCols_.swap(iOther.floatCols_);
          vfloatCols_.swap(iOther.vfloatCols_);
          intCols_.swap(iOther.intCols_);
          vintCols_.swap(iOther.vintCols_);
        }

   private:
        std::vector<FloatColumn> floatCols_;
        std::vector<VFloatColumn> vfloatCols_;
        std::vector<IntColumn> intCols_;
        std::vector<VIntColumn> vintCols_;

        template<typename T>
        bool tryMerge(std::vector<T> & one, const std::vector<T> & two) {
            if (one.size() != two.size()) return false;
            for (unsigned int i = 0, n = one.size(); i < n; ++i) {
                if (!one[i].compatible(two[i])) return false;
                one[i] += two[i];
            }
            return true;
        }
};

} // namespace nanoaod

#endif
