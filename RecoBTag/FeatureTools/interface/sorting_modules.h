#ifndef RecoBTag_FeatureTools_sorting_modules_h
#define RecoBTag_FeatureTools_sorting_modules_h

#include <algorithm>
#include <cmath>
#include <vector>

#include "FWCore/Utilities/interface/isFinite.h"

namespace btagbtvdeep{

/*
 * the std::sort function needs a strict ordering.
 * means if your_function(a,b) is true, than your_function(b,a) must be false for all a and b
 */

template <class T>
class SortingClass{

  public:

    SortingClass():sortValA(0),sortValB(0),sortValC(0),t_(0) {}

    SortingClass(const T& t, float sortA, float sortB=0, float sortC=0) :
      t_(t), sortValA(sortA), sortValB(sortB), sortValC(sortC) {}

    const T& get() const {return t_;}

    enum compareResult{cmp_smaller,cmp_greater,cmp_invalid};

    static inline compareResult compare(const SortingClass& a, const SortingClass& b,int validx=0){
        float vala=a.sortValA;
        float valb=b.sortValA;
        if(validx==1){
            vala=a.sortValB;
            valb=b.sortValB;
        }else if(validx==2){
            vala=a.sortValC;
            valb=b.sortValC;
        }
        if(edm::isFinite(vala) && edm::isFinite(valb) && valb!=vala){
            if(vala>valb) return cmp_greater;
            else return cmp_smaller;
        }
        if(edm::isFinite(vala) && !edm::isFinite(valb))
            return cmp_greater;
        if(!edm::isFinite(vala) && edm::isFinite(valb))
            return cmp_smaller;
        return cmp_invalid;
    }

       //hierarchical sort
    static bool compareByABC(const SortingClass& a, const SortingClass& b){

        compareResult tmpres=compare(a,b,0);
        if(tmpres==cmp_smaller) return true;
        if(tmpres==cmp_greater) return false;

        tmpres=compare(a,b,1);
        if(tmpres==cmp_smaller) return true;
        if(tmpres==cmp_greater) return false;

        tmpres=compare(a,b,2);
        if(tmpres==cmp_smaller) return true;
        if(tmpres==cmp_greater) return false;

        return false;

    }

    static bool compareByABCInv(const SortingClass& a, const SortingClass& b){
			return compareByABC(b,a);
    }

 private:

    T t_;
    float sortValA,sortValB,sortValC;
};

std::vector<std::size_t> invertSortingVector(const std::vector<SortingClass<std::size_t> > & in);

}
#endif //RecoBTag_FeatureTools_sorting_modules_h
