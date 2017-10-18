#ifndef RecoBTag_DeepFlavour_sorting_modules_h
#define RecoBTag_DeepFlavour_sorting_modules_h

#include <algorithm>
#include <cmath>
#include <vector>

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
      sortValA(sortA), sortValB(sortB), sortValC(sortC), t_(t) { }

    const T& get() const{return t_;}

    // hierarchical sort
    static bool compareByABC(SortingClass a, SortingClass b){
        if(std::isnormal(a.sortValA) && std::isnormal(b.sortValA) && a.sortValA!=b.sortValA){
            return CompareA(a,b);
        }
        else if(!std::isnormal(a.sortValA) && std::isnormal(b.sortValA)){
            return true;
        }
        else if(std::isnormal(a.sortValA) && !std::isnormal(b.sortValA)){
            return false;
        }
        else{
            if(std::isnormal(a.sortValB) && std::isnormal(b.sortValB) && a.sortValB!=b.sortValB){
                return CompareB(a,b);
            }
            else if(!std::isnormal(a.sortValB) && std::isnormal(b.sortValB)){
                return true;
            }
            else if(std::isnormal(a.sortValB) && !std::isnormal(b.sortValB)){
                return false;
            }
            else{
                if(std::isnormal(a.sortValC) && std::isnormal(b.sortValC) && a.sortValC!=b.sortValC){
                    return CompareC(a,b);
                }
                else if(!std::isnormal(a.sortValC) && std::isnormal(b.sortValC)){
                    return true;
                }
                else if(std::isnormal(a.sortValC) && !std::isnormal(b.sortValC)){
                    return false;
                }
                else{
                    return true;
                }
            }
        }
        return true; //never reached
    }

    static bool compareByABCInv(SortingClass a, SortingClass b){
        return !compareByABC(a,b);
    }

 private:
    float sortValA,sortValB,sortValC;
    
    static bool CompareA(SortingClass a, SortingClass b){
        return a.sortValA<b.sortValA;
    }
    static bool CompareB(SortingClass a, SortingClass b){
        return a.sortValB<b.sortValB;
    }
    static bool CompareC(SortingClass a, SortingClass b){
        return a.sortValC<b.sortValC;
    }
    T t_;
};

std::vector<std::size_t> invertSortingVector(const std::vector<SortingClass<std::size_t> > & in);

}
#endif //RecoBTag_DeepFlavour_sorting_modules_h
