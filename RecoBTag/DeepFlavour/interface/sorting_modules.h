/*
 * sorting_modules.h
 *
 *  Created on: 5 Mar 2017
 *      Author: jkiesele
 */

#ifndef DEEPNTUPLES_DEEPNTUPLIZER_INTERFACE_SORTING_MODULES_H_
#define DEEPNTUPLES_DEEPNTUPLIZER_INTERFACE_SORTING_MODULES_H_

#include <algorithm>
#include <cmath>
#include <vector>

namespace sorting{

/*
 * the std::sort function needs a strict ordering.
 * means if your_function(a,b) is true, than your_function(b,a) must be false for all a and b
 */

template<class T>
bool comparePt(T a, T b){
    if(!a)return true;
    if(!b)return false;
    return a->pt()<b->pt();
}


template <class T>
class sortingClass{
public:

    sortingClass():sortValA(0),sortValB(0),sortValC(0),t_(0){}

    sortingClass(const T& t, float sortA, float sortB=0, float sortC=0){
        t_=t;
        sortValA=sortA;
        sortValB=sortB;
        sortValC=sortC;
    }
    sortingClass(const sortingClass&rhs):
    sortValA(rhs.sortValA),sortValB(rhs.sortValB),sortValC(rhs.sortValC),t_(rhs.t_)
    {	}
    
    sortingClass& operator=(const sortingClass&rhs){
    	sortValA=(rhs.sortValA);
    	sortValB=(rhs.sortValB);
    	sortValC=(rhs.sortValC);
    	t_=(rhs.t_);
    	return *this;
    }

    const T& get()const{return t_;}

    //hierarchical sort
    static bool compareByABC(sortingClass a, sortingClass b){
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

    static bool compareByABCInv(sortingClass a, sortingClass b){
        return !compareByABC(a,b);
    }

 private:
    float sortValA,sortValB,sortValC;
    
    static bool CompareA(sortingClass a, sortingClass b){
        return a.sortValA<b.sortValA;
    }
    static bool CompareB(sortingClass a, sortingClass b){
        return a.sortValB<b.sortValB;
    }
    static bool CompareC(sortingClass a, sortingClass b){
        return a.sortValC<b.sortValC;
    }
    T t_;
};



std::vector<std::size_t> invertSortingVector(const std::vector<sortingClass<std::size_t> > & in);


template<class T>
bool compareDxy(T b, T a){
    if(!a && b)return true;
    if(!b && a)return false;
    if(!a && !b)return false;
    return a->dxy()<b->dxy();
}

template<class T>
bool compareDxyDxyErr(T b, T a){
    if(!a) return true;
    if(!b) return false;
    if(!a && !b)return false;

    float aerr=a->dxyError();
    float berr=b->dxyError();


    float asig=a->dxy()/aerr;
    float bsig=b->dxy()/berr;

    if(!std::isnormal(asig) && std::isnormal(bsig))
        return true;
    else if(!std::isnormal(bsig) && std::isnormal(asig))
        return false;
    else if(!std::isnormal(bsig) && !std::isnormal(asig))
        return false;

    return asig<bsig;
}


template<class T>
bool pfCCandSort(T b, T a){

    bool ret=false;

    if(!a) ret= true;
    else if(!b) ret= false;
    else if(!a && !b)ret= false;
    else{

        float aerr=a->dxyError();
        float berr=b->dxyError();


        float asig=a->dxy()/aerr;
        float bsig=b->dxy()/berr;

        if(std::isnormal(asig) && std::isnormal(bsig)){
            return asig<bsig;
        }
        else if(!std::isnormal(asig) && std::isnormal(bsig))
            return true;
        else if(!std::isnormal(bsig) && std::isnormal(asig))
            return false;
        else if(!std::isnormal(bsig) && !std::isnormal(asig)){





        }
    }
}


}
#endif /* DEEPNTUPLES_DEEPNTUPLIZER_INTERFACE_SORTING_MODULES_H_ */
