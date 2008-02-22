#ifndef PhysicsTools_Utilities_MultiHistoChiSquare_h
#define PhysicsTools_Utilities_MultiHistoChiSquare_h

#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "TH1.h"
#include <iostream>

namespace fit {
  namespace helper {
    struct MultiHistoChiSquareNoArg { };
  }

  template<typename T1, typename T2 = helper::MultiHistoChiSquareNoArg, typename T3 = helper::MultiHistoChiSquareNoArg>
  class MultiHistoChiSquare { 
   public:
    MultiHistoChiSquare() { }
    MultiHistoChiSquare(T1 & t1, TH1D *histo1,
			T2 & t2, TH1D *histo2,
			T3 & t3, TH1D *histo3,
			double rangeMin, double rangeMax) :
   chi1_(t1, histo1, rangeMin, rangeMax), 
   chi2_(t2, histo2, rangeMin, rangeMax), 
   chi3_(t3, histo3, rangeMin, rangeMax) {
   }
   double operator()() const { 
     return chi1_() + chi2_() + chi3_();
   }
   void setHistos(TH1D *histo1, TH1D *histo2, TH1D *histo3) { 
     chi1_.setHistos(histo1);
     chi2_.setHistos(histo2);
     chi3_.setHistos(histo3);
   }
   size_t degreesOfFreedom() const { 
     return 
       chi1_.degreesOfFreedom() +
       chi2_.degreesOfFreedom() +
       chi3_.degreesOfFreedom();
   }
   T1 & function1() { return chi1_.function(); }
   const T1 & function1() const { return chi1_.function(); }
   T2 & function2() { return chi2_.function(); }
   const T2 & function2() const { return chi2_.function(); }
   T3 & function3() { return chi3_.function(); }
   const T3 & function3() const { return chi3_.function(); }
  private:
   HistoChiSquare<T1> chi1_;
   HistoChiSquare<T2> chi2_;
   HistoChiSquare<T3> chi3_;
};

  template<typename T1, typename T2>
  class MultiHistoChiSquare<T1, T2, helper::MultiHistoChiSquareNoArg> {
   public:
    MultiHistoChiSquare() { }
    MultiHistoChiSquare(T1 & t1, TH1D *histo1,
			T2 & t2, TH1D *histo2, 
			double rangeMin, double rangeMax): 
      chi1_(t1, histo1, rangeMin, rangeMax), 
      chi2_(t2, histo2, rangeMin, rangeMax) {
    }
    double operator()() const { 
      return chi1_() + chi2_();
    }
    void setHistos(TH1D *histo1, TH1D *histo2) { 
      chi1_.setHistos(histo1);
      chi2_.setHistos(histo2);
    }
    size_t degreesOfFreedom() const { 
      return 
	chi1_.degreesOfFreedom() +
	chi2_.degreesOfFreedom();
    }
   private:
    HistoChiSquare<T1> chi1_;
    HistoChiSquare<T2> chi2_;
  };
  
  template<typename T1>
  class MultiHistoChiSquare<T1, helper::MultiHistoChiSquareNoArg, helper::MultiHistoChiSquareNoArg> {
   public:
    MultiHistoChiSquare() { }
    MultiHistoChiSquare(T1 & t1, TH1D *histo1, double rangeMin, double rangeMax) 
      : chi1_(t1, histo1, rangeMin, rangeMax) {
    }
    double operator()() const { 
      return chi1_();
    }
    void setHistos(TH1D *histo1) { 
      chi1_.setHistos(histo1);
    }
    size_t degreesOfFreedom() const { 
      return chi1_.degreesOfFreedom(); 
    }
   private:
    HistoChiSquare<T1> chi1_;
  };
}

#endif
