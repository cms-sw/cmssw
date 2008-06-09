#ifndef PhysicsTools_Utilities_MultiHistoChiSquare_h
#define PhysicsTools_Utilities_MultiHistoChiSquare_h

#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "TH1.h"
#include <iostream>

namespace fit {
  namespace helper {
    struct MultiHistoChiSquareNoArg { };
  }
  
  template<typename T1, typename T2 = helper::MultiHistoChiSquareNoArg, 
	   typename T3 = helper::MultiHistoChiSquareNoArg, 
	   typename T4 = helper::MultiHistoChiSquareNoArg>
  class MultiHistoChiSquare { 
   public:
    MultiHistoChiSquare() { }
    MultiHistoChiSquare(T1 & t1, TH1 *histo1,
			T2 & t2, TH1 *histo2,
			T3 & t3, TH1 *histo3,
			T4 & t4, TH1 *histo4,
			double rangeMin, double rangeMax) :
   chi1_(t1, histo1, rangeMin, rangeMax), 
   chi2_(t2, histo2, rangeMin, rangeMax), 
   chi3_(t3, histo3, rangeMin, rangeMax),
   chi4_(t4, histo4, rangeMin, rangeMax) {
   }
   double operator()() const { 
     return chi1_() + chi2_() + chi3_() + chi4_();
   }
   void setHistos(TH1 *histo1, TH1 *histo2, TH1 *histo3, TH1 * histo4) { 
     chi1_.setHistos(histo1);
     chi2_.setHistos(histo2);
     chi3_.setHistos(histo3);
     chi4_.setHistos(histo4);
   }
   size_t degreesOfFreedom() const { 
     return 
       chi1_.degreesOfFreedom() +
       chi2_.degreesOfFreedom() +
       chi3_.degreesOfFreedom() +
       chi4_.degreesOfFreedom();
   }
   T1 & function1() { return chi1_.function(); }
   const T1 & function1() const { return chi1_.function(); }
   T2 & function2() { return chi2_.function(); }
   const T2 & function2() const { return chi2_.function(); }
   T3 & function3() { return chi3_.function(); }
   const T3 & function3() const { return chi3_.function(); }
   T4 & function4() { return chi4_.function(); }
   const T4 & function4() const { return chi4_.function(); }
  private:
   HistoChiSquare<T1> chi1_;
   HistoChiSquare<T2> chi2_;
   HistoChiSquare<T3> chi3_;
   HistoChiSquare<T4> chi4_;
};
  
  template<typename T1, typename T2,typename T3>
  class MultiHistoChiSquare<T1, T2, T3, helper::MultiHistoChiSquareNoArg> { 
   public:
    MultiHistoChiSquare() { }
    MultiHistoChiSquare(T1 & t1, TH1 *histo1,
			T2 & t2, TH1 *histo2,
			T3 & t3, TH1 *histo3,
			double rangeMin, double rangeMax) :
   chi1_(t1, histo1, rangeMin, rangeMax), 
   chi2_(t2, histo2, rangeMin, rangeMax), 
   chi3_(t3, histo3, rangeMin, rangeMax) {
   }
   double operator()() const { 
     return chi1_() + chi2_() + chi3_();
   }
   void setHistos(TH1 *histo1, TH1 *histo2, TH1 *histo3) { 
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
  class MultiHistoChiSquare<T1, T2, 
			    helper::MultiHistoChiSquareNoArg, 
			    helper::MultiHistoChiSquareNoArg> {
   public:
    MultiHistoChiSquare() { }
    MultiHistoChiSquare(T1 & t1, TH1 *histo1,
			T2 & t2, TH1 *histo2, 
			double rangeMin, double rangeMax): 
      chi1_(t1, histo1, rangeMin, rangeMax), 
      chi2_(t2, histo2, rangeMin, rangeMax) {
    }
    double operator()() const { 
      return chi1_() + chi2_();
    }
    void setHistos(TH1 *histo1, TH1 *histo2) { 
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
  class MultiHistoChiSquare<T1, helper::MultiHistoChiSquareNoArg, 
			    helper::MultiHistoChiSquareNoArg, 
			    helper::MultiHistoChiSquareNoArg> {
   public:
    MultiHistoChiSquare() { }
    MultiHistoChiSquare(T1 & t1, TH1 *histo1, double rangeMin, double rangeMax) 
      : chi1_(t1, histo1, rangeMin, rangeMax) {
    }
    double operator()() const { 
      return chi1_();
    }
    void setHistos(TH1 *histo1) { 
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
