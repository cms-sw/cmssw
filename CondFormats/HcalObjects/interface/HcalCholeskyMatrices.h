#ifndef HcalCholeskyMatrices_h
#define HcalCholeskyMatrices_h

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrix.h"

class HcalCholeskyMatrices: public HcalCondObjectContainer<HcalCholeskyMatrix>
{
   public:
      HcalCholeskyMatrices():HcalCondObjectContainer<HcalCholeskyMatrix>() {}
      std::string myname() const {return (std::string)"HcalCholeskyMatrices";}
   private:
};

#endif

