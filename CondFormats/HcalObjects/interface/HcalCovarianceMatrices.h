#ifndef HcalCovarianceMatrices_h
#define HcalCovarianceMatrices_h

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalCovarianceMatrix.h"

class HcalCovarianceMatrices: public HcalCondObjectContainer<HcalCovarianceMatrix>
{
   public:
      HcalCovarianceMatrices():HcalCondObjectContainer<HcalCovarianceMatrix>() {}
      std::string myname() const {return (std::string)"HcalCovarianceMatrices";}
   private:
};

#endif

