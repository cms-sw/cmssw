#ifndef HcalCovarianceMatrices_h
#define HcalCovarianceMatrices_h

//#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include <iostream>
#include <vector>
#include <string>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/HcalObjects/interface/HcalCovarianceMatrix.h"

class HcalCovarianceMatrices//: public HcalCondObjectContainer<HcalCovarianceMatrix>
{
   public:
//      HcalCovarianceMatrices():HcalCondObjectContainer<HcalCovarianceMatrix>() {}
      HcalCovarianceMatrices();
      ~HcalCovarianceMatrices();
      std::string myname() const {return (std::string)"HcalCovarianceMatrices";}
      const HcalCovarianceMatrix* getValues(DetId fId) const;
      const bool exists(DetId fId) const;
      bool addValues(const HcalCovarianceMatrix& myHcalCovarianceMatrix, bool h2mode_=false);
      std::vector<DetId> getAllChannels() const;

   private:
      void initContainer(int container, bool h2mode_ = false);
      std::vector<HcalCovarianceMatrix> HBcontainer;
      std::vector<HcalCovarianceMatrix> HEcontainer;
      std::vector<HcalCovarianceMatrix> HOcontainer;
      std::vector<HcalCovarianceMatrix> HFcontainer;
};

#endif

