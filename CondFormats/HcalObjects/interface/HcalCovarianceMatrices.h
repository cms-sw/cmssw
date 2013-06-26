#ifndef HcalCovarianceMatrices_h
#define HcalCovarianceMatrices_h

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include <iostream>
#include <vector>
#include <string>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/HcalObjects/interface/HcalCovarianceMatrix.h"

class HcalTopology;

class HcalCovarianceMatrices: public HcalCondObjectContainerBase
{
   public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
 HcalCovarianceMatrices() : HcalCondObjectContainerBase(0) { }
#endif
      HcalCovarianceMatrices(const HcalTopology* topo);
      ~HcalCovarianceMatrices();
      std::string myname() const {return (std::string)"HcalCovarianceMatrices";}
      const HcalCovarianceMatrix* getValues(DetId fId, bool throwOnFail=true) const;
      const bool exists(DetId fId) const;
      bool addValues(const HcalCovarianceMatrix& myHcalCovarianceMatrix);
      std::vector<DetId> getAllChannels() const;

   private:
      void initContainer(DetId container);
      std::vector<HcalCovarianceMatrix> HBcontainer;
      std::vector<HcalCovarianceMatrix> HEcontainer;
      std::vector<HcalCovarianceMatrix> HOcontainer;
      std::vector<HcalCovarianceMatrix> HFcontainer;
};

#endif

