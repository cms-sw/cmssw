#ifndef HcalCholeskyMatrices_h
#define HcalCholeskyMatrices_h

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include <iostream>
#include <vector>
#include <string>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrix.h"

class HcalTopology;

class HcalCholeskyMatrices : public HcalCondObjectContainerBase
{
   public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
 HcalCholeskyMatrices() : HcalCondObjectContainerBase(0) { }
#endif
      HcalCholeskyMatrices(const HcalTopology* topo);
      ~HcalCholeskyMatrices();
      std::string myname() const {return (std::string)"HcalCholeskyMatrices";}

      const HcalCholeskyMatrix* getValues(DetId fId, bool throwOnFail=true) const;
      const bool exists(DetId fId) const;
      bool addValues(const HcalCholeskyMatrix& myHcalCholeskyMatrix);
      std::vector<DetId> getAllChannels() const;

   private:
      void initContainer(DetId fId);
      std::vector<HcalCholeskyMatrix> HBcontainer;
      std::vector<HcalCholeskyMatrix> HEcontainer;
      std::vector<HcalCholeskyMatrix> HOcontainer;
      std::vector<HcalCholeskyMatrix> HFcontainer;
};

#endif

