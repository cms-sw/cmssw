#ifndef HcalCholeskyMatrices_h
#define HcalCholeskyMatrices_h

//#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include <iostream>
#include <vector>
#include <string>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrix.h"

class HcalCholeskyMatrices//: public HcalCondObjectContainer<HcalCholeskyMatrix>
{
   public:
      HcalCholeskyMatrices();
      ~HcalCholeskyMatrices();
//      HcalCholeskyMatrices():HcalCondObjectContainer<HcalCholeskyMatrix>() {}
      std::string myname() const {return (std::string)"HcalCholeskyMatrices";}

      const HcalCholeskyMatrix* getValues(DetId fId) const;
      const bool exists(DetId fId) const;
      bool addValues(const HcalCholeskyMatrix& myHcalCholeskyMatrix, bool h2mode_=false);
      std::vector<DetId> getAllChannels() const;

   private:
      void initContainer(int container, bool h2mode_ = false);
      std::vector<HcalCholeskyMatrix> HBcontainer;
      std::vector<HcalCholeskyMatrix> HEcontainer;
      std::vector<HcalCholeskyMatrix> HOcontainer;
      std::vector<HcalCholeskyMatrix> HFcontainer;
};

#endif

