#ifndef _RECOMET_METALGORITHMS_HCALNOISERBXARRAY_H_
#define _RECOMET_METALGORITHMS_HCALNOISERBXARRAY_H_

//
// HcalNoiseRBXArray.h
//
//   description: A boost::array of 72 HcalNoiseRBXs designed to simply search/sorting of elements
//                Automatically labels each RBX individually, and provides O(1) searching tools
//
//
//   author: J.P. Chou, Brown
//
//

#include "boost/array.hpp"

#include "DataFormats/METReco/interface/HcalNoiseHPD.h"
#include "DataFormats/METReco/interface/HcalNoiseRBX.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "RecoMET/METAlgorithms/interface/HcalHPDRBXMap.h"

namespace reco {

  class HcalNoiseRBXArray : public boost::array<HcalNoiseRBX, HcalHPDRBXMap::NUM_RBXS>
    {
    public:
      // constructor/destructor
      HcalNoiseRBXArray();
      virtual ~HcalNoiseRBXArray();
      
      // one past the "last" HPD
      // provides the same functionality as HcalNoiseRBXArray::iterator end()
      // defined already by the base class, to denote that the HPD was not found
      std::vector<HcalNoiseHPD>::iterator endHPD(void);
      std::vector<HcalNoiseHPD>::const_iterator endHPD(void) const;
      
      // endRBX() and end() are identical
      // added for symmetry with endHPD()
      inline HcalNoiseRBXArray::iterator endRBX(void) { return end(); }
      inline HcalNoiseRBXArray::const_iterator endRBX(void) const { return end(); }
      
      // search tools to get the appropriate HPD/RBX in the array
      // if input is invalid, returns endHPD() or endRBX() when appropriate
      std::vector<HcalNoiseHPD>::iterator       findHPD(int hpdindex);
      std::vector<HcalNoiseHPD>::const_iterator findHPD(int hpdindex) const;
      HcalNoiseRBXArray::iterator       findRBX(int rbxindex);
      HcalNoiseRBXArray::const_iterator findRBX(int rbxindex) const;
      std::vector<HcalNoiseHPD>::iterator       findHPD(const HcalDetId&);
      std::vector<HcalNoiseHPD>::const_iterator findHPD(const HcalDetId&) const;
      HcalNoiseRBXArray::iterator       findRBX(const HcalDetId&);
      HcalNoiseRBXArray::const_iterator findRBX(const HcalDetId&) const;
      std::vector<HcalNoiseHPD>::iterator       findHPD(const HBHEDataFrame&);
      std::vector<HcalNoiseHPD>::const_iterator findHPD(const HBHEDataFrame&) const;
      HcalNoiseRBXArray::iterator       findRBX(const HBHEDataFrame&);
      HcalNoiseRBXArray::const_iterator findRBX(const HBHEDataFrame&) const;
      std::vector<HcalNoiseHPD>::iterator       findHPD(const HBHERecHit&);
      std::vector<HcalNoiseHPD>::const_iterator findHPD(const HBHERecHit&) const;
      HcalNoiseRBXArray::iterator       findRBX(const HBHERecHit&);
      HcalNoiseRBXArray::const_iterator findRBX(const HBHERecHit&) const;
      
      // same as above but, multiple HPDs/RBXs are possible within one calotower
      void findHPD(const CaloTower&, std::vector<std::vector<HcalNoiseHPD>::iterator>&);
      void findHPD(const CaloTower&, std::vector<std::vector<HcalNoiseHPD>::const_iterator>&) const;
      void findRBX(const CaloTower&, std::vector<HcalNoiseRBXArray::iterator>&);
      void findRBX(const CaloTower&, std::vector<HcalNoiseRBXArray::const_iterator>&) const;
  
    private:
      
    };

} // end of namespace

#endif
