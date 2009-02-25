#ifndef _DATAFORMATS_METRECO_HCALNOISERBXARRAY_H_
#define _DATAFORMATS_METRECO_HCALNOISERBXARRAY_H_

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
      HcalNoiseHPDArray::iterator endHPD(void);
      HcalNoiseHPDArray::const_iterator endHPD(void) const;
      
      // endRBX() and end() are identical
      // added for symmetry with endHPD()
      inline HcalNoiseRBXArray::iterator endRBX(void) { return end(); }
      inline HcalNoiseRBXArray::const_iterator endRBX(void) const { return end(); }
      
      // search tools to get the appropriate HPD/RBX in the array
      // if input is invalid, returns endHPD() or endRBX() when appropriate
      HcalNoiseHPDArray::iterator       findHPD(int hpdindex);
      HcalNoiseHPDArray::const_iterator findHPD(int hpdindex) const;
      HcalNoiseRBXArray::iterator       findRBX(int rbxindex);
      HcalNoiseRBXArray::const_iterator findRBX(int rbxindex) const;
      HcalNoiseHPDArray::iterator       findHPD(const HcalDetId&);
      HcalNoiseHPDArray::const_iterator findHPD(const HcalDetId&) const;
      HcalNoiseRBXArray::iterator       findRBX(const HcalDetId&);
      HcalNoiseRBXArray::const_iterator findRBX(const HcalDetId&) const;
      HcalNoiseHPDArray::iterator       findHPD(const HBHEDataFrame&);
      HcalNoiseHPDArray::const_iterator findHPD(const HBHEDataFrame&) const;
      HcalNoiseRBXArray::iterator       findRBX(const HBHEDataFrame&);
      HcalNoiseRBXArray::const_iterator findRBX(const HBHEDataFrame&) const;
      HcalNoiseHPDArray::iterator       findHPD(const HBHERecHit&);
      HcalNoiseHPDArray::const_iterator findHPD(const HBHERecHit&) const;
      HcalNoiseRBXArray::iterator       findRBX(const HBHERecHit&);
      HcalNoiseRBXArray::const_iterator findRBX(const HBHERecHit&) const;
      
      // same as above but, multiple HPDs/RBXs are possible within one calotower
      void findHPD(const CaloTower&, std::vector<HcalNoiseHPDArray::iterator>&);
      void findHPD(const CaloTower&, std::vector<HcalNoiseHPDArray::const_iterator>&) const;
      void findRBX(const CaloTower&, std::vector<HcalNoiseRBXArray::iterator>&);
      void findRBX(const CaloTower&, std::vector<HcalNoiseRBXArray::const_iterator>&) const;
  
    private:
      
    };

} // end of namespace

#endif
