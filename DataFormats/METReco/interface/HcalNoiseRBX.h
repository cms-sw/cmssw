#ifndef _DATAFORMATS_METRECO_HCALNOISERBX_H_
#define _DATAFORMATS_METRECO_HCALNOISERBX_H_

//
// HcalNoiseRBX.h
//
//   description: container class of RBX information for the HCAL
//                Noise Filter.  HcalNoiseRBX's are filled by
//                HcalNoiseAlgs defined in RecoMET/METProducers.
//
//   author: J.P. Chou, Brown
//
//

#include "boost/array.hpp"

#include "DataFormats/METReco/interface/HcalNoiseHPD.h"
#include "DataFormats/METReco/interface/HcalHPDRBXMap.h"

namespace reco {

  //
  // forward declarations
  //
  
  class HcalNoiseRBX;
  
  //
  // typedefs
  //
  
  typedef boost::array<HcalNoiseHPD, HcalHPDRBXMap::NUM_HPDS_PER_RBX> HcalNoiseHPDArray;
  typedef std::vector<HcalNoiseRBX> HcalNoiseRBXCollection;
  
  
  class HcalNoiseRBX {
    
    friend class HcalNoiseInfoProducer; // allows this class the fill the HPDs with info
    friend class HcalNoiseRBXArray;     // allows this class to manage the idnumber
    
  public:
    // constructors
    HcalNoiseRBX();
    
    // destructor
    virtual ~HcalNoiseRBX();
    
    // accessors
    int idnumber(void) const;
    
    // subdetector (HB or HE)
    HcalSubdetector subdet(void) const;
    
    // z-side (-1, 1)
    int zside(void) const;
    
    // lowest and highest iphi coordinate used by the RBX
    int iphilo(void) const;
    int iphihi(void) const;
    
    // total number of rechits in the RBX
    int numHits(void) const;
    
    // total number of rechits above threshold in the RBX
    int numHitsAboveThreshold(void) const;
    
    // total number of adc zeros in the RBX
    int totalZeros(void) const;
    
    // largest number of zeros from an adc in the RBX
    int maxZeros(void) const;
    
    // return HPD with the largest rechit energy in the RBX
    HcalNoiseHPDArray::const_iterator maxHPD(void) const;
    
    // return first and last+1 iterators to HPDs
    HcalNoiseHPDArray::const_iterator beginHPD(void) const;
    HcalNoiseHPDArray::const_iterator endHPD(void) const;
    
    // return a *copy* of the HPD array
    // note that it's probably faster to use iterator access above
    HcalNoiseHPDArray HPDs(void) const;
    
    // sum of the rechit energy in the HPDs
    double rechitEnergy(void) const;
    
    // caloTower energy in the RBX
    // these are not identical to the sum of HPD caloTower energies
    double caloTowerHadE(void) const;
    double caloTowerEmE(void) const;
    double caloTowerTotalE(void) const;
    double caloTowerEmFraction(void) const;
    
  private:
    
    // members
    int idnumber_;
    
    // calotower energy
    double twrHadE_, twrEmE_;
    
    // boost::array of hpds
    HcalNoiseHPDArray hpds_;
  };
  
} // end of namespace

#endif
