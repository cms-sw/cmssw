#ifndef _DATAFORMATS_METRECO_HCALNOISERBX_H_
#define _DATAFORMATS_METRECO_HCALNOISERBX_H_

//
// HcalNoiseRBX.h
//
//   description: Container class of RBX information to study anomalous noise in the HCAL.
//                The information for HcalNoiseHPD's are filled in RecoMET/METProducers/HcalNoiseInfoProducer,
//                but the idnumber is managed by DataFormats/METReco/HcalNoiseRBXArray.
//                Essentially contains 4 HcalNoiseHPDs.
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
  
  typedef std::vector<HcalNoiseRBX> HcalNoiseRBXCollection;
  
  
  class HcalNoiseRBX {
    
    friend class HcalNoiseInfoProducer; // allows this class the fill the HPDs with info
    friend class HcalNoiseRBXArray;     // allows this class to manage the idnumber
    
  public:
    // constructors
    HcalNoiseRBX();
    
    // destructor
    virtual ~HcalNoiseRBX();

    //
    // Detector ID accessors
    //
    
    // accessors
    int idnumber(void) const;
    
    // subdetector (HB or HE)
    HcalSubdetector subdet(void) const;
    
    // z-side (-1, 1)
    int zside(void) const;
    
    // lowest and highest iphi coordinate used by the RBX
    int iphilo(void) const;
    int iphihi(void) const;
    
    //
    // other accessors
    //

    // returns a reference to a vector of HcalNoiseHPDs
    const std::vector<HcalNoiseHPD> HPDs(void) const;

    // return HPD with the highest rechit energy in the RBX
    // individual rechits only contribute if they have E>threshold
    std::vector<HcalNoiseHPD>::const_iterator maxHPD(double threshold=1.5) const;

    // pedestal subtracted fC information for all of the pixels in the RBX
    const std::vector<float> allCharge(void) const;
    float allChargeTotal(void) const;
    float allChargeHighest2TS(void) const;
    float allChargeHighest3TS(void) const;

    // total number of adc zeros in the RBX
    int totalZeros(void) const;
    
    // largest number of zeros from an adc in the RBX
    int maxZeros(void) const;
    
    // sum of the energy of rechits in the RBX with E>threshold
    double recHitEnergy(double theshold=1.5) const;

    // minimum and maximum time for rechits in the RBX with E>threshold
    double minRecHitTime(double threshold=10.0) const;
    double maxRecHitTime(double threshold=10.0) const;

    // total number of rechits above some threshold in the RBX
    int numRecHits(double threshold=1.5) const;
    
    // calotower properties integrated over the entire RBX
    //    double caloTowerHadE(void) const;
    //    double caloTowerEmE(void) const;
    //    double caloTowerTotalE(void) const;
    //    double caloTowerEmFraction(void) const;
    
  private:
    
    // members
    int idnumber_;
    
    // the hpds
    std::vector<HcalNoiseHPD> hpds_;

    // the charge
    std::vector<float> allCharge_;
  };
  
} // end of namespace

#endif
