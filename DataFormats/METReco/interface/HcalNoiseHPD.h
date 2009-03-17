#ifndef _DATAFORMATS_METRECO_HCALNOISEHPD_H_
#define _DATAFORMATS_METRECO_HCALNOISEHPD_H_

//
// HcalNoiseHPD.h
//
//   description: Container class of HPD information to study anomalous noise in the HCAL.
//                The information for HcalNoiseHPD's are filled in RecoMET/METProducers/HcalNoiseInfoProducer,
//                but the idnumber is managed by DataFormats/METReco/HcalNoiseRBXArray.
//                Provides relevant digi, rechit, and calotower information.
//
//   author: J.P. Chou, Brown
//

#include <algorithm>

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/METReco/interface/HcalHPDRBXMap.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

namespace reco {

  //
  // forward declarations
  //
  
  class HcalNoiseHPD;
  class RefHBHERecHitEnergyComparison;
  
  //
  // typdefs
  //
  
  typedef std::vector<HcalNoiseHPD> HcalNoiseHPDCollection;
  
  //
  // RefHBHERecHitEnergyComparison is a class functor to compare energies between Ref<HBHERecHitCollection>
  //
  class RefHBHERecHitEnergyComparison : public std::binary_function<HBHERecHit,HBHERecHit,bool> {
  public:
    bool operator()(const edm::Ref<HBHERecHitCollection>& x, const edm::Ref<HBHERecHitCollection>& y) const
    { return x->energy()>y->energy(); }
  };


  //  
  // class definition
  //
  
  class HcalNoiseHPD {
    friend class HcalNoiseInfoProducer; // allows this class the fill the HPDs with info
    friend class HcalNoiseRBXArray;     // allows this class to manage the idnumber
    
  public:
    // constructor
    HcalNoiseHPD();
    
    // destructor
    virtual ~HcalNoiseHPD();

    //
    // Detector ID accessors
    //
    
    // unique integer specifier for the hpd [1,NUM_HPDS]
    // correlates roughly with the detector phi slice
    int idnumber(void) const;
    
    // subdetector (HB or HE)
    HcalSubdetector subdet(void) const;
    
    // z-side (-1, 1)
    int zside(void) const;
    
    // lowest and highest iPhi coordinate of the HPD
    int iphilo(void) const;
    int iphihi(void) const;

    //
    // Digi accessors
    //
    
    // pedestal subtracted fC information for the highest energy digi in the HPD by timeslice
    // bigDigi() returns a const reference to the digis
    // also returns the average time and total fC for the digi integrated over all timeslices
    // bigDigiHighest2/3TS are the sum of the highest 2/3 consecutive time slices
    const std::vector<float>& bigDigi(void) const;
    float bigDigiTime(void) const;
    float bigDigiTotal(void) const;
    float bigDigiHighest2TS(void) const;
    float bigDigiHighest3TS(void) const;
    
    // same as above but the integral over the 5 highest energy Digis
    const std::vector<float>& big5Digi(void) const;
    float big5DigiTime(void) const;
    float big5DigiTotal(void) const;
    float big5DigiHighest2TS(void) const;
    float big5DigiHighest3TS(void) const;
    
    // same as above but the integral over all the digis in the HPD
    const std::vector<float>& allDigi(void) const;
    float allDigiTime(void) const;
    float allDigiTotal(void) const;
    float allDigiHighest2TS(void) const;
    float allDigiHighest3TS(void) const;
    
    // total number of adc zeros
    int totalZeros(void) const;
    
    // largest number of adc zeros in a digi in the HPD
    int maxZeros(void) const;

    //
    // RecHit accessors
    //

    // returns a reference to a vector of references to the rechits
    const edm::RefVector<HBHERecHitCollection>& recHits(void) const;
    
    // integral of rechit energies in the HPD with E>threshold (default is 1.5 GeV)
    float recHitEnergy(float threshold=1.5) const;
    
    // minimum and maximum time for rechits with E>threshold (default is 10.0 GeV)
    float minRecHitTime(float threshold=10.0) const;
    float maxRecHitTime(float threshold=10.0) const;
    
    // number of rechits with E>threshold (default is 1.5 GeV)
    int numRecHits(float threshold=1.5) const;

    //
    // CaloTower accessors
    //
    
    // returns a reference to a vector of references to the calotowers
    const edm::RefVector<CaloTowerCollection>& caloTowers(void) const;

    // calotower properties integrated over the entire HPD
    double caloTowerHadE(void) const;
    double caloTowerEmE(void) const;
    double caloTowerTotalE(void) const;
    double caloTowerEmFraction(void) const;
    
    
  private:

    // unique id number specifying the HPD
    int idnumber_;
    
    // digi data members
    int totalZeros_;
    int maxZeros_;
    std::vector<float> bigDigi_;
    std::vector<float> big5Digi_;
    std::vector<float> allDigi_;
    
    // a vector of references to rechits
    edm::RefVector<HBHERecHitCollection> rechits_;

    // a transient set of rechits for sorting purposes
    // at some point before storing, these get transfered to the RefVector rechits_
    std::set<edm::Ref<HBHERecHitCollection>, RefHBHERecHitEnergyComparison> refrechitset_;

    // a vector of references to calotowers
    edm::RefVector<CaloTowerCollection> calotowers_;
  };

} // end of namespace

#endif
