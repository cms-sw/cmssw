#ifndef _DATAFORMATS_METRECO_HCALNOISEHPD_H_
#define _DATAFORMATS_METRECO_HCALNOISEHPD_H_

//
// HcalNoiseHPD.h
//
//   description: container class of HPD information for the HCAL
//                Noise Filter.  HcalNoiseHPD's are filled by
//                HcalNoiseAlgs defined in RecoMET/METProducers, hence
//                the friend relationship.  HcalNoiseRBXArray also manages
//                the idnumber.
//
//   author: J.P. Chou, Brown
//
//

#include "boost/array.hpp"

#include <algorithm>

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/METReco/interface/HcalHPDRBXMap.h"

namespace reco {

  //
  // forward declarations
  //
  
  class HcalNoiseHPD;
  class HBHERecHitEnergyComparison;
  class HBHERecHitIdComparison;
  
  //
  // typdefs
  //
  
  typedef boost::array<double, HBHEDataFrame::MAXSAMPLES> DigiArray;
  typedef std::set<HBHERecHit, HBHERecHitEnergyComparison> EnergySortedHBHERecHits;
  typedef std::vector<HcalNoiseHPD> HcalNoiseHPDCollection;
  
  //
  // algorithm classes used by the HPD
  //
  
  // algorithm class used to search/sort HBHERecHits by energy
  class HBHERecHitEnergyComparison : public std::binary_function<HBHERecHit,HBHERecHit,bool> {
  public:
    bool operator()(const HBHERecHit& x, const HBHERecHit& y) const
    { return x.energy()>y.energy(); }
  };
  
  //  
  // main class
  //
  
  class HcalNoiseHPD {
    friend class HcalNoiseInfoProducer; // allows this class the fill the HPDs with info
    friend class HcalNoiseRBXArray;     // allows this class to manage the idnumber
    
  public:
    // constructors
    HcalNoiseHPD();
    
    // destructor
    virtual ~HcalNoiseHPD();
    
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
    
    // pedestal subtracted fC information for the highest energy digi in the HPD by timeslice
    // bigDigi() returns a copy of the DigiArray
    // it's probably faster to access the information using the begin/endBigDigi() const iterators
    // also returns the average time and total fC for the digi
    // bigDigiHighest2/3TS are the sum of the highest 2/3 consecutive time slices
    DigiArray bigDigi(void) const;
    DigiArray::const_iterator beginBigDigi(void) const;
    DigiArray::const_iterator endBigDigi(void) const;
    double bigDigiTime(void) const;
    double bigDigiTotal(void) const;
    double bigDigiHighest2TS(void) const;
    double bigDigiHighest3TS(void) const;
    
    // same as above but the integral over the 5 highest energy Digis
    DigiArray big5Digi(void) const;
    DigiArray::const_iterator beginBig5Digi(void) const;
    DigiArray::const_iterator endBig5Digi(void) const;
    double big5DigiTime(void) const;
    double big5DigiTotal(void) const;
    double big5DigiHighest2TS(void) const;
    double big5DigiHighest3TS(void) const;
    
    // same as above but the integral over all the digis in the HPD
    DigiArray allDigi(void) const;
    DigiArray::const_iterator beginAllDigi(void) const;
    DigiArray::const_iterator endAllDigi(void) const;
    double allDigiTime(void) const;
    double allDigiTotal(void) const;
    double allDigiHighest2TS(void) const;
    double allDigiHighest3TS(void) const;
    
    // total number of adc zeros
    int totalZeros(void) const;
    
    // largest number of adc zeros in a digi in the HPD
    int maxZeros(void) const;
    
    // minimum and maximum time for rechits with E>10 GeV
    double minTime(void) const;
    double maxTime(void) const;
    
    // integral of rechit energy in the HPD
    double rechitEnergy(void) const;
    
    // number of hits
    int numHits(void) const;
    
    // number of hits above threshold
    int numHitsAboveThreshold(void) const;
    
    // calotower energies
    double caloTowerHadE(void) const;
    double caloTowerEmE(void) const;
    double caloTowerTotalE(void) const;
    double caloTowerEmFraction(void) const;
    
    
    // rechits
    EnergySortedHBHERecHits recHits(void) const { return recHits_; }
    
  private:
    
    // unique id number specifying the HPD
    int idnumber_;
    
    // digi data members
    DigiArray bigDigi_;
    DigiArray big5Digi_;
    DigiArray allDigi_;
    int totalZeros_;
    int maxZeros_;
    
    // rechit data members
    double minTime_, maxTime_;
    double rechitEnergy_;
    int numHits_, numHitsAboveThreshold_;
    
    // calotower data members
    double twrHadE_, twrEmE_;
    
    // rec hits sorted by energy
    // keep only the top 5
    static const int MAXRECHITS=5;
    EnergySortedHBHERecHits recHits_;
    
  };

} // end of namespace

#endif
