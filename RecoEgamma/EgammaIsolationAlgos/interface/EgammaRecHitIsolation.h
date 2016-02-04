#ifndef EgammaIsolationProducers_EgammaRecHitIsolation_h
#define EgammaIsolationProducers_EgammaRecHitIsolation_h
//*****************************************************************************
// File:      EgammaRecHitIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, adapted from EgammaHcalIsolation by S. Harper
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************

//C++ includes
#include <vector>
#include <functional>

//CMSSW includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

class EgammaRecHitIsolation {
 public:
  
  //constructors
  EgammaRecHitIsolation (double extRadius,
                         double intRadius,
                         double etaSlice,
                         double etLow,
                         double eLow,
                         edm::ESHandle<CaloGeometry> ,
                         CaloRecHitMetaCollectionV* ,
                         const EcalSeverityLevelAlgo*,
                         DetId::Detector detector);
  
  double getEtSum(const reco::Candidate * emObject) const {return getSum_(emObject,true);}
  double getEnergySum(const reco::Candidate * emObject) const{ return  getSum_(emObject,false);}

  double getEtSum(const reco::SuperCluster* emObject ) const {return getSum_(emObject,true);}
  double getEnergySum(const reco::SuperCluster * emObject) const{ return  getSum_(emObject,false);}

  void setUseNumCrystals(bool b=true) { useNumCrystals_ = b; }
  void setVetoClustered(bool b=true) { vetoClustered_ = b; }
  void doSpikeRemoval(const EcalRecHitCollection *const recHits, 
                      const EcalChannelStatus *const chStatus,
                      const int &severityLevelCut = 3 /*0 - 4*/
                      //const float &sevRecHitThresh = 5.0, /*GeV*/
                      //const EcalSeverityLevelAlgo::SpikeId &id = EcalSeverityLevelAlgo::kSwissCross, /*kE1OverE9=0 or kSwissCross=1*/
                      //const float &spIdThresh = 0.95
                      ) { 
    ecalBarHits_ = recHits; 
    chStatus_ = chStatus;
    severityLevelCut_ = severityLevelCut;
    //severityRecHitThreshold_ = sevRecHitThresh;
    //spId_ = id;
    //spIdThreshold_ = spIdThresh;
  }

  void doFlagChecks(const std::vector<int> v) {
    v_chstatus_.clear();
    v_chstatus_.insert(v_chstatus_.begin(),v.begin(),v.end());
    std::sort( v_chstatus_.begin(), v_chstatus_.end() );
  }

  //destructor 
  ~EgammaRecHitIsolation() ;
  
 private:
  double getSum_(const reco::Candidate *,bool returnEt )const;
  double getSum_(const reco::SuperCluster *,bool returnEt )const;

  double extRadius_ ;
  double intRadius_ ;
  double etaSlice_;
  double etLow_ ;
  double eLow_ ;

  
  edm::ESHandle<CaloGeometry>  theCaloGeom_ ;
  CaloRecHitMetaCollectionV* caloHits_ ;
  const EcalSeverityLevelAlgo* sevLevel_;

  bool useNumCrystals_;
  bool vetoClustered_;
  const EcalRecHitCollection *ecalBarHits_;
  const EcalChannelStatus *chStatus_;
  int severityLevelCut_;
  //float severityRecHitThreshold_;
  //EcalSeverityLevelAlgo::SpikeId spId_;
  //float spIdThreshold_;
  std::vector<int> v_chstatus_;

  const CaloSubdetectorGeometry* subdet_[2]; // barrel+endcap
};

#endif
