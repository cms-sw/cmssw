#ifndef RecoTauTag_RecoTau_CaloRecoTauDiscriminationAgainstElectron_H_
#define RecoTauTag_RecoTau_CaloRecoTauDiscriminationAgainstElectron_H_

/* class CaloRecoTauDiscriminationAgainstElectron
 * created : Feb 17 2008,
 * revised : ,
 * contributors : Konstantinos Petridis, Sebastien Greder, Maiko Takahashi, Alexandre Nikitenko (Imperial College, London)
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminator.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

using namespace std; 
using namespace edm;
using namespace edm::eventsetup; 
using namespace reco;

class CaloRecoTauDiscriminationAgainstElectron : public EDProducer {
 public:
  explicit CaloRecoTauDiscriminationAgainstElectron(const ParameterSet& iConfig){   
    CaloTauProducer_                            = iConfig.getParameter<InputTag>("CaloTauProducer");
    leadTrack_HCAL3x3hitsEtSumOverPt_minvalue_  = iConfig.getParameter<double>("leadTrack_HCAL3x3hitsEtSumOverPt_minvalue");  
    ApplyCut_maxleadTrackHCAL3x3hottesthitDEta_ = iConfig.getParameter<bool>("ApplyCut_maxleadTrackHCAL3x3hottesthitDEta");
    maxleadTrackHCAL3x3hottesthitDEta_          = iConfig.getParameter<double>("maxleadTrackHCAL3x3hottesthitDEta");
    ApplyCut_leadTrackavoidsECALcrack_          = iConfig.getParameter<bool>("ApplyCut_leadTrackavoidsECALcrack");
    
    produces<CaloTauDiscriminator>();
  }
  ~CaloRecoTauDiscriminationAgainstElectron(){} 
  virtual void produce(Event&, const EventSetup&);
 private:  
  InputTag CaloTauProducer_;
  double leadTrack_HCAL3x3hitsEtSumOverPt_minvalue_;   
  bool ApplyCut_maxleadTrackHCAL3x3hottesthitDEta_;
  double maxleadTrackHCAL3x3hottesthitDEta_;
  bool ApplyCut_leadTrackavoidsECALcrack_;
};
#endif
