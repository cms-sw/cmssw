#ifndef RecoTauTag_RecoTau_PFRecoTauDiscriminationAgainstElectron_H_
#define RecoTauTag_RecoTau_PFRecoTauDiscriminationAgainstElectron_H_

/* class PFRecoTauDiscriminationAgainstElectron
 * created : May 02 2008,
 * revised : ,
 * Authorss : Chi Nhan Nguyen (Texas A&M)
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

using namespace std; 
using namespace edm;
using namespace edm::eventsetup; 
using namespace reco;

class PFRecoTauDiscriminationAgainstElectron : public EDProducer {
 public:
  explicit PFRecoTauDiscriminationAgainstElectron(const ParameterSet& iConfig){   
    PFTauProducer_        = iConfig.getParameter<string>("PFTauProducer");
    emFraction_maxValue_  = iConfig.getParameter<double>("EmFraction_maxValue");  
    applyCut_emFraction_  = iConfig.getParameter<bool>("ApplyCut_EmFraction");
    hcalTotOverPLead_minValue_  = iConfig.getParameter<double>("HcalTotOverPLead_minValue");  
    applyCut_hcalTotOverPLead_  = iConfig.getParameter<bool>("ApplyCut_HcalTotOverPLead");
    hcalMaxOverPLead_minValue_  = iConfig.getParameter<double>("HcalMaxOverPLead_minValue");  
    applyCut_hcalMaxOverPLead_  = iConfig.getParameter<bool>("ApplyCut_HcalMaxOverPLead");
    hcal3x3OverPLead_minValue_  = iConfig.getParameter<double>("Hcal3x3OverPLead_minValue");  

    applyCut_hcal3x3OverPLead_  = iConfig.getParameter<bool>("ApplyCut_Hcal3x3OverPLead");
    ecalStripSumEOverPLead_minValue_  = iConfig.getParameter<double>("EcalStripSumEOverPLead_minValue");  
    ecalStripSumEOverPLead_maxValue_  = iConfig.getParameter<double>("EcalStripSumEOverPLead_maxValue");  
    applyCut_ecalStripSumEOverPLead_  = iConfig.getParameter<bool>("ApplyCut_EcalStripSumEOverPLead");
    bremsRecoveryEOverPLead_minValue_  = iConfig.getParameter<double>("BremsRecoveryEOverPLead_minValue");  
    bremsRecoveryEOverPLead_maxValue_  = iConfig.getParameter<double>("BremsRecoveryEOverPLead_maxValue");  

    applyCut_bremsRecoveryEOverPLead_  = iConfig.getParameter<bool>("ApplyCut_BremsRecoveryEOverPLead");

    applyCut_electronPreID_  = iConfig.getParameter<bool>("ApplyCut_ElectronPreID");

    elecPreID0_SumEOverPLead_maxValue  = iConfig.getParameter<double>("ElecPreID0_SumEOverPLead_maxValue");
    elecPreID0_Hcal3x3_minValue  = iConfig.getParameter<double>("ElecPreID0_Hcal3x3_minValue");
    elecPreID1_SumEOverPLead_maxValue  = iConfig.getParameter<double>("ElecPreID1_SumEOverPLead_maxValue");
    elecPreID1_Hcal3x3_minValue  = iConfig.getParameter<double>("ElecPreID1_Hcal3x3_minValue");

    applyCut_ecalCrack_  = iConfig.getParameter<bool>("ApplyCut_EcalCrackCut");
    
    produces<PFTauDiscriminator>();
  }
  ~PFRecoTauDiscriminationAgainstElectron(){} 
  virtual void produce(Event&, const EventSetup&);
 private:
  bool isInEcalCrack(double) const; 
  string PFTauProducer_;
  bool applyCut_emFraction_;
  double emFraction_maxValue_;   
  bool applyCut_hcalTotOverPLead_;
  double hcalTotOverPLead_minValue_;   
  bool applyCut_hcalMaxOverPLead_;
  double hcalMaxOverPLead_minValue_;   
  bool applyCut_hcal3x3OverPLead_;
  double hcal3x3OverPLead_minValue_;   

  bool applyCut_ecalStripSumEOverPLead_;
  double ecalStripSumEOverPLead_minValue_;   
  double ecalStripSumEOverPLead_maxValue_;   
  bool applyCut_bremsRecoveryEOverPLead_;
  double bremsRecoveryEOverPLead_minValue_;   
  double bremsRecoveryEOverPLead_maxValue_;   

  bool applyCut_electronPreID_;
  double elecPreID0_SumEOverPLead_maxValue;
  double elecPreID0_Hcal3x3_minValue;
  double elecPreID1_SumEOverPLead_maxValue;
  double elecPreID1_Hcal3x3_minValue;

  bool applyCut_ecalCrack_;


};
#endif
