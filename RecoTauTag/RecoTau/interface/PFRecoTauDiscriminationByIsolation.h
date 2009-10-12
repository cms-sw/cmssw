#ifndef RecoTauTag_RecoTau_PFRecoTauDiscriminationByIsolation_H_
#define RecoTauTag_RecoTau_PFRecoTauDiscriminationByIsolation_H_

/* class PFRecoTauDiscriminationByIsolation
 * created : Jul 23 2007,
 * revised : Jan 27 2009,
 * contributors : Ludovic Houchu (Ludovic.Houchu@cern.ch ; IPHC, Strasbourg), Christian Veelken (veelken@fnal.gov ; UC Davis), 
 *                Evan K. Friis (friis@physics.ucdavis.edu ; UC Davis)
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "RecoTauTag/TauTagTools/interface/PFTauElementsOperators.h"

using namespace std; 
using namespace edm;
using namespace edm::eventsetup; 
using namespace reco;

class PFRecoTauDiscriminationByIsolation : public EDProducer {
 public:
  explicit PFRecoTauDiscriminationByIsolation(const ParameterSet& iConfig){   
    PFTauProducer_                              = iConfig.getParameter<InputTag>("PFTauProducer");
    ApplyDiscriminationByTrackerIsolation_      = iConfig.getParameter<bool>("ApplyDiscriminationByTrackerIsolation");
    maxChargedPt_                               = iConfig.getParameter<double>("maxChargedPt");
    ManipulateTracks_insteadofChargedHadrCands_ = iConfig.getParameter<bool>("ManipulateTracks_insteadofChargedHadrCands");
    TrackerIsolAnnulus_Candsmaxn_               = iConfig.getParameter<int>("TrackerIsolAnnulus_Candsmaxn");
    maxGammaPt_                                 = iConfig.getParameter<double>("maxGammaPt");
    ApplyDiscriminationByECALIsolation_         = iConfig.getParameter<bool>("ApplyDiscriminationByECALIsolation");
    ECALIsolAnnulus_Candsmaxn_                  = iConfig.getParameter<int>("ECALIsolAnnulus_Candsmaxn");
    //     following parameters are considered when ManipulateTracks_insteadofChargedHadrCands_ parameter is set true
    TrackerIsolAnnulus_Tracksmaxn_              = iConfig.getParameter<int>("TrackerIsolAnnulus_Tracksmaxn");   
    
    produces<PFTauDiscriminator>();
  }
  ~PFRecoTauDiscriminationByIsolation(){
    //delete ;
  } 
  virtual void produce(Event&, const EventSetup&);
 private:  
  InputTag PFTauProducer_;
  bool ApplyDiscriminationByTrackerIsolation_;
  double maxChargedPt_;
  bool ManipulateTracks_insteadofChargedHadrCands_;
  unsigned int TrackerIsolAnnulus_Candsmaxn_;   
  bool ApplyDiscriminationByECALIsolation_; 
  double maxGammaPt_;
  unsigned int ECALIsolAnnulus_Candsmaxn_; 
  unsigned int TrackerIsolAnnulus_Tracksmaxn_;   
};
#endif

