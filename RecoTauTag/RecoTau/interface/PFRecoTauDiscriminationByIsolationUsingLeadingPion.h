#ifndef RecoTauTag_RecoTau_PFRecoTauDiscriminationByIsolationUsingLeadingPion_H_
#define RecoTauTag_RecoTau_PFRecoTauDiscriminationByIsolationUsingLeadingPion_H_

/* class PFRecoTauDiscriminationByIsolationUsingLeadingPion
 * created : Jul 23 2007,
 * revised : Sep 5 2007,
 * contributors : Ludovic Houchu (Ludovic.Houchu@cern.ch ; IPHC, Strasbourg), Christian Veelken (veelken@fnal.gov ; UC Davis)
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

class PFRecoTauDiscriminationByIsolationUsingLeadingPion : public EDProducer {
 public:
  explicit PFRecoTauDiscriminationByIsolationUsingLeadingPion(const ParameterSet& iConfig){   
    PFTauProducer_                      = iConfig.getParameter<InputTag>("PFTauProducer");
    ApplyDiscriminationByTrackerIsolation_ = iConfig.getParameter<bool>("ApplyDiscriminationByTrackerIsolation");
    ManipulateTracks_insteadofChargedHadrCands_ = iConfig.getParameter<bool>("ManipulateTracks_insteadofChargedHadrCands");
    TrackerIsolAnnulus_Candsmaxn_       = iConfig.getParameter<int>("TrackerIsolAnnulus_Candsmaxn");       
    ApplyDiscriminationByECALIsolation_ = iConfig.getParameter<bool>("ApplyDiscriminationByECALIsolation");
    ECALIsolAnnulus_Candsmaxn_          = iConfig.getParameter<int>("ECALIsolAnnulus_Candsmaxn");
    //     following parameters are considered when ManipulateTracks_insteadofChargedHadrCands_ parameter is set true
    //     *BEGIN*
    TrackerIsolAnnulus_Tracksmaxn_      = iConfig.getParameter<int>("TrackerIsolAnnulus_Tracksmaxn");   
    //     *END*    
    
    produces<PFTauDiscriminator>();
  }
  ~PFRecoTauDiscriminationByIsolationUsingLeadingPion(){
    //delete ;
  } 
  virtual void produce(Event&, const EventSetup&);
 private:  
  InputTag PFTauProducer_;
  bool ApplyDiscriminationByTrackerIsolation_;
  bool ManipulateTracks_insteadofChargedHadrCands_;
  int TrackerIsolAnnulus_Candsmaxn_;   
  bool ApplyDiscriminationByECALIsolation_; 
  int ECALIsolAnnulus_Candsmaxn_; 
  int TrackerIsolAnnulus_Tracksmaxn_;   
};
#endif

