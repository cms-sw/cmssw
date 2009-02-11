#ifndef RecoTauTag_RecoTau_PFRecoTauDiscriminationByNeutralHadrons_H_
#define RecoTauTag_RecoTau_PFRecoTauDiscriminationByNeutralHadrons_H_

/* class PFRecoTauDiscriminationByNeutralHadrons
 * created : Nov 3 2008
 * contributors : Simone Gennai (Simone.Gennai@cern.ch)
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

using namespace std; 
using namespace edm;
using namespace edm::eventsetup; 
using namespace reco;

class PFRecoTauDiscriminationByNeutralHadrons : public EDProducer {
 public:
  explicit PFRecoTauDiscriminationByNeutralHadrons(const ParameterSet& iConfig){   
    PFTauProducer_                      = iConfig.getParameter<InputTag>("PFTauProducer");
    neutralHadrons_                     = iConfig.getParameter<unsigned int>("NumberOfAllowedNeutralHadronsInSignalCone");   
    
    produces<PFTauDiscriminator>();
  }
  ~PFRecoTauDiscriminationByNeutralHadrons(){
    //delete ;
  } 
  virtual void produce(Event&, const EventSetup&);
 private:  
  InputTag PFTauProducer_;
  unsigned int neutralHadrons_;   
};
#endif

