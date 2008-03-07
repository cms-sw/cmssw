#ifndef RecoJetsJetProducerGenJetInputParticleSelector
#define RecoJetsJetProducerGenJetInputParticleSelector
/* \class GenJetInputParticleSelector
 * 
 * GenJet input particles selector
 * \author: Fedor Ratnikov, UMd.
 * $Id: GenJetInputParticleSelector.h,v 1.1 2008/03/06 00:55:36 fedor Exp $
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GenJetInputParticleSelector : public edm::EDProducer
{
 public:
  GenJetInputParticleSelector (const edm::ParameterSet& ps);
  virtual ~GenJetInputParticleSelector ();
  /**Produces the EDM products*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:
  edm::InputTag mSrc;
  bool mVerbose;
};

#endif
