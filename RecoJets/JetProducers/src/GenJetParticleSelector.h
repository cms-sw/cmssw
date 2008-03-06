#ifndef RecoJetsJetProducerGenJetParticleSelector
#define RecoJetsJetProducerGenJetParticleSelector
/* \class GenJetParticleSelector
 * 
 * GenJet input particles selector
 * \author: Fedor Ratnikov, UMd.
 * $Id$
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GenJetParticleSelector : public edm::EDProducer
{
 public:
  GenJetParticleSelector (const edm::ParameterSet& ps);
  virtual ~GenJetParticleSelector ();
  /**Produces the EDM products*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:
  edm::InputTag mSrc;
  bool mVerbose;
};

#endif






#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"

typedef SingleObjectSelector <reco::CaloJetCollection, PtMinSelector> PtMinCaloJetSelector;
typedef SingleObjectSelector <reco::GenJetCollection, PtMinSelector> PtMinGenJetSelector;
typedef SingleObjectSelector <reco::PFJetCollection, PtMinSelector> PtMinPFJetSelector;
typedef SingleObjectSelector <reco::BasicJetCollection, PtMinSelector> PtMinBasicJetSelector;

