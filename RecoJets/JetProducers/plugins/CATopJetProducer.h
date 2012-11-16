#ifndef RecoJets_JetProducers_CATopJetProducer_h
#define RecoJets_JetProducers_CATopJetProducer_h


/* *********************************************************


 * \class CATopJetProducer
 * Jet producer to produce top jets using the C-A algorithm to break
 * jets into subjets as described here:
 * "Top-tagging: A Method for Identifying Boosted Hadronic Tops"
 * David E. Kaplan, Keith Rehermann, Matthew D. Schwartz, Brock Tweedie
 * arXiv:0806.0848v1 [hep-ph] 

  \brief Jet producer to run the CATopJetAlgorithm

  \author   Salvatore Rappoccio
  \version  

         Notes on implementation:

	 Because the BaseJetProducer only allows the user to produce
	 one jet collection at a time, this algorithm cannot
	 fit into that paradigm. 

	 All of the "hard" jets are of type BasicJet, since
	 they are "jets of jets". The subjets will be either
	 CaloJets, GenJets, etc.

	 In order to avoid a templatization of the entire
	 EDProducer itself, we only use a templated method
	 to write out the subjets to the event record,
	 and to use that information to write out the
	 hard jets to the event record.

	 This templated method is called "write_outputs". It
	 relies on a second templated method called "write_specific",
	 which relies on some template specialization to create
	 different specific objects (i.e. CaloJets, BasicJets, GenJets, etc). 

 ************************************************************/




#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "RecoJets/JetProducers/plugins/FastjetJetProducer.h"
#include "RecoJets/JetAlgorithms/interface/CATopJetAlgorithm.h"

namespace cms
{
  class CATopJetProducer : public FastjetJetProducer
  {
  public:

    CATopJetProducer(const edm::ParameterSet& ps);

    virtual ~CATopJetProducer() {}

    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

    virtual void runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup );

  private:
    CATopJetAlgorithm        alg_;         /// The algorithm to do the work

  };

}


#endif
