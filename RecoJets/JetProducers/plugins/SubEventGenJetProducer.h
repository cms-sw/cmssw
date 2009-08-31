#ifndef RecoJets_JetProducers_SubEventGenJetProducer_h
#define RecoJets_JetProducers_SubEventGenJetProducer_h

/* *********************************************************
  \class SubEventGenJetProducer

  \brief Jet producer to produce compound jets (i.e. jets of jets)

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

Modifications:
         25Feb09: Updated to use anomalous cells, also 
	          included corrected CaloTowers for the PV.

 ************************************************************/

#include <vector>
#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"
#include "SimDataFormats/HiGenData/interface/SubEventMap.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

namespace cms
{
  class SubEventGenJetProducer : public VirtualJetProducer
  {
  public:

    SubEventGenJetProducer(const edm::ParameterSet& ps);
    virtual ~SubEventGenJetProducer() {}
    void produce(edm::Event&, const edm::EventSetup&);
    void runAlgorithm(edm::Event&, const edm::EventSetup&);
    
  protected:
   std::vector<std::vector<fastjet::PseudoJet> > subInputs_;
   std::vector<reco::GenJet>* subJets_;
   const edm::SubEventMap* subEvMap_;
   std::vector<int> hydroTag_;
   std::vector<int> nSubParticles_;
   edm::InputTag mapSrc_;
   bool ignoreHydro_;

  protected:

    // overridden inputTowers method. Resets fjCompoundJets_ and 
    // calls VirtualJetProducer::inputTowers
    virtual void inputTowers();
  };



  
}


#endif
