#ifndef RecoJets_JetProducers_CompoundJetProducer_h
#define RecoJets_JetProducers_CompoundJetProducer_h

/* *********************************************************
  \class CompoundJetProducer

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


#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"

#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"

namespace cms
{
  class CompoundJetProducer : public VirtualJetProducer
  {
  public:

    CompoundJetProducer(const edm::ParameterSet& ps);

    virtual ~CompoundJetProducer() {}
    
  protected:
    std::vector<CompoundPseudoJet> fjCompoundJets_; /// compound fastjet::PseudoJets

  protected:

    // overridden inputTowers method. Resets fjCompoundJets_ and 
    // calls VirtualJetProducer::inputTowers
    virtual void inputTowers();

    /// Overridden output method. For the compound jet producer, this will
    /// call the "writeCompoundJets" function template. 
    virtual void output( edm::Event & iEvent, edm::EventSetup const& iSetup );
    template< typename T >
    void writeCompoundJets( edm::Event & iEvent, edm::EventSetup const& iSetup);


  };



  
}


#endif
