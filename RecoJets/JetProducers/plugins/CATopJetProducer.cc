#include "FWCore/Framework/interface/MakerMacros.h"
#include "CATopJetProducer.h"

using namespace edm;
using namespace cms;
using namespace reco;

CATopJetProducer::CATopJetProducer(edm::ParameterSet const& conf):
  CompoundJetProducer( conf ),
  alg_(src_,
       conf.getParameter<int>("algorithm"),                    // 0 = KT, 1 = CA, 2 = anti-KT
       conf.getParameter<double>("inputEtMin"),                // seed threshold
       conf.getParameter<double>("centralEtaCut"),             // eta for defining "central" jets
       conf.getParameter<double>("sumEtEtaCut"),               // eta for event SumEt
       conf.getParameter<double>("jetPtMin"),                  // min jet pt
       conf.getParameter<double>("etFrac"),                    // fraction of event sumEt / 2 for a jet to be considered "hard"
       conf.getParameter<bool>  ("useAdjacency"),              // veto adjacent subjets
       conf.getParameter<bool>  ("useMaxTower"),               // use max tower as adjacency criterion, otherwise use centroid
       conf.getParameter<std::vector<double> > ("ptBins"),     // pt bins over which cuts may vary
       conf.getParameter<std::vector<double> >("rBins"),       // cone size bins,
       conf.getParameter<std::vector<double> >("ptFracBins"),  // fraction of hard jet that subjet must have
       conf.getParameter<std::vector<int> >("nCellBins")       // number of cells to consider two subjets adjacent
       )
{}

void CATopJetProducer::produce(  edm::Event & e, const edm::EventSetup & c ) 
{
  CompoundJetProducer::produce(e, c);
}

void CATopJetProducer::runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  alg_.run( fjInputs_, fjCompoundJets_, iSetup );

}

  
//define this as a plug-in
DEFINE_FWK_MODULE(CATopJetProducer);
