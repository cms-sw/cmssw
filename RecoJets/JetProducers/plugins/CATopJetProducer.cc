#include "FWCore/Framework/interface/MakerMacros.h"
#include "CATopJetProducer.h"

using namespace edm;
using namespace cms;
using namespace reco;

CATopJetProducer::CATopJetProducer(edm::ParameterSet const& conf):
       FastjetJetProducer( conf ),
  	   alg_(src_,
       conf.getParameter<bool>  ("verbose"),              
       conf.getParameter<int>	("algorithm"),                  // 0 = KT, 1 = CA, 2 = anti-KT
       conf.getParameter<int>   ("useAdjacency"),              	// choose adjacency requirement:
				                                                //  0 = no adjacency
                				                                //  1 = deltar adjacency 
                                				                //  2 = modified adjacency
                                                				//  3 = calotower neirest neigbor based adjacency (untested)
       conf.getParameter<double>("centralEtaCut"),             	// eta for defining "central" jets
       conf.getParameter<double>("jetPtMin"),                  	// min jet pt
       conf.getParameter<std::vector<double> >("sumEtBins"),    // sumEt bins over which cuts may vary. vector={bin 0 lower bound, bin 1 lower bound, ...} 
       conf.getParameter<std::vector<double> >("rBins"),       	// Jet distance paramter R. R values depend on sumEt bins.
       conf.getParameter<std::vector<double> >("ptFracBins"),  	// fraction of hard jet pt that subjet must have (deltap)
       conf.getParameter<std::vector<double> >("deltarBins"),  	// Applicable only if useAdjacency=1. deltar adjacency values for each sumEtBin
       conf.getParameter<std::vector<double> >("nCellBins"),	// Applicable only if useAdjacency=3. number of cells to consider two subjets adjacent
       conf.getParameter<double>("inputEtMin"),                	// seed threshold - NOT USED
       conf.getParameter<bool>  ("useMaxTower"),               	// use max tower as adjacency criterion, otherwise use centroid - NOT USED
       conf.getParameter<double>("sumEtEtaCut"),               	// eta for event SumEt - NOT USED
       conf.getParameter<double>("etFrac")                    	// fraction of event sumEt / 2 for a jet to be considered "hard" -NOT USED
       )
{}

void CATopJetProducer::produce(  edm::Event & e, const edm::EventSetup & c ) 
{
  FastjetJetProducer::produce(e, c);
}

void CATopJetProducer::runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if ( !doAreaFastjet_ && !doRhoFastjet_) {
    fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequence( fjInputs_, *fjJetDefinition_ ) );
  } else if (voronoiRfact_ <= 0) {
    fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequenceArea( fjInputs_, *fjJetDefinition_ , *fjAreaDefinition_ ) );
  } else {
    fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequenceVoronoiArea( fjInputs_, *fjJetDefinition_ , fastjet::VoronoiAreaSpec(voronoiRfact_) ) );
  }

  alg_.run( fjInputs_, fjJets_, fjClusterSeq_ );

}

  
//define this as a plug-in
DEFINE_FWK_MODULE(CATopJetProducer);
