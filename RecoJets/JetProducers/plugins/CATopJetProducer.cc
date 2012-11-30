#include "FWCore/Framework/interface/MakerMacros.h"
#include "CATopJetProducer.h"


using namespace edm;
using namespace cms;
using namespace reco;
using namespace std;

CATopJetProducer::CATopJetProducer(edm::ParameterSet const& conf):
       FastjetJetProducer( conf ),
       tagAlgo_(conf.getParameter<int>("tagAlgo")),
       ptMin_(conf.getParameter<double>("jetPtMin")),
       centralEtaCut_(conf.getParameter<double>("centralEtaCut")),
       verbose_(conf.getParameter<bool>("verbose"))
{

	if (tagAlgo_ == CA_TOPTAGGER ) {
		
		legacyCMSTopTagger_ = std::auto_ptr<CATopJetAlgorithm>(
			new CATopJetAlgorithm(src_,
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
      		));
	}
	else if (tagAlgo_ == FJ_CMS_TOPTAG ) {
		fjCMSTopTagger_ = std::auto_ptr<fastjet::CMSTopTagger>(
			new fastjet::CMSTopTagger()
		);
	}
	else if (tagAlgo_ == FJ_HEP_TOPTAG ) {
		fjHEPTopTagger_ = std::auto_ptr<fastjet::HEPTopTagger>(
			new fastjet::HEPTopTagger()
		);
	}
	else if (tagAlgo_ == FJ_JHU_TOPTAG ) {
		fjJHUTopTagger_ = std::auto_ptr<fastjet::JHTopTagger>(
			new fastjet::JHTopTagger()
		);
	}
	else if (tagAlgo_ == FJ_NSUB_TAG ) {
		
		fastjet::JetDefinition::Plugin *plugin = new fastjet::SISConePlugin(0.6, 0.75);
		fastjet::JetDefinition NsubJetDef(plugin);
		fjNSUBTagger_ = std::auto_ptr<fastjet::RestFrameNSubjettinessTagger>(
			new fastjet::RestFrameNSubjettinessTagger(NsubJetDef)
		);
	}
				
		



}

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

  if (tagAlgo_ == CA_TOPTAGGER){
	(*legacyCMSTopTagger_).run( fjInputs_, fjJets_, fjClusterSeq_ );
	
  }
  else {
	
	//Run the jet clustering
	vector<fastjet::PseudoJet> inclusiveJets = fjClusterSeq_->inclusive_jets(ptMin_);

	if ( verbose_ ) cout << "Getting central jets" << endl;
	// Find the transient central jets
	vector<fastjet::PseudoJet> centralJets;
	for (unsigned int i = 0; i < inclusiveJets.size(); i++) {
		
		if (inclusiveJets[i].perp() > ptMin_ && fabs(inclusiveJets[i].rapidity()) < centralEtaCut_) {
			centralJets.push_back(inclusiveJets[i]);
		}
	}

	fastjet::CMSTopTagger & CMSTagger = *fjCMSTopTagger_;
	fastjet::HEPTopTagger & HEPTagger = *fjHEPTopTagger_;
	fastjet::JHTopTagger & JHUTagger = *fjJHUTopTagger_;
	fastjet::RestFrameNSubjettinessTagger & NSUBTagger = *fjNSUBTagger_;


	vector<fastjet::PseudoJet>::iterator jetIt = centralJets.begin(), centralJetsEnd = centralJets.end();
	if ( verbose_ )cout<<"Loop over jets"<<endl;
	for ( ; jetIt != centralJetsEnd; ++jetIt ) {
		
		if (verbose_) cout << "CMS FJ jet pt: " << (*jetIt).perp() << endl;

		fastjet::PseudoJet taggedJet;
		if (tagAlgo_ == FJ_CMS_TOPTAG) taggedJet = CMSTagger.result(*jetIt);
		else if (tagAlgo_ == FJ_HEP_TOPTAG) taggedJet = HEPTagger.result(*jetIt);
		else if (tagAlgo_ == FJ_JHU_TOPTAG) taggedJet = JHUTagger.result(*jetIt);
		else if (tagAlgo_ == FJ_NSUB_TAG) taggedJet = NSUBTagger.result(*jetIt);
		else cout << "NOT A VALID TAGGING ALGORITHM CHOICE!" << endl;

		if (taggedJet != 0) fjJets_.push_back(taggedJet);
	}
  }
} 
//define this as a plug-in
DEFINE_FWK_MODULE(CATopJetProducer);
