#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoJets/JetProducers/plugins/CATopJetProducer.h"


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
			new fastjet::CMSTopTagger(conf.getParameter<double> ("ptFrac"),
						  conf.getParameter<double> ("rFrac"),
						  conf.getParameter<double> ("adjacencyParam"))
		);
	}
	else if (tagAlgo_ == FJ_JHU_TOPTAG ) {
		fjJHUTopTagger_ = std::auto_ptr<fastjet::JHTopTagger>(
			new fastjet::JHTopTagger(conf.getParameter<double>("ptFrac"),
						 conf.getParameter<double>("deltaRCut"),
						 conf.getParameter<double>("cosThetaWMax")
						 )
		);
	}
	else if (tagAlgo_ == FJ_NSUB_TAG ) {
		
		fastjet::JetDefinition::Plugin *plugin = new fastjet::SISConePlugin(0.6, 0.75);
		fastjet::JetDefinition NsubJetDef(plugin);
		fjNSUBTagger_ = std::auto_ptr<fastjet::RestFrameNSubjettinessTagger>(
			new fastjet::RestFrameNSubjettinessTagger(NsubJetDef,
								  conf.getParameter<double>("tau2Cut"),
								  conf.getParameter<double>("cosThetaSCut"),
								  conf.getParameter<bool>("useExclusive")
								  )
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
	fastjet::JHTopTagger & JHUTagger = *fjJHUTopTagger_;
	fastjet::RestFrameNSubjettinessTagger & NSUBTagger = *fjNSUBTagger_;


	vector<fastjet::PseudoJet>::iterator jetIt = centralJets.begin(), centralJetsEnd = centralJets.end();
	if ( verbose_ )cout<<"Loop over jets"<<endl;
	for ( ; jetIt != centralJetsEnd; ++jetIt ) {
		
		if (verbose_) cout << "CMS FJ jet pt: " << (*jetIt).perp() << endl;

		fastjet::PseudoJet taggedJet;
		if (tagAlgo_ == FJ_CMS_TOPTAG) taggedJet = CMSTagger.result(*jetIt);
		else if (tagAlgo_ == FJ_JHU_TOPTAG) taggedJet = JHUTagger.result(*jetIt);
		else if (tagAlgo_ == FJ_NSUB_TAG) taggedJet = NSUBTagger.result(*jetIt);
		else cout << "NOT A VALID TAGGING ALGORITHM CHOICE!" << endl;

		if (taggedJet != 0) fjJets_.push_back(taggedJet);
	}
  }
} 

void CATopJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

	edm::ParameterSetDescription desc;
	/// Cambridge-Aachen top jet producer parameters
	desc.add<int>("tagAlgo",	0); 			// choice of top tagging algorithm
    	desc.add<double>("centralEtaCut", 	2.5 );        	// eta for defining "central" jets                                     
    	desc.add<bool >("verbose", 	false );        	
	desc.add<string>("jetCollInstanceName",	"caTopSubJets"); 	// subjet collection
	desc.add<int>("algorithm",	1); 			// 0 = KT, 1 = CA, 2 = anti-KT
	desc.add<int>("useAdjacency", 	2); 			// veto adjacent subjets: 
								// 0,	no adjacency
								//  1,	deltar adjacency 
								//  2,	modified adjacency
								//  3,	calotower neirest neigbor based adjacency (untested)
	vector<double>  sumEtBinsDefault;
	sumEtBinsDefault.push_back(0);
	sumEtBinsDefault.push_back(1600);
	sumEtBinsDefault.push_back(2600);
	desc.add<vector<double>>("sumEtBins",	sumEtBinsDefault); 	// sumEt bins over which cuts vary. vector={bin 0 lower bound, bin 1 lower bound, ...} 
	vector<double>  rBinsDefault;
	rBinsDefault.push_back(0.8);
	rBinsDefault.push_back(0.8);
	rBinsDefault.push_back(0.8);
	desc.add<vector<double>>("rBins", 	rBinsDefault); 		// Jet distance paramter R. R values depend on sumEt bins.
	vector<double>  ptFracBinsDefault;
	ptFracBinsDefault.push_back(0.05);
	ptFracBinsDefault.push_back(0.05);
	ptFracBinsDefault.push_back(0.05);
	desc.add<vector<double>>("ptFracBins", 	ptFracBinsDefault); 	// minimum fraction of central jet pt for subjets (deltap)
	vector<double>  deltarBinsDefault;
	deltarBinsDefault.push_back(0.019);
	deltarBinsDefault.push_back(0.019);
	deltarBinsDefault.push_back(0.019);
	desc.add<vector<double>>("deltarBins",	deltarBinsDefault); 	// Applicable only if useAdjacency=1. deltar adjacency values for each sumEtBin
	vector<double>  nCellBinsDefault;
	nCellBinsDefault.push_back(1.9);
	nCellBinsDefault.push_back(1.9);
	nCellBinsDefault.push_back(1.9);
	desc.add<vector<double>>("nCellBins", 	nCellBinsDefault); 	// Applicable only if useAdjacency=3. number of cells apart for two subjets to be considered "independent"
	desc.add<bool>("useMaxTower", 	false); 			// use max tower in adjacency criterion, otherwise use centroid - NOT USED
	desc.add<double>("sumEtEtaCut", 3.0); 				// eta for event SumEt - NOT USED                                                 
	desc.add<double>("etFrac", 	0.7); 				// fraction of event sumEt / 2 for a jet to be considered "hard" - NOT USED
	desc.add<double>("ptFrac", 	0.05); 
	desc.add<double>("rFrac", 	0); 
	desc.add<double>("adjacencyParam", 	0); 
	desc.add<double>("deltaRCut", 	0.19); 
	desc.add<double>("cosThetaWMax", 	0.7); 
	desc.add<double>("tau2Cut", 	0); 
	desc.add<double>("cosThetaSCut", 	0); 
	desc.add<bool>("useExclusive", 	false); 

	////// From FastjetJetProducer
	desc.add<bool>("useMassDropTagger",	false);
	desc.add<bool>("useFiltering",	false);
	desc.add<bool>("useDynamicFiltering",	false);
	desc.add<bool>("useTrimming",	false);
	desc.add<bool>("usePruning",	false);
	desc.add<bool>("useCMSBoostedTauSeedingAlgorithm",	false);
	desc.add<bool>("useKtPruning",	false);
	desc.add<bool>("useConstituentSubtraction",	false);
	desc.add<bool>("useSoftDrop",	false);
	desc.add<bool>("correctShape",	false);
	desc.add<bool>("UseOnlyVertexTracks",	false);
	desc.add<bool>("UseOnlyOnePV",	false);
	desc.add<double>("muCut",	-1.0);
	desc.add<double>("yCut",	-1.0);
	desc.add<double>("rFilt",	-1.0);
	desc.add<double>("rFiltFactor",	-1.0);
	desc.add<double>("trimPtFracMin",	-1.0);
	desc.add<double>("zcut",	-1.0);
	desc.add<double>("rcut_factor",	-1.0);
	desc.add<double>("csRho_EtaMax",	-1.0);
	desc.add<double>("csRParam",	-1.0);
	desc.add<double>("beta",	-1.0);
	desc.add<double>("R0",	-1.0);
	desc.add<double>("gridMaxRapidity",	-1.0); // For fixed-grid rho
	desc.add<double>("gridSpacing",	-1.0);  // For fixed-grid rho
	desc.add<double>("DzTrVtxMax",	999999.);  
	desc.add<double>("DxyTrVtxMax",	999999.);  
	desc.add<double>("MaxVtxZ",	15.0);  
	desc.add<double>("subjetPtMin",	-1.0);
	desc.add<double>("muMin",	-1.0);
	desc.add<double>("muMax",	-1.0);
	desc.add<double>("yMin",	-1.0);
	desc.add<double>("yMax",	-1.0);
	desc.add<double>("dRMin",	-1.0);
	desc.add<double>("dRMax",	-1.0);
	desc.add<int>("maxDepth",	-1);
	desc.add<int>("nFilt",	-1);
	desc.add<int>("MinVtxNdof",	5);
	///// From VirtualJetProducer
	desc.add<string> ("@module_label",	"" );
	desc.add<InputTag>("src",	InputTag("particleFlow") );
	desc.add<bool>("doAreaFastjet",	false );
	desc.add<double>("Rho_EtaMax", 	4.4 	);
	desc.add<double>("rParam",		0.4 );
	desc.add<string>("jetAlgorithm",	"AntiKt" );
	desc.add<InputTag>("srcPVs",	InputTag("") );
	desc.add<string>("jetType",		"PFJet" );
	desc.add<double>("inputEtMin", 	0.0 );
	desc.add<double>("inputEMin",		0.0 );
	desc.add<double>("jetPtMin",		5. );
	desc.add<bool>("doPVCorrection",	false );
	desc.add<bool>("doRhoFastjet",	false );
	desc.add<bool>("doPUOffsetCorr", 	false	);
	desc.add<string>("subtractorName", 	""	);
	desc.add<bool>("useExplicitGhosts", 	false	);
	desc.add<bool>("doAreaDiskApprox", 	false 	);
	desc.add<double>("voronoiRfact", 	-0.9 	);
	desc.add<double>("Ghost_EtaMax",	5. 	);
	desc.add<int>("Active_Area_Repeats",	1 	);
	desc.add<double>("GhostArea",	 	0.01 	);
	desc.add<bool>("restrictInputs", 	false 	);
	desc.add<unsigned int>("maxInputs", 	1 	);
	desc.add<bool>("writeCompound", 	false 	);
	desc.add<bool>("doFastJetNonUniform", false 	);
	desc.add<bool>("useDeterministicSeed",false 	);
	desc.add<unsigned int>("minSeed", 	14327 	);
	desc.add<int>("verbosity", 		0 	);
	desc.add<double>("puWidth",	 	0. 	);
	desc.add<unsigned int>("nExclude", 	0 	);
	desc.add<unsigned int>("maxBadEcalCells", 	9999999	);
	desc.add<unsigned int>("maxBadHcalCells",	9999999 );
	desc.add<unsigned int>("maxProblematicEcalCells",	9999999 );
	desc.add<unsigned int>("maxProblematicHcalCells",	9999999 );
	desc.add<unsigned int>("maxRecoveredEcalCells",	9999999 );
	desc.add<unsigned int>("maxRecoveredHcalCells",	9999999 );
	vector<double>  puCentersDefault;
	desc.add<vector<double>>("puCenters", 	puCentersDefault);
	/////////////////////
	descriptions.addDefault(desc);

}
//define this as a plug-in
DEFINE_FWK_MODULE(CATopJetProducer);
