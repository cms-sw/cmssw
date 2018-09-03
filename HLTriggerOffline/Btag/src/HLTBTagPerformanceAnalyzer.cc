#include "HLTriggerOffline/Btag/interface/HLTBTagPerformanceAnalyzer.h"
#include <set>
#include <algorithm>

using namespace edm;
using namespace reco;

// find the index of the object key of an association vector closest to a given jet, within a given distance
template <typename T, typename V>
int closestJet(const RefToBase<reco::Jet>   jet, const edm::AssociationVector<T, V> & association, double distance) {
	int closest = -1;
	for (unsigned int i = 0; i < association.size(); ++i) {
		double d = ROOT::Math::VectorUtil::DeltaR(jet->momentum(), association[i].first->momentum());
		if (d < distance) {
			distance = d;
			closest  = i;
		}
	}
	return closest;
}

std::set<std::string> keepSetJet{"jetNSecondaryVertices",
				"jetNSelectedTracks",
				"jetNTracks",
				"Jet_JP",
				"chargedHadronEnergyFraction",
				"neutralHadronEnergyFraction",
				"photonEnergyFraction",
				"electronEnergyFraction",
				"muonEnergyFraction",
				"chargedHadronMultiplicity",
				"neutralHadronMultiplicity",
				"photonMultiplicity",
				"electronMultiplicity",
				"muonMultiplicity",
				"hadronMultiplicity",
				"hadronPhotonMultiplicity",
				"totalMultiplicity"
				};

std::set<std::string> keepSetTrack{"trackChi2",
				"trackNTotalHits",
				"trackNPixelHits",
				"trackSip3dVal",
				"trackSip3dSig",
				"trackSip2dVal",
				"trackSip2dSig",
				"trackPtRel",
				"trackDeltaR",
				"trackPtRatio",
				"trackSip3dSig_0",
				"trackSip3dSig_1",
				"trackSip3dSig_2",
				"trackSip3dSig_3",
				"trackMomentum", 
				"trackEta", 
				"trackPhi",
				"trackDecayLenVal",
				"trackDecayLenSig",
				"trackJetDistVal",
				"trackJetDistSig",
				"trackSumJetEtRatio",
				"trackSumJetDeltaR",
				"trackEtaRel"

	};
std::set<std::string> keepSetVtx{"vertexMass",
				"vertexNTracks"
				"vertexFitProb",
				"vertexCategory",
				"vertexEnergyRatio",
				"vertexJetDeltaR",
				"vertexBoostOverSqrtJetPt",
				"flightDistance1dVal",
				"flightDistance1dSig", 
				"flightDistance2dVal",
				"flightDistance2dSig",
				"flightDistance3dVal",
				"flightDistance3dSig"
	};
std::set<std::string> keepSet;

// constructors and destructor
HLTBTagPerformanceAnalyzer::HLTBTagPerformanceAnalyzer(const edm::ParameterSet& iConfig)
{
	mainFolder_                     = iConfig.getParameter<std::string>("mainFolder");
	hlTriggerResults_   		= consumes<edm::TriggerResults>(iConfig.getParameter<InputTag> ("TriggerResults"));
	JetTagCollection_ 			= edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >( "JetTag" ), [this](edm::InputTag const & tag){return mayConsume< reco::JetTagCollection>(tag);});
        shallowTagInfosTokenCalo_ = consumes<std::vector<reco::ShallowTagInfo> > (edm::InputTag("hltDeepCombinedSecondaryVertexBJetTagsInfosCalo"));
        shallowTagInfosTokenPf_ = consumes<std::vector<reco::ShallowTagInfo> > (edm::InputTag("hltDeepCombinedSecondaryVertexBJetTagsInfos"));
	m_mcPartons 				= consumes<JetFlavourMatchingCollection>(iConfig.getParameter<InputTag> ("mcPartons") ); 
	hltPathNames_        		= iConfig.getParameter< std::vector<std::string> > ("HLTPathNames");
	edm::ParameterSet mc 		= iConfig.getParameter<edm::ParameterSet>("mcFlavours");
	m_mcLabels 					= mc.getParameterNamesForType<std::vector<unsigned int> >();  
	
	EDConsumerBase::labelsForToken(m_mcPartons,label);
	m_mcPartons_Label = label.module;
	
	for(unsigned int i=0; i<JetTagCollection_.size() ; i++){
		EDConsumerBase::labelsForToken(JetTagCollection_[i],label);
		JetTagCollection_Label.push_back(label.module);
	}

	EDConsumerBase::labelsForToken(hlTriggerResults_,label);
	hlTriggerResults_Label = label.module;
	
	for (unsigned int i = 0; i < m_mcLabels.size(); ++i)
		m_mcFlavours.push_back( mc.getParameter<std::vector<unsigned int> >(m_mcLabels[i]) );
	m_mcMatching = m_mcPartons_Label != "none" ;

	m_mcRadius=0.3;

	HCALSpecialsNames[HEP17] = "HEP17";
	HCALSpecialsNames[HEP18] = "HEP18";
	HCALSpecialsNames[HEM17] = "HEM17";
		
}


HLTBTagPerformanceAnalyzer::~HLTBTagPerformanceAnalyzer()
{

	// do anything here that needs to be done at desctruction time
	// (e.g. close files, deallocate resources etc.)
}

void HLTBTagPerformanceAnalyzer::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
	// Make a combined set of inputs avaiable 
		std::set_union(std::begin(keepSetJet), std::end(keepSetJet),
        	       std::begin(keepSetTrack), std::end(keepSetTrack),                  
	               std::inserter(keepSet, std::begin(keepSet)));
		std::set_union(std::begin(keepSet), std::end(keepSet),
	               std::begin(keepSetVtx), std::end(keepSetVtx),                  
	               std::inserter(keepSet, std::begin(keepSet)));
	triggerConfChanged_ = true;
	EDConsumerBase::labelsForToken(hlTriggerResults_,label);
	
	hltConfigProvider_.init(iRun, iSetup, label.process, triggerConfChanged_);
	const std::vector< std::string > & allHltPathNames = hltConfigProvider_.triggerNames();

	//fill hltPathIndexs_ with the trigger number of each hltPathNames_
	for ( size_t trgs=0; trgs<hltPathNames_.size(); trgs++) {
		unsigned int found = 1;
		int it_mem = -1;
		for (size_t it=0 ; it < allHltPathNames.size() ; ++it )
		{
			found = allHltPathNames.at(it).find(hltPathNames_[trgs]);
			if ( found == 0 )
			{
				it_mem= (int) it;
			}
		}//for allallHltPathNames
		hltPathIndexs_.push_back(it_mem);
	}//for hltPathNames_
	
	//fill _isfoundHLTs for each hltPathNames_
	for ( size_t trgs=0; trgs<hltPathNames_.size(); trgs++) {
		if ( hltPathIndexs_[trgs] < 0 ) {
			_isfoundHLTs.push_back(false);
		} 
		else {
			_isfoundHLTs.push_back(true);
		}
	}
}


void HLTBTagPerformanceAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

	bool trigRes=false;
	bool MCOK=false;
	using namespace edm;

	//get triggerResults
	Handle<TriggerResults> TriggerResulsHandler;
	Handle<reco::JetFlavourMatchingCollection> h_mcPartons;
	if ( hlTriggerResults_Label.empty() || hlTriggerResults_Label == "NULL" ) 
	{
		edm::LogInfo("NoTriggerResults") << "TriggerResults ==> Empty";
		return;
	}
	iEvent.getByToken(hlTriggerResults_, TriggerResulsHandler);
	if (TriggerResulsHandler.isValid())   trigRes=true;
	if ( !trigRes ) { edm::LogInfo("NoTriggerResults") << "TriggerResults ==> not readable"; return;}
	const TriggerResults & triggerResults = *(TriggerResulsHandler.product());
	
	//get partons
	if (m_mcMatching && !m_mcPartons_Label.empty() && m_mcPartons_Label != "NULL" ) {
		iEvent.getByToken(m_mcPartons, h_mcPartons);
		if (h_mcPartons.isValid()) MCOK=true;
	}

	//fill the 1D and 2D DQM plot
	Handle<reco::JetTagCollection> JetTagHandler;
	for (unsigned int ind=0; ind<hltPathNames_.size();ind++) {
		bool BtagOK=false;
		JetTagMap JetTag;
		if ( !_isfoundHLTs[ind]) continue;	//if the hltPath is not in the event, skip the event
		if ( !triggerResults.accept(hltPathIndexs_[ind]) ) continue;	//if the hltPath was not accepted skip the event
		
		//get JetTagCollection
		if (!JetTagCollection_Label[ind].empty() && JetTagCollection_Label[ind] != "NULL" )
		{
			iEvent.getByToken(JetTagCollection_[ind], JetTagHandler);
                        iEvent.getByToken(shallowTagInfosTokenPf_, shallowTagInfosPf);
                        iEvent.getByToken(shallowTagInfosTokenCalo_, shallowTagInfosCalo);
			if (JetTagHandler.isValid())   BtagOK=true;
		}

		//fill JetTag map
		if (BtagOK) for ( auto  iter = JetTagHandler->begin(); iter != JetTagHandler->end(); iter++ )
		{
			JetTag.insert(JetTagMap::value_type(iter->first, iter->second));
		}
		else {
		    edm::LogInfo("NoCollection") << "Collection " << JetTagCollection_Label[ind] <<  " ==> not found"; return;
		}
		//fill Inputs for All
		for(auto& info :  *(shallowTagInfosPf)) {
			TaggingVariableList vars = info.taggingVariables();
			for(auto entry = vars.begin(); entry != vars.end(); ++entry) 
				{
				if ( keepSet.find(TaggingVariableTokens[entry->first]) != keepSet.end()) {  // if Input name in defined list to keep
					try {H1_.at(ind)[TaggingVariableTokens[entry->first]] -> Fill(std::fmax(0.0, entry->second));  }
					catch (const std::exception& e) {
						continue;
						}
					}
				else continue; 
				}
			}
		//fill tagging 
		for (auto & BtagJT: JetTag) {
			std::map<HCALSpecials, bool> inmodule;
			inmodule[HEP17]=(BtagJT.first->phi() >= -0.87) && (BtagJT.first->phi() < -0.52) && (BtagJT.first->eta() > 1.3);
			inmodule[HEP18]=(BtagJT.first->phi() >= -0.52) && (BtagJT.first->phi() < -0.17) && (BtagJT.first->eta() > 1.3);
			inmodule[HEM17]=(BtagJT.first->phi() >= -0.87) && (BtagJT.first->phi() < -0.52) && (BtagJT.first->eta() < -1.3);
			
				//fill 1D btag plot for 'all'
			H1_.at(ind)[JetTagCollection_Label[ind]] -> Fill(std::fmax(0.0,BtagJT.second));
			for (auto i: HCALSpecialsNames){
				if (inmodule[i.first])
				H1mod_.at(ind)[JetTagCollection_Label[ind]][i.first] -> Fill(std::fmax(0.0,BtagJT.second));
			}
			if (MCOK) {
				int m = closestJet(BtagJT.first, *h_mcPartons, m_mcRadius);
				unsigned int flavour = (m != -1) ? abs((*h_mcPartons)[m].second.getFlavour()) : 0;
				for (unsigned int i = 0; i < m_mcLabels.size(); ++i) {
					std::string flavour_str= m_mcLabels[i];
					flavours_t flav_collection=  m_mcFlavours[i];
					auto it = std::find(flav_collection.begin(), flav_collection.end(), flavour);
					if (it== flav_collection.end())   continue;
					std::string label=JetTagCollection_Label[ind] + "__";
					label+=flavour_str;
					H1_.at(ind)[label]->Fill(std::fmax(0.0,BtagJT.second));	//fill 1D btag plot for 'b,c,uds'
					for (auto j: HCALSpecialsNames){
						if (inmodule[j.first])
						H1mod_.at(ind)[label][j.first]->Fill(std::fmax(0.0,BtagJT.second));	//fill 1D btag plot for 'b,c,uds' in modules (HEP17 etc.)
					}
					label=JetTagCollection_Label[ind] + "___";
					label+=flavour_str;
					std::string labelEta = label;
					std::string labelPhi = label;
					std::string labelEtaPhi = label;
					std::string labelEtaPhi_threshold = label;
					label+="_disc_pT";
					H2_.at(ind)[label]->Fill(std::fmax(0.0,BtagJT.second),BtagJT.first->pt());	//fill 2D btag, jetPt plot for 'b,c,uds'
					for (auto j: HCALSpecialsNames){
						if (inmodule[j.first])
						H2mod_.at(ind)[label][j.first]->Fill(std::fmax(0.0,BtagJT.second),BtagJT.first->pt());
					}
					labelEta+="_disc_eta";
					H2Eta_.at(ind)[labelEta]->Fill(std::fmax(0.0,BtagJT.second),BtagJT.first->eta());	//fill 2D btag, jetEta plot for 'b,c,uds'
					labelPhi+="_disc_phi";
					H2Phi_.at(ind)[labelPhi]->Fill(std::fmax(0.0,BtagJT.second),BtagJT.first->phi());	//fill 2D btag, jetPhi plot for 'b,c,uds'
					labelEtaPhi+="_eta_phi";
	                                H2EtaPhi_.at(ind)[labelEtaPhi]->Fill(BtagJT.first->eta(),BtagJT.first->phi());  //fill 2D btag, jetPhi plot for 'b,c,uds'
					labelEtaPhi_threshold+="_eta_phi_disc05";
					if (BtagJT.second > 0.5) {
	                                H2EtaPhi_threshold_.at(ind)[labelEtaPhi_threshold]->Fill(BtagJT.first->eta(),BtagJT.first->phi());  //fill 2D btag, jetPhi plot for 'b,c,uds' 
					}
				} /// for flavour
			} /// if MCOK
		} /// for BtagJT
	}//for triggers
}




//// ------------ method called once each job just before starting event loop  ------------
void HLTBTagPerformanceAnalyzer::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun,edm::EventSetup const &  iSetup )
{
	//book the DQM plots for each path and for each flavour
	using namespace std;
	assert(hltPathNames_.size()== JetTagCollection_.size());
	std::string dqmFolder;
	for (unsigned int ind=0; ind<hltPathNames_.size();ind++){
		float btagL = 0.;
		float btagU = 1.;
		int   btagBins = 100;
		dqmFolder = Form("%s/Discriminator/%s",mainFolder_.c_str(),hltPathNames_[ind].c_str());
		H1_.push_back(std::map<std::string, MonitorElement *>());
		H2_.push_back(std::map<std::string, MonitorElement *>());
		H1mod_.push_back(std::map<std::string, std::map<HCALSpecials, MonitorElement *> > ());
		H2mod_.push_back(std::map<std::string, std::map<HCALSpecials, MonitorElement *> > ());
		H2Eta_.push_back(std::map<std::string, MonitorElement *>());
		H2Phi_.push_back(std::map<std::string, MonitorElement *>());
		H2EtaPhi_.push_back(std::map<std::string, MonitorElement *>());
		H2EtaPhi_threshold_.push_back(std::map<std::string, MonitorElement *>());
		ibooker.setCurrentFolder(dqmFolder);
		
		//book 1D btag plot for 'all'
		if ( !JetTagCollection_Label[ind].empty() && JetTagCollection_Label[ind] != "NULL" ) { 
			H1_.back()[JetTagCollection_Label[ind]]       = ibooker.book1D(JetTagCollection_Label[ind] + "_all",      JetTagCollection_Label[ind]+ "_all",  btagBins, btagL, btagU );
			H1_.back()[JetTagCollection_Label[ind]]      -> setAxisTitle(JetTagCollection_Label[ind] +"discriminant",1);
			//Input storing 
			ibooker.setCurrentFolder(dqmFolder+"/inputs");
			ibooker.setCurrentFolder(dqmFolder+"/inputs/Jet");
			for (int i=0; i < 100; i++) {
				if ( keepSetJet.find(TaggingVariableTokens[i]) != keepSetJet.end()) { // if input name in defined set
					std::string inpt = TaggingVariableTokens[i];
					H1_.back()[inpt]       = ibooker.book1D(inpt,inpt,  105 , -5  , 100.);
					H1_.back()[inpt]      -> setAxisTitle(inpt,1);
					}
				else continue;
			}
			ibooker.setCurrentFolder(dqmFolder+"/inputs/Track");
			for (int i=0; i < 100; i++) {
				if ( keepSetTrack.find(TaggingVariableTokens[i]) != keepSetTrack.end()) { // if input name in defined set
					std::string inpt = TaggingVariableTokens[i];
					H1_.back()[inpt]       = ibooker.book1D(inpt,inpt,  105 , -5  , 100.);
					H1_.back()[inpt]      -> setAxisTitle(inpt,1);
					}
				else continue;
			}
			ibooker.setCurrentFolder(dqmFolder+"/inputs/Vertex");
			for (int i=0; i < 100; i++) {
				if ( keepSetVtx.find(TaggingVariableTokens[i]) != keepSetVtx.end()) { // if input name in defined set
					std::string inpt = TaggingVariableTokens[i];
					H1_.back()[inpt]       = ibooker.book1D(inpt,inpt,  105 , -5  , 100.);
					H1_.back()[inpt]      -> setAxisTitle(inpt,1);
					}
				else continue;
			}

			for (auto i: HCALSpecialsNames){
				ibooker.setCurrentFolder(dqmFolder+"/"+i.second);
				H1mod_.back()[JetTagCollection_Label[ind]][i.first]       = ibooker.book1D(JetTagCollection_Label[ind] + "_all",      JetTagCollection_Label[ind]+ "_all",  btagBins, btagL, btagU );
				H1mod_.back()[JetTagCollection_Label[ind]][i.first]      -> setAxisTitle(JetTagCollection_Label[ind] +"discriminant",1);
			}
			ibooker.setCurrentFolder(dqmFolder);
		} 
		int nBinsPt=60;
		double pTmin=30;
		double pTMax=330;
		int nBinsPhi=54;
		double phimin=-M_PI;
		double phiMax=M_PI;
		int nBinsEta=40;
		double etamin=-2.4;
		double etaMax=2.4;

		for (unsigned int i = 0; i < m_mcLabels.size(); ++i)
		{
			std::string flavour= m_mcLabels[i];
			std::string label;
			std::string labelEta;
			std::string labelPhi;
			std::string labelEtaPhi;
			std::string labelEtaPhi_threshold;
			if ( !JetTagCollection_Label[ind].empty() && JetTagCollection_Label[ind] != "NULL" ) {
				label=JetTagCollection_Label[ind]+"__";
				label+=flavour;

				//book 1D btag plot for 'b,c,light,g'
				H1_.back()[label] = 		 ibooker.book1D(label,   Form("%s %s",JetTagCollection_Label[ind].c_str(),flavour.c_str()), btagBins, btagL, btagU );
				H1_.back()[label]->setAxisTitle("disc",1);
				for (auto j: HCALSpecialsNames){
					ibooker.setCurrentFolder(dqmFolder+"/"+j.second);
					H1mod_.back()[label][j.first] = 		 ibooker.book1D(label,   Form("%s %s",JetTagCollection_Label[ind].c_str(),flavour.c_str()), btagBins, btagL, btagU );
					H1mod_.back()[label][j.first]->setAxisTitle("disc",1);
				}
				ibooker.setCurrentFolder(dqmFolder);
				label=JetTagCollection_Label[ind]+"___";
				labelEta=label;
				labelPhi=label;
				labelEtaPhi=label;
				labelEtaPhi_threshold=label;
				label+=flavour+"_disc_pT";
				labelEta+=flavour+"_disc_eta";
				labelPhi+=flavour+"_disc_phi";
				labelEtaPhi+=flavour+"_eta_phi";
				labelEtaPhi_threshold+=flavour+"_eta_phi_disc05";

				//book 2D btag plot for 'b,c,light,g'
				H2_.back()[label] =  ibooker.book2D( label, label, btagBins, btagL, btagU, nBinsPt, pTmin, pTMax );
				H2_.back()[label]->setAxisTitle("pT",2);
				H2_.back()[label]->setAxisTitle("disc",1);
				for (auto j: HCALSpecialsNames){
					ibooker.setCurrentFolder(dqmFolder+"/"+j.second);
					H2mod_.back()[label][j.first] =  ibooker.book2D( label, label, btagBins, btagL, btagU, nBinsPt, pTmin, pTMax );
					H2mod_.back()[label][j.first]->setAxisTitle("pT",2);
					H2mod_.back()[label][j.first]->setAxisTitle("disc",1);
				}
				ibooker.setCurrentFolder(dqmFolder);
				H2Eta_.back()[labelEta] =  ibooker.book2D( labelEta, labelEta, btagBins, btagL, btagU, nBinsEta, etamin, etaMax );
				H2Eta_.back()[labelEta]->setAxisTitle("eta",2);
				H2Eta_.back()[labelEta]->setAxisTitle("disc",1);
				H2Phi_.back()[labelPhi] =  ibooker.book2D( labelPhi, labelPhi, btagBins, btagL, btagU, nBinsPhi, phimin, phiMax );
				H2Phi_.back()[labelPhi]->setAxisTitle("phi",2);
				H2Phi_.back()[labelPhi]->setAxisTitle("disc",1);
				H2EtaPhi_.back()[labelEtaPhi] =  ibooker.book2D( labelEtaPhi, labelEtaPhi, nBinsEta, etamin, etaMax, nBinsPhi, phimin, phiMax );
                                H2EtaPhi_.back()[labelEtaPhi]->setAxisTitle("phi",2);
                                H2EtaPhi_.back()[labelEtaPhi]->setAxisTitle("eta",1);
				H2EtaPhi_threshold_.back()[labelEtaPhi_threshold] =  ibooker.book2D( labelEtaPhi_threshold, labelEtaPhi_threshold, nBinsEta, etamin, etaMax, nBinsPhi, phimin, phiMax );
                                H2EtaPhi_threshold_.back()[labelEtaPhi_threshold]->setAxisTitle("phi",2);
                                H2EtaPhi_threshold_.back()[labelEtaPhi_threshold]->setAxisTitle("eta",1);

			}
		} /// for mc.size()
	} /// for hltPathNames_.size()
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTBTagPerformanceAnalyzer);

