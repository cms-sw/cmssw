#include "HLTriggerOffline/Btag/interface/HLTBTagPerformanceAnalyzer.h"

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


// constructors and destructor
HLTBTagPerformanceAnalyzer::HLTBTagPerformanceAnalyzer(const edm::ParameterSet& iConfig)
{
	hlTriggerResults_   		= consumes<edm::TriggerResults>(iConfig.getParameter<InputTag> ("TriggerResults"));
	JetTagCollection_ 			= edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >( "JetTag" ), [this](edm::InputTag const & tag){return mayConsume< reco::JetTagCollection>(tag);});
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
}


HLTBTagPerformanceAnalyzer::~HLTBTagPerformanceAnalyzer()
{

	// do anything here that needs to be done at desctruction time
	// (e.g. close files, deallocate resources etc.)
}

void HLTBTagPerformanceAnalyzer::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
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
	if ( hlTriggerResults_Label == "" || hlTriggerResults_Label == "NULL" ) 
	{
		edm::LogInfo("NoTriggerResults") << "TriggerResults ==> Empty";
		return;
	}
	iEvent.getByToken(hlTriggerResults_, TriggerResulsHandler);
	if (TriggerResulsHandler.isValid())   trigRes=true;
	if ( !trigRes ) { edm::LogInfo("NoTriggerResults") << "TriggerResults ==> not readable"; return;}
	const TriggerResults & triggerResults = *(TriggerResulsHandler.product());

	//get partons
	if (m_mcMatching &&  m_mcPartons_Label!= "" && m_mcPartons_Label != "NULL" ) {
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
		if (JetTagCollection_Label[ind] != "" && JetTagCollection_Label[ind] != "NULL" )
		{
			iEvent.getByToken(JetTagCollection_[ind], JetTagHandler);
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

		for (auto & BtagJT: JetTag) {
			//fill 1D btag plot for 'all'
			H1_.at(ind)[JetTagCollection_Label[ind]] -> Fill(std::fmax(0.0,BtagJT.second));
			if (MCOK) {
				int m = closestJet(BtagJT.first, *h_mcPartons, m_mcRadius);
				unsigned int flavour = (m != -1) ? abs((*h_mcPartons)[m].second.getFlavour()) : 0;
				for (unsigned int i = 0; i < m_mcLabels.size(); ++i) {
					TString flavour_str= m_mcLabels[i].c_str();
					flavours_t flav_collection=  m_mcFlavours[i];
					auto it = std::find(flav_collection.begin(), flav_collection.end(), flavour);
					if (it== flav_collection.end())   continue;
					TString label=JetTagCollection_Label[ind] + "__";
					label+=flavour_str;
					H1_.at(ind)[label.Data()]->Fill(std::fmax(0.0,BtagJT.second));	//fill 1D btag plot for 'b,c,uds'
					label=JetTagCollection_Label[ind] + "___";
					label+=flavour_str;
					label+=TString("_disc_pT");
					H2_.at(ind)[label.Data()]->Fill(std::fmax(0.0,BtagJT.second),BtagJT.first->pt());	//fill 2D btag, jetPt plot for 'b,c,uds'
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
		dqmFolder = Form("HLT/BTag/Discriminator/%s",hltPathNames_[ind].c_str());
		H1_.push_back(std::map<std::string, MonitorElement *>());
		H2_.push_back(std::map<std::string, MonitorElement *>());
		ibooker.setCurrentFolder(dqmFolder);
		
		//book 1D btag plot for 'all'
		if ( JetTagCollection_Label[ind] != "" && JetTagCollection_Label[ind] != "NULL" ) { 
			H1_.back()[JetTagCollection_Label[ind]]       = ibooker.book1D(JetTagCollection_Label[ind] + "_all",      (JetTagCollection_Label[ind]+ "_all").c_str(),  btagBins, btagL, btagU );
			H1_.back()[JetTagCollection_Label[ind]]      -> setAxisTitle(JetTagCollection_Label[ind] +"discriminant",1);
		} 
		int nBinsPt=60;
		double pTmin=30;
		double pTMax=330;

		for (unsigned int i = 0; i < m_mcLabels.size(); ++i)
		{
			TString flavour= m_mcLabels[i].c_str();
			TString label;
			if ( JetTagCollection_Label[ind] != "" && JetTagCollection_Label[ind] != "NULL" ) {
				label=JetTagCollection_Label[ind]+"__";
				label+=flavour;

				//book 1D btag plot for 'b,c,light,g'
				H1_.back()[label.Data()] = 		 ibooker.book1D(label.Data(),   Form("%s %s",JetTagCollection_Label[ind].c_str(),flavour.Data()), btagBins, btagL, btagU );
				H1_.back()[label.Data()]->setAxisTitle("disc",1);
				label=JetTagCollection_Label[ind]+"___";
				label+=flavour+TString("_disc_pT");

				//book 2D btag plot for 'b,c,light,g'
				H2_.back()[label.Data()] =  ibooker.book2D( label.Data(), label.Data(), btagBins, btagL, btagU, nBinsPt, pTmin, pTMax );
				H2_.back()[label.Data()]->setAxisTitle("pT",2);
				H2_.back()[label.Data()]->setAxisTitle("disc",1);
			}
		} /// for mc.size()
	} /// for hltPathNames_.size()
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTBTagPerformanceAnalyzer);

