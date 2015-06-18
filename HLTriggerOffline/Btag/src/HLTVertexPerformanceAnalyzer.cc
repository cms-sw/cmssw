#include "HLTriggerOffline/Btag/interface/HLTVertexPerformanceAnalyzer.h"

HLTVertexPerformanceAnalyzer::HLTVertexPerformanceAnalyzer(const edm::ParameterSet& iConfig)
{
	hlTriggerResults_   		= consumes<TriggerResults>(iConfig.getParameter<InputTag> ("TriggerResults"));
	VertexCollection_           = 	edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >( "Vertex" ), [this](edm::InputTag const & tag){return mayConsume< reco::VertexCollection>(tag);});
	hltPathNames_        		= iConfig.getParameter< std::vector<std::string> > ("HLTPathNames");
	simVertexCollection_ = consumes<std::vector<SimVertex> >(iConfig.getParameter<edm::InputTag> ("SimVertexCollection"));

	EDConsumerBase::labelsForToken(hlTriggerResults_,label);
	hlTriggerResults_Label = label.module;
	
	for(unsigned int i=0; i<VertexCollection_.size() ; i++){
		EDConsumerBase::labelsForToken(VertexCollection_[i],label);
		VertexCollection_Label.push_back(label.module);
	}
}


HLTVertexPerformanceAnalyzer::~HLTVertexPerformanceAnalyzer()
{
}


void HLTVertexPerformanceAnalyzer::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
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


void HLTVertexPerformanceAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	bool trigRes=false;
	using namespace edm;

	//get triggerResults
	Handle<TriggerResults> TriggerResulsHandler;
	if ( hlTriggerResults_Label == "" || hlTriggerResults_Label == "NULL" ) {
		edm::LogInfo("NoTriggerResults") << "TriggerResults ==> Empty";
		return;
	}
	iEvent.getByToken(hlTriggerResults_, TriggerResulsHandler);
	if (TriggerResulsHandler.isValid())   trigRes=true;
	if ( !trigRes ) { edm::LogInfo("NoTriggerResults") << "TriggerResults ==> not readable"; return;}
	const TriggerResults & triggerResults = *(TriggerResulsHandler.product());

	//get simVertex
	float simPV=0;

	Handle<std::vector<SimVertex> > simVertexCollection;
	iEvent.getByToken(simVertexCollection_, simVertexCollection);
	const SimVertex simPVh = *(simVertexCollection->begin());
	simPV=simPVh.position().z();
	
	//fill the DQM plot
	Handle<VertexCollection> VertexHandler;
	for (unsigned int ind=0; ind<hltPathNames_.size();ind++) {
		for (unsigned int coll=0; coll<VertexCollection_.size();coll++) {
			bool VertexOK=false;
			if ( !_isfoundHLTs[ind]) continue;	//if the hltPath is not in the event, skip the event
			if ( !triggerResults.accept(hltPathIndexs_[ind]) ) continue;	//if the hltPath was not accepted skip the event
			
			//get the recoVertex
			if (VertexCollection_Label.at(coll) != "" && VertexCollection_Label.at(coll) != "NULL" )
			{
				iEvent.getByToken(VertexCollection_.at(coll), VertexHandler);
				if (VertexHandler.isValid()>0)   VertexOK=true;
			}
			
			if (VertexOK){
				//calculate the variable (RecoVertex - SimVertex)
				float value=VertexHandler->begin()->z()-simPV;
				
				//if value is over/under flow, assign the extreme value
				float maxValue=H1_.at(ind)["Vertex_"+VertexCollection_Label.at(coll)]->getTH1F()->GetXaxis()->GetXmax();
				if(value>maxValue)	value=maxValue-0.0001; 
				if(value<-maxValue)	value=-maxValue+0.0001; 
				//fill the histo
				H1_.at(ind)["Vertex_"+VertexCollection_Label.at(coll)] -> Fill(value);
			}
		}// for on VertexCollection_
	}//for on hltPathNames_
}


void HLTVertexPerformanceAnalyzer::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun,edm::EventSetup const &  iSetup )
{
	//book the DQM plots
	using namespace std;
	std::string dqmFolder;
	for (unsigned int ind=0; ind<hltPathNames_.size();ind++) {
		dqmFolder = Form("HLT/BTag/Vertex/%s",hltPathNames_[ind].c_str());
		H1_.push_back(std::map<std::string, MonitorElement *>());
		ibooker.setCurrentFolder(dqmFolder);
		for (unsigned int coll=0; coll<VertexCollection_.size();coll++) {
			float maxValue = 0.02;
			if(VertexCollection_Label.at(coll)==("hltFastPrimaryVertex")) maxValue = 2.; //for the hltFastPrimaryVertex use a larger scale (res ~ 1 cm)
			float vertexU = maxValue;
			float vertexL = -maxValue;
			int   vertexBins = 100;
			if ( VertexCollection_Label.at(coll) != "" && VertexCollection_Label.at(coll) != "NULL" ) { 
				H1_.back()["Vertex_"+VertexCollection_Label.at(coll)]       = ibooker.book1D("Vertex_"+VertexCollection_Label.at(coll),      VertexCollection_Label.at(coll).c_str(),  vertexBins, vertexL, vertexU );
				H1_.back()["Vertex_"+VertexCollection_Label.at(coll)]      -> setAxisTitle("vertex error (cm)",1);
			}
		}
	}
}

DEFINE_FWK_MODULE(HLTVertexPerformanceAnalyzer);

