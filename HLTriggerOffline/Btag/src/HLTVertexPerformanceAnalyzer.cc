#include "HLTriggerOffline/Btag/interface/HLTVertexPerformanceAnalyzer.h"

HLTVertexPerformanceAnalyzer::HLTVertexPerformanceAnalyzer(const edm::ParameterSet& iConfig)
{
	hlTriggerResults_   = iConfig.getParameter<InputTag> ("TriggerResults");
	hltPathNames_        = iConfig.getParameter< std::vector<std::string> > ("HLTPathNames");
	VertexCollection_           = iConfig.getParameter<std::vector<InputTag> >("Vertex");
	dqm = edm::Service<DQMStore>().operator->();
}


HLTVertexPerformanceAnalyzer::~HLTVertexPerformanceAnalyzer()
{

	// do anything here that needs to be done at desctruction time
	// (e.g. close files, deallocate resources etc.)
}

void HLTVertexPerformanceAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	bool trigRes=false;
	using namespace edm;

	//get triggerResults
	Handle<TriggerResults> TriggerResulsHandler;
	Exception excp(errors::LogicError);
	if ( hlTriggerResults_.label() == "" || hlTriggerResults_.label() == "NULL" ) {
		excp << "TriggerResults ==> Empty";
		excp.raise();
	}
	try {
		iEvent.getByLabel(hlTriggerResults_, TriggerResulsHandler);
		if (TriggerResulsHandler.isValid())   trigRes=true;
	}  catch (...) { std::cout<<"Exception caught in TriggerResulsHandler"<<std::endl;}
	if ( !trigRes ) {    excp << "TriggerResults ==> not readable";            excp.raise(); }
	const TriggerResults & triggerResults = *(TriggerResulsHandler.product());

	//get simVertex
	Handle<reco::VertexCollection> VertexHandler;
	float simPV=0;
	Handle<std::vector<SimVertex> > simVertexCollection;
	try {
		iEvent.getByLabel("g4SimHits", simVertexCollection);
		const SimVertex simPVh = *(simVertexCollection->begin());
		simPV=simPVh.position().z();
	}
	catch (...) { std::cout<<"Exception caught in simVertexCollection"<<std::endl;}
	
	//fill the DQM plot
	for (unsigned int ind=0; ind<hltPathNames_.size();ind++) {
		for (unsigned int coll=0; coll<VertexCollection_.size();coll++) {
			bool VertexOK=false;
			if ( !_isfoundHLTs[ind]) continue;	//if the hltPath is not in the event, skip the event
			if ( !triggerResults.accept(hltPathIndexs_[ind]) ) continue;	//if the hltPath was not accepted skip the event
			
			//get the recoVertex
			if (VertexCollection_.at(coll).label() != "" && VertexCollection_.at(coll).label() != "NULL" )
			{
				try {
					iEvent.getByLabel(VertexCollection_.at(coll), VertexHandler);
					if (VertexHandler.isValid())   VertexOK=true;						
					else std::cout<<"Check:"<< VertexHandler <<std::endl;
				}  catch (...) { std::cout<<"Exception caught in VertexHandler"<<std::endl;}			
			}
			
			//calculate the variable (RecoVertex - SimVertex)
			float value=VertexHandler->begin()->z()-simPV;
			
			//if value is over/under flow, assign the extreme value
			float maxValue=H1_.at(ind)["Vertex_"+VertexCollection_.at(coll).label()]->getTH1F()->GetXaxis()->GetXmax();
			if(value>maxValue)	value=maxValue-0.0001; 
			if(value<-maxValue)	value=-maxValue+0.0001; 
			
			//fill the histo
			if (VertexOK) H1_.at(ind)["Vertex_"+VertexCollection_.at(coll).label()] -> Fill(value);
		}// for on VertexCollection_
	}//for on hltPathNames_
}




// ------------ method called once each job just before starting event loop  ------------
void HLTVertexPerformanceAnalyzer::beginJob()
{
	//book the DQM plots
	using namespace std;
	std::string dqmFolder;
	for (unsigned int ind=0; ind<hltPathNames_.size();ind++) {
		dqmFolder = Form("HLT/Vertex/%s",hltPathNames_[ind].c_str());
		H1_.push_back(std::map<std::string, MonitorElement *>());
		dqm->setCurrentFolder(dqmFolder);
		for (unsigned int coll=0; coll<VertexCollection_.size();coll++) {
			float maxValue = 0.02;
			if(VertexCollection_.at(coll).label()==("hltFastPrimaryVertex")) maxValue = 2.; //for the hltFastPrimaryVertex use a larger scale (res ~ 1 cm)
			float vertexU = maxValue;
			float vertexL = -maxValue;
			int   vertexBins = 400;
			if ( VertexCollection_.at(coll).label() != "" && VertexCollection_.at(coll).label() != "NULL" ) { 
				H1_.back()["Vertex_"+VertexCollection_.at(coll).label()]       = dqm->book1D("Vertex_"+VertexCollection_.at(coll).label(),      VertexCollection_.at(coll).label().c_str(),  vertexBins, vertexL, vertexU );
				H1_.back()["Vertex_"+VertexCollection_.at(coll).label()]      -> setAxisTitle("vertex error (cm)",1);
			}
		}
	}
}




// ------------ method called once each job just after ending the event loop  ------------
void HLTVertexPerformanceAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void HLTVertexPerformanceAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
	triggerConfChanged_ = true;
	hltConfigProvider_.init(iRun, iSetup, hlTriggerResults_.process(), triggerConfChanged_);
	const std::vector< std::string > & allHltPathNames = hltConfigProvider_.triggerNames();

	//fill hltPathIndexs_ with the trigger number of each hltPathNames_
	for ( size_t trgs=0; trgs<hltPathNames_.size(); trgs++) {
		unsigned int found = 1;
		int it_mem = -1;
		for (size_t it=0 ; it < allHltPathNames.size() ; ++it )
		{
			std::cout<<"The available path : "<< allHltPathNames.at(it)<<std::endl;
			found = allHltPathNames.at(it).find(hltPathNames_[trgs]);
			if ( found == 0 )
			{
				it_mem= (int) it;
			}
		}//for allHltPathNames
		hltPathIndexs_.push_back(it_mem); 
	}//for hltPathNames_
	
	//fill _isfoundHLTs for each hltPathNames_
	for ( size_t trgs=0; trgs<hltPathNames_.size(); trgs++) {
		if ( hltPathIndexs_[trgs] < 0 ) {
			std::cout << "Path " << hltPathNames_[trgs] << " does not exist" << std::endl;
			_isfoundHLTs.push_back(false);
		} 
		else {
			_isfoundHLTs.push_back(true);
		}
	}
}

// ------------ method called when ending the processing of a run  ------------
void HLTVertexPerformanceAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
	{
	}

// ------------ method called when starting to processes a luminosity block  ------------
void HLTVertexPerformanceAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const & , edm::EventSetup const & )
	{
	}
	// ------------ method called when ending the processing of a luminosity block  ------------
void HLTVertexPerformanceAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
	{
	}
	// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HLTVertexPerformanceAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
		//The following says we do not know what parameters are allowed so do no validation
		// Please change this to state exactly what you do use, even if it is no parameters
		edm::ParameterSetDescription desc;
		desc.setUnknown();
		descriptions.addDefault(desc);
	}
	//define this as a plug-in
DEFINE_FWK_MODULE(HLTVertexPerformanceAnalyzer);

