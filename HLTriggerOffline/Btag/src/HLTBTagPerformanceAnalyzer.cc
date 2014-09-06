#include "HLTriggerOffline/Btag/interface/HLTBTagPerformanceAnalyzer.h"
#include "DataFormats/Common/interface/View.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"

/// for gen matching 
/// _BEGIN_
#include <Math/GenVector/VectorUtil.h>
#include "SimDataFormats/JetMatching/interface/JetFlavour.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"

#include <algorithm>
#include <cassert>

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

//
// constructors and destructor
//
HLTBTagPerformanceAnalyzer::HLTBTagPerformanceAnalyzer(const edm::ParameterSet& iConfig)
{
	hlTriggerResults_   = iConfig.getParameter<InputTag> ("TriggerResults");
	hltPathNames_        = iConfig.getParameter< std::vector<std::string> > ("HLTPathNames");
	JetTagCollection_           = iConfig.getParameter< std::vector<InputTag> >("JetTag");
	edm::ParameterSet mc = iConfig.getParameter<edm::ParameterSet>("mcFlavours");
	m_mcPartons =  iConfig.getParameter<edm::InputTag>("mcPartons"); 
	m_mcLabels = mc.getParameterNamesForType<std::vector<unsigned int> >();  

	for (unsigned int i = 0; i < m_mcLabels.size(); ++i)
		m_mcFlavours.push_back( mc.getParameter<std::vector<unsigned int> >(m_mcLabels[i]) );
	m_mcMatching = m_mcPartons.label() != "none" ;

	m_mcRadius=0.3;
	dqm = edm::Service<DQMStore>().operator->();
}


HLTBTagPerformanceAnalyzer::~HLTBTagPerformanceAnalyzer()
{

	// do anything here that needs to be done at desctruction time
	// (e.g. close files, deallocate resources etc.)
}

void HLTBTagPerformanceAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

	bool trigRes=false;
	bool MCOK=false;
	using namespace edm;
	Handle<TriggerResults> TriggerResulsHandler;
	Handle<reco::JetFlavourMatchingCollection> h_mcPartons;
	Exception excp(errors::LogicError);
	if ( hlTriggerResults_.label() == "" || hlTriggerResults_.label() == "NULL" ) 
	{
		excp << "TriggerResults ==> Empty";
		excp.raise();
	}
	try {
		iEvent.getByLabel(hlTriggerResults_, TriggerResulsHandler);
		if (TriggerResulsHandler.isValid())   trigRes=true;
	}  catch (...) { std::cout<<"Exception caught in TriggerResulsHandler"<<std::endl;}
	if ( !trigRes ) {    excp << "TriggerResults ==> not readable";            excp.raise(); }
	   const TriggerResults & triggerResults = *(TriggerResulsHandler.product());
	   std::cout<< !triggerResults.accept(0);
	if (m_mcMatching &&  m_mcPartons.label()!= "" && m_mcPartons.label() != "NULL" ) {
		iEvent.getByLabel(m_mcPartons, h_mcPartons);
		try {
			if (h_mcPartons.isValid()) MCOK=true;
			else {
				std::cout<<"Something wrong with partons "<<std::endl;
				std::cout<<"Partons:" << m_mcPartons.label()  <<std::endl;
				std::cout<<"Partons valid:" << h_mcPartons.isValid()  <<std::endl;
				std::cout<<"mcMatching:" << m_mcMatching <<std::endl;
			}
		} catch(...) { std::cout<<"Partons collection is not valid "<<std::endl; }
	if(h_mcPartons->size()==0) std::cout<<"Partons collection is empty "<<std::endl;
	}
	Handle<reco::JetTagCollection> JetTagHandler;
	for (unsigned int ind=0; ind<hltPathNames_.size();ind++) {
		bool BtagOK=false;
		JetTagMap JetTag;
		if ( !_isfoundHLTs[ind]) continue;
		if ( !triggerResults.accept(hltPathIndexs_[ind]) ) continue;
		
		if (JetTagCollection_[ind].label() != "" && JetTagCollection_[ind].label() != "NULL" )
		{
			try {
				iEvent.getByLabel(JetTagCollection_[ind], JetTagHandler);
				if (JetTagHandler.isValid())   BtagOK=true;						
			}  catch (...) { std::cout<<"Exception caught in JetTagHandler"<<std::endl;}			
		}
		if (BtagOK) for ( auto  iter = JetTagHandler->begin(); iter != JetTagHandler->end(); iter++ )
		{
			JetTag.insert(JetTagMap::value_type(iter->first, iter->second));
		}
		else {
			std::cout << "JetTagCollection=" << JetTagCollection_[ind].label() << std::endl;
			excp << "Collections ==> not found";            
			excp.raise(); 
		}

		for (auto & BtagJT: JetTag) {
			H1_.at(ind)[JetTagCollection_[ind].label()] -> Fill(BtagJT.second);
			if (MCOK) {
				int m = closestJet(BtagJT.first, *h_mcPartons, m_mcRadius);
				unsigned int flavour = (m != -1) ? abs((*h_mcPartons)[m].second.getFlavour()) : 0;
				for (unsigned int i = 0; i < m_mcLabels.size(); ++i) {
					TString flavour_str= m_mcLabels[i].c_str();
					flavours_t flav_collection=  m_mcFlavours[i];
					auto it = std::find(flav_collection.begin(), flav_collection.end(), flavour);
					if (it== flav_collection.end())   continue;
					TString label=JetTagCollection_[ind].label() + "__";
					label+=flavour_str;
					H1_.at(ind)[label.Data()]->Fill(BtagJT.second);
					label=JetTagCollection_[ind].label() + "___";
					label+=flavour_str;
					label+=TString("_disc_pT");
					H2_.at(ind)[label.Data()]->Fill(BtagJT.second,BtagJT.first->pt());
				} /// for flavor
			} /// if MCOK
		} /// for  BtagJT
	}  //for triggers
}




//// ------------ method called once each job just before starting event loop  ------------
	void 
HLTBTagPerformanceAnalyzer::beginJob()
{
	std::string title;
	using namespace std;
	assert(hltPathNames_.size()== JetTagCollection_.size());   
	std::string dqmFolder;
	for (unsigned int ind=0; ind<hltPathNames_.size();ind++) {
		float btagL = -11.;
		float btagU = 1.;
		int   btagBins = 600;
		dqmFolder = Form("HLT/BTag/%s",hltPathNames_[ind].c_str());
		H1_.push_back(std::map<std::string, MonitorElement *>());
		H2_.push_back(std::map<std::string, MonitorElement *>());
		dqm->setCurrentFolder(dqmFolder);
		if ( JetTagCollection_[ind].label() != "" && JetTagCollection_[ind].label() != "NULL" ) { 
			H1_.back()[JetTagCollection_[ind].label()]       = dqm->book1D(JetTagCollection_[ind].label() + "_all",      (JetTagCollection_[ind].label()+ "_all").c_str(),  btagBins, btagL, btagU );
			H1_.back()[JetTagCollection_[ind].label()]      -> setAxisTitle(JetTagCollection_[ind].label() +"discriminant",1);
		} 
		std::cout<<"Booking of flavour-independent plots's been finished."<<std::endl;
		int nBinsPt=60;
		double pTmin=30;
		double pTMax=330;


		for (unsigned int i = 0; i < m_mcLabels.size(); ++i)
		{
			TString flavour= m_mcLabels[i].c_str();
			TString label;
			if ( JetTagCollection_[ind].label() != "" && JetTagCollection_[ind].label() != "NULL" ) {
				label=JetTagCollection_[ind].label()+"__";
				label+=flavour;
				H1_.back()[label.Data()] = 		 dqm->book1D(label.Data(),   Form("%s %s",JetTagCollection_[ind].label().c_str(),flavour.Data()), btagBins, btagL, btagU );
				H1_.back()[label.Data()]->setAxisTitle("disc",1);
				label=JetTagCollection_[ind].label()+"___";
				label+=flavour+TString("_disc_pT");
				H2_.back()[label.Data()] =  dqm->book2D( label.Data(), label.Data(), btagBins, btagL, btagU, nBinsPt, pTmin, pTMax );
				H2_.back()[label.Data()]->setAxisTitle("pT",2);
				H2_.back()[label.Data()]->setAxisTitle("disc",1);
			}
		} /// for mc.size()
	} /// for hltPathNames_.size()
	std::cout<<"Booking of flavour-dependent plots's been finished."<<std::endl;   
}




//// ------------ method called once each job just after ending the event loop  ------------
	void 
HLTBTagPerformanceAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
	void 
HLTBTagPerformanceAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
	triggerConfChanged_ = true;
	hltConfigProvider_.init(iRun, iSetup, hlTriggerResults_.process(), triggerConfChanged_);
	const std::vector< std::string > & hltPathNames = hltConfigProvider_.triggerNames();
	for ( size_t trgs=0; trgs<hltPathNames_.size(); trgs++) {
		unsigned int found = 1;
		int it_mem = -1;
		for (size_t it=0 ; it < hltPathNames.size() ; ++it )
		{
			std::cout<<"The available path : "<< hltPathNames.at(it)<<std::endl;
			found = hltPathNames.at(it).find(hltPathNames_[trgs]);
			if ( found == 0 )
			{
				it_mem= (int) it;
			}
		}
		hltPathIndexs_.push_back(it_mem);
	}
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
	void 
		HLTBTagPerformanceAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
		{
		}

	// ------------ method called when starting to processes a luminosity block  ------------
	void 
		HLTBTagPerformanceAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const & , edm::EventSetup const & )
		{
		}

	// ------------ method called when ending the processing of a luminosity block  ------------
	void 
		HLTBTagPerformanceAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
		{
		}

	// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
	void
		HLTBTagPerformanceAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
			//The following says we do not know what parameters are allowed so do no validation
			// Please change this to state exactly what you do use, even if it is no parameters
			edm::ParameterSetDescription desc;
			desc.setUnknown();
			descriptions.addDefault(desc);
		}

	//define this as a plug-in
	DEFINE_FWK_MODULE(HLTBTagPerformanceAnalyzer);

