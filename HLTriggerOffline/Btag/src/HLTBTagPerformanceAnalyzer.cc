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
//int closestJet(const reco::Jet & jet, const edm::AssociationVector<T, V> & association, double distance) {
int closestJet(const RefToBase<reco::Jet>   jet, const edm::AssociationVector<T, V> & association, double distance) {
  int closest = -1;
//  std::cout<<" closestJet : distance"<<distance<<std::endl;
  for (unsigned int i = 0; i < association.size(); ++i) {
    double d = ROOT::Math::VectorUtil::DeltaR(jet->momentum(), association[i].first->momentum());
//  std::cout<<" 2 closestJet : distance"<<d<<std::endl;

    if (d < distance) {
      distance = d;
      closest  = i;
    }
  }
  return closest;
}


/// 
/// _END_


//
// constructors and destructor
//
HLTBTagPerformanceAnalyzer::HLTBTagPerformanceAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   hlTriggerResults_   = iConfig.getParameter<InputTag> ("TriggerResults");
   hltPathNames_        = iConfig.getParameter< std::vector<std::string> > ("HLTPathNames");

   // trigger   
   l25IPTagInfoCollection_       = iConfig.getParameter< std::vector<InputTag> >("L25IPTagInfo");
   l3IPTagInfoCollection_        = iConfig.getParameter< std::vector<InputTag> >("L3IPTagInfo");
   l25JetTagCollection_          = iConfig.getParameter< std::vector<InputTag> >("L25JetTag");
   l3JetTagCollection_           = iConfig.getParameter< std::vector<InputTag> >("L3JetTag");
   
   // offline
   trackIPTagInfoCollection_       = iConfig.getParameter<InputTag> ("TrackIPTagInfo");
   offlineJetTagCollection_        = iConfig.getParameter<InputTag> ("OfflineJetTag");
   minJetPT_                       = iConfig.getParameter<double>   ("MinJetPT");
   btagAlgos_                       = iConfig.getParameter< std::vector<std::string> >("BTagAlgorithms");


  //gen level partons

  edm::ParameterSet mc = iConfig.getParameter<edm::ParameterSet>("mcFlavours");
  m_mcPartons =  iConfig.getParameter<edm::InputTag>("mcPartons"); 
  m_mcLabels = mc.getParameterNamesForType<std::vector<unsigned int> >();  
    for (unsigned int i = 0; i < m_mcLabels.size(); ++i)
    m_mcFlavours.push_back( mc.getParameter<std::vector<unsigned int> >(m_mcLabels[i]) );
  
  m_mcMatching = m_mcPartons.label() != "none" ;
  m_mcRadius=0.5;

   // various parameters
  //   isData_                         = iConfig.getParameter<bool>   ("IsData");
   
   // DQMStore services   
   dqm = edm::Service<DQMStore>().operator->();
}


HLTBTagPerformanceAnalyzer::~HLTBTagPerformanceAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called for each event  ------------

	
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



   if (m_mcMatching &&  m_mcPartons.label()!= "" && m_mcPartons.label() != "NULL" ) {
    	   iEvent.getByLabel(m_mcPartons, h_mcPartons);
			try {
				if (h_mcPartons.isValid()) MCOK=true;
				else std::cout<<"Something wrong with partons "<<std::endl;
			} catch(...) { std::cout<<"Partons collection is not valid "<<std::endl; }


	}



   Handle<reco::TrackIPTagInfoCollection> l25IPTagInfoHandler; 
   Handle<reco::TrackIPTagInfoCollection> l3IPTagInfoHandler;
   Handle<reco::SecondaryVertexTagInfoCollection> l25CSVTagInfoHandler;
   Handle<reco::SecondaryVertexTagInfoCollection> l3CSVTagInfoHandler;

   Handle<reco::JetTagCollection> l25JetTagHandler;
   Handle<reco::JetTagCollection> l3JetTagHandler;




   reco::TrackIPTagInfoCollection l25IPTagInfos;
   reco::TrackIPTagInfoCollection l3IPTagInfos;
   reco::SecondaryVertexTagInfoCollection l25CSVTagInfos;
   reco::SecondaryVertexTagInfoCollection l3CSVTagInfos;


   // OFFLINE BTAGGING
      bool Off_JetTagOK=false;
      bool Off_PV_OK=false;

	// Btag mapping
      JetTagMap offlineJetTag;


      Handle<reco::TrackIPTagInfoCollection> trackIPTagInfoHandler;
      Handle<reco::JetTagCollection> offlineJetTagHandler;


      if ( trackIPTagInfoCollection_.label() != "" && trackIPTagInfoCollection_.label() != "NULL" ) {

      // IPTagInfo      
       try {

		      iEvent.getByLabel(trackIPTagInfoCollection_, trackIPTagInfoHandler);     
				 if (trackIPTagInfoHandler.isValid())   Off_PV_OK=true;
			} 	catch (...) { std::cout<<"Exception caught in Offline trackIPTagInfoHandler"<<std::endl;}

	  }



	if ( offlineJetTagCollection_.label() != "" && offlineJetTagCollection_.label() != "NULL" )
    	{

	      try {

    		  // JetTag Offline
	    	  iEvent.getByLabel(offlineJetTagCollection_, offlineJetTagHandler);
			  if (offlineJetTagHandler.isValid()) Off_JetTagOK=true;

               }  catch (...) { std::cout<<"Exception caught in offlineJetTagHandler"<<std::endl;}


		}


	if (Off_JetTagOK)
		 for ( auto  iter = offlineJetTagHandler->begin(); iter != offlineJetTagHandler->end(); iter++ )
                 offlineJetTag.insert(JetTagMap::value_type(iter->first, iter->second));




	for (unsigned int ind=0; ind<hltPathNames_.size();ind++) {

   // ONLINE BTAGGING

	    bool L25OK=false;
    	bool L3OK=false;
        bool HLT_PV_OK=false;
		reco::Vertex  hltVertex;

	   JetTagMap l25JetTag;
	   JetTagMap l3JetTag;

   		if ( !_isfoundHLTs[ind]) continue;
  	 	if ( !triggerResults.accept(hltPathIndexs_[ind]) ) continue;


      if ( l3IPTagInfoCollection_[ind].label() != "" && l3IPTagInfoCollection_[ind].label() != "NULL" )
      {
	        try {
			        iEvent.getByLabel(l3IPTagInfoCollection_[ind], l3IPTagInfoHandler);
    			    if (l3IPTagInfoHandler.isValid())   {
			           const reco::TrackIPTagInfoCollection & l3IPTagInfos = *(l3IPTagInfoHandler.product());
                       const reco::Vertex & hltVertex2 = (*(l3IPTagInfos.at(0).primaryVertex().product())).at(0);
					   hltVertex = hltVertex2;	
   					   HLT_PV_OK=true;

					} 
					
			}  catch (...) { std::cout<<"Exception caught in HLT PV Handlers"<<std::endl;}

	}



      if (l25JetTagCollection_[ind].label() != "" && l25JetTagCollection_[ind].label() != "NULL" )
         {
			try {
	            iEvent.getByLabel(l25JetTagCollection_[ind], l25JetTagHandler);

				 if (l25JetTagHandler.isValid())   L25OK=true;						

			   }  catch (...) { std::cout<<"Exception caught in l25JetTagHandler"<<std::endl;}			

 		 }


      if (l3JetTagCollection_[ind].label() != "" && l3JetTagCollection_[ind].label() != "NULL" )
         {
			try {
	            iEvent.getByLabel(l3JetTagCollection_[ind], l3JetTagHandler);

				 if (l3JetTagHandler.isValid())   L3OK=true;						

			   }  catch (...) { std::cout<<"Exception caught in l3JetTagHandler"<<std::endl;}			

 		 }
  




//      if ( !(L25OK || L3OK) ) continue;
      if ( !(L25OK || L3OK) ) {    excp << "Collections for L2.5 and L3 ==> not found";            excp.raise(); }




  	  if (L25OK) 
			 for ( auto  iter = l25JetTagHandler->begin(); iter != l25JetTagHandler->end(); iter++ )
                 l25JetTag.insert(JetTagMap::value_type(iter->first, iter->second));

  	  if (L3OK) 
			 for ( auto  iter = l3JetTagHandler->begin(); iter != l3JetTagHandler->end(); iter++ )
                 l3JetTag.insert(JetTagMap::value_type(iter->first, iter->second));


  
       for (auto & l25JT: l25JetTag) {       
	        H1_.at(ind)["JetTag_L25"] -> Fill(l25JT.second);

            if (MCOK) {
                int m = closestJet(l25JT.first, *h_mcPartons, m_mcRadius);
                unsigned int flavour = (m != -1) ? abs((*h_mcPartons)[m].second.getFlavour()) : 0;
                for (unsigned int i = 0; i < m_mcLabels.size(); ++i) {
                 	TString flavour_str= m_mcLabels[i].c_str();
					flavours_t flav_collection=  m_mcFlavours[i];
					auto it = std::find(flav_collection.begin(), flav_collection.end(), flavour);
                    if (it== flav_collection.end())   continue;
				    	TString label="JetTag_L25_";
	                    label+=flavour_str;
    	                H1_.at(ind)[label.Data()]->Fill(l25JT.second);
                        label+=TString("_disc_pT");
                         H2_.at(ind)[label.Data()]->Fill(l25JT.second,l25JT.first->pt());
               
				} /// for flavor
                
			} /// if MCOK
	  
  		} /// for  l25JT


      for (auto & l3JT: l3JetTag) {
            H1_.at(ind)["JetTag_L3"] -> Fill(l3JT.second);

            if (MCOK) {
                int m = closestJet(l3JT.first, *h_mcPartons, m_mcRadius);
                unsigned int flavour = (m != -1) ? abs((*h_mcPartons)[m].second.getFlavour()) : 0;
                for (unsigned int i = 0; i < m_mcLabels.size(); ++i) {
                    TString flavour_str= m_mcLabels[i].c_str();
                    flavours_t flav_collection=  m_mcFlavours[i];
                    auto it = std::find(flav_collection.begin(), flav_collection.end(), flavour);
                    if (it== flav_collection.end())   continue;
                        TString label="JetTag_L3_";
                        label+=flavour_str;
                        H1_.at(ind)[label.Data()]->Fill(l3JT.second);
                        label+=TString("_disc_pT");
                         H2_.at(ind)[label.Data()]->Fill(l3JT.second,l3JT.first->pt());

                } /// for flavor

            } /// if MCOK

        } /// for  l3JT


      for (auto & OffJT: offlineJetTag) {
         double drMatch = 9999.;
         float l3BtagMatch = 9999.;
		 for (auto & l3JT: l3JetTag) {			
			double dr = reco::deltaR(*OffJT.first,*l3JT.first);
            if ( dr < drMatch )
            {
               drMatch = dr;
               l3BtagMatch = l3JT.second;
            }
		}
			if (OffJT.first->pt()> minJetPT_) {
			  H1_.at(ind)["JetTag_Offline"]->Fill(OffJT.second);
    	     if ( drMatch < 0.5  )  H2_.at(ind)["JetTag_OffvsL3"]->Fill (OffJT.second,l3BtagMatch);
         
			}


	} /// for OffJT	


	if (HLT_PV_OK) 
     {
      H1_.at(ind)["Vertex_HLT_x"] -> Fill(hltVertex.y());
      H1_.at(ind)["Vertex_HLT_y"] -> Fill(hltVertex.y());
      H1_.at(ind)["Vertex_HLT_z"] -> Fill(hltVertex.z());		
	}

	if (Off_PV_OK) {
	  // Offline primary vertex
      const reco::TrackIPTagInfoCollection & trackIPTagInfos = *(trackIPTagInfoHandler.product());
      const reco::Vertex & offVertex = (*(trackIPTagInfos.at(0).primaryVertex().product())).at(0);
		
	  H1_.at(ind)["Vertex_Off_x"] -> Fill(offVertex.x());
      H1_.at(ind)["Vertex_Off_y"] -> Fill(offVertex.y());
      H1_.at(ind)["Vertex_Off_z"] -> Fill(offVertex.z());

	if (HLT_PV_OK) {
	      H2_.at(ind)["Vertex_OffvsHLT_x"] -> Fill(hltVertex.x(),offVertex.x());
    	  H2_.at(ind)["Vertex_OffvsHLT_y"] -> Fill(hltVertex.y(),offVertex.y());
	      H2_.at(ind)["Vertex_OffvsHLT_z"] -> Fill(hltVertex.z(),offVertex.z());

		}
		
	}
 
   
} // for triggers



}




// ------------ method called once each job just before starting event loop  ------------
void 
HLTBTagPerformanceAnalyzer::beginJob()
{
   std::string title;

// ---------------------------------------------   


   assert(hltPathNames_.size()== l25JetTagCollection_.size());
   assert(hltPathNames_.size()== l3JetTagCollection_.size());
   assert(hltPathNames_.size()== l3IPTagInfoCollection_.size());
   
   std::string dqmFolder;
   for (unsigned int ind=0; ind<hltPathNames_.size();ind++) {


   // discriminant range default TC (track counting)


   float btagL = -10.;
   float btagU = 50.;
   int   btagBins = 300;
  

if ( btagAlgos_[ind] == "CSV" )
    {
       btagL = -11.;
       btagU = 1.;
       btagBins = 600;
    }
/* 
   if ( btagAlgos_[ind] == "CSV" )
   {
      btagL = -3.;
      btagU = 3.;
      btagBins = 300;
   }
*/

	   dqmFolder = Form("HLT/BTag/%s",hltPathNames_[ind].c_str());
         H1_.push_back(std::map<std::string, MonitorElement *>());
         H2_.push_back(std::map<std::string, MonitorElement *>());
 	    dqm->setCurrentFolder(dqmFolder);

   // BTag discriminant Histograms
     if ( l25JetTagCollection_[ind].label() != "" && l25JetTagCollection_[ind].label() != "NULL" ) {
      H1_.back()["JetTag_L25"]   = dqm->book1D("JetTag_L25",     l25JetTagCollection_[ind].label().c_str(), btagBins, btagL, btagU );
      H1_.back()["JetTag_L25"] -> setAxisTitle("L25 discriminant",1);
   }

 if ( l3JetTagCollection_[ind].label() != "" && l3JetTagCollection_[ind].label() != "NULL" ) { 
   H1_.back()["JetTag_L3"]       = dqm->book1D("JetTag_L3",      l3JetTagCollection_[ind].label().c_str(),  btagBins, btagL, btagU );
   H1_.back()["JetTag_L3"]      -> setAxisTitle("L3 discriminant",1);

	}





   H1_.back()["JetTag_Offline"]  = dqm->book1D("JetTag_Offline", offlineJetTagCollection_.label().c_str(), btagBins, btagL, btagU );
   H1_.back()["JetTag_Offline"] -> setAxisTitle("Offline discriminant",1);
  
   if ( l3JetTagCollection_[ind].label() != "" && l3JetTagCollection_[ind].label() != "NULL" ) {
   title = Form("%s versus %s", offlineJetTagCollection_.label().c_str(),l3JetTagCollection_[ind].label().c_str());
   H2_.back()["JetTag_OffvsL3"]  = dqm->book2D("JetTag_OffvsL3", title.c_str(), btagBins, btagL, btagU, btagBins, btagL, btagU );
      // axis titles
   H2_.back()["JetTag_OffvsL3"] -> setAxisTitle("Offline discriminant",1);
   H2_.back()["JetTag_OffvsL3"] -> setAxisTitle("L3 discriminant",2);
	
}
   
   
// ---------------------------------------------   
   // Vertex Histograms
   // Vertex position ranges for MC
   float vtxXL = 0.392;
   float vtxXU = 0.396;
    
//   float vtxXL = 0.386;
//   float vtxXU = 0.400;
    float vtxYL = 0.392;
   float vtxYU = 0.396;
   float vtxZL = -20.0;
   float vtxZU =  20.0;
   
/*
   if ( isData_ )
   {
      vtxXL = 0.062;
      vtxXU = 0.076;
      vtxYL = 0.057;
      vtxYU = 0.071;
      vtxZL = -20.0;
      vtxZU =  20.0;
   }
  */ 
   
   H1_.back()["Vertex_HLT_x"]      = dqm->book1D( "Vertex_HLT_x", "HLT vertex x position", 280, vtxXL, vtxXU );
   H1_.back()["Vertex_HLT_y"]      = dqm->book1D( "Vertex_HLT_y", "HLT vertex y position", 280, vtxYL, vtxYU );
   H1_.back()["Vertex_HLT_z"]      = dqm->book1D( "Vertex_HLT_z", "HLT vertex z position", 400, vtxZL, vtxZU );
   H1_.back()["Vertex_Off_x"]      = dqm->book1D( "Vertex_Off_x", "Offline vertex x position", 280, vtxXL, vtxXU );
   H1_.back()["Vertex_Off_y"]      = dqm->book1D( "Vertex_Off_y", "Offline vertex y position", 280, vtxYL, vtxYU );
   H1_.back()["Vertex_Off_z"]      = dqm->book1D( "Vertex_Off_z", "Offline vertex z position", 400, vtxZL, vtxZU );
   H2_.back()["Vertex_OffvsHLT_x"] = dqm->book2D( "Vertex_OffvsHLT_x", "Offline vs HLT vertex x position", 280, vtxXL, vtxXU, 280, vtxXL, vtxXU );
   H2_.back()["Vertex_OffvsHLT_y"] = dqm->book2D( "Vertex_OffvsHLT_y", "Offline vs HLT vertex y position", 280, vtxYL, vtxYU, 280, vtxYL, vtxYU );
   H2_.back()["Vertex_OffvsHLT_z"] = dqm->book2D( "Vertex_OffvsHLT_z", "Offline vs HLT vertex z position", 400, vtxZL, vtxZU, 400, vtxZL, vtxZU );
      // axis titles
   H1_.back()["Vertex_HLT_x"]      -> setAxisTitle("HLT vtx x(cm)",1);
   H1_.back()["Vertex_HLT_y"]      -> setAxisTitle("HLT vtx y(cm)",1);
   H1_.back()["Vertex_HLT_z"]      -> setAxisTitle("HLT vtx z(cm)",1);
   H1_.back()["Vertex_Off_x"]      -> setAxisTitle("Offline vtx x(cm)",1);
   H1_.back()["Vertex_Off_y"]      -> setAxisTitle("Offline vtx y(cm)",1);
   H1_.back()["Vertex_Off_z"]      -> setAxisTitle("Offline vtx z(cm)",1);
   H2_.back()["Vertex_OffvsHLT_x"] -> setAxisTitle("HLT vtx x(cm)",1);
   H2_.back()["Vertex_OffvsHLT_y"] -> setAxisTitle("HLT vtx y(cm)",1);
   H2_.back()["Vertex_OffvsHLT_z"] -> setAxisTitle("HLT vtx z(cm)",1);
   H2_.back()["Vertex_OffvsHLT_x"] -> setAxisTitle("Offline vtx x(cm)",2);
   H2_.back()["Vertex_OffvsHLT_y"] -> setAxisTitle("Offline vtx y(cm)",2);
   H2_.back()["Vertex_OffvsHLT_z"] -> setAxisTitle("Offline vtx z(cm)",2);


std::cout<<"Booking of flavour-independent plots's been finished."<<std::endl;

/// mc related plots

/// 2D for efficiency calculations
///Pt turn-on
    int nBinsPt=30;
    double pTmin=0;
    double pTMax=300;


  for (unsigned int i = 0; i < m_mcLabels.size(); ++i)
{

     TString flavour= m_mcLabels[i].c_str();
     TString label;
/// 1D for L25 discriminator

      if ( l25JetTagCollection_[ind].label() != "" && l25JetTagCollection_[ind].label() != "NULL" ) {
     label="JetTag_L25_";
     label+=flavour;
	 H1_.back()[label.Data()] = 		 dqm->book1D(label.Data(),   Form("%s_%s",flavour.Data(),l25JetTagCollection_[ind].label().c_str()), btagBins, btagL, btagU );
     H1_.back()[label.Data()]->setAxisTitle("disc",1);

/// 2D for efficiency calculations
///Pt turn-on


    label="JetTag_L25_";
    label+=flavour+TString("_disc_pT");
    H2_.back()[label.Data()] =  dqm->book2D( label.Data(), label.Data(), btagBins, btagL, btagU, nBinsPt, pTmin, pTMax );
    H2_.back()[label.Data()]->setAxisTitle("pT",2);
    H2_.back()[label.Data()]->setAxisTitle("disc",1);


	}

/// 1D for L3 discriminator
	
	 if ( l3JetTagCollection_[ind].label() != "" && l3JetTagCollection_[ind].label() != "NULL" ) {
     label="JetTag_L3_";
     label+=flavour;
	 H1_.back()[label.Data()] = 		 dqm->book1D(label.Data(),   Form("%s_%s",flavour.Data(),l3JetTagCollection_[ind].label().c_str()), btagBins, btagL, btagU );
     H1_.back()[label.Data()]->setAxisTitle("disc",1);
	

/// 2D for efficiency calculations
///Pt turn-on

    label="JetTag_L3_";
    label+=flavour+TString("_disc_pT");
    H2_.back()[label.Data()] =  dqm->book2D( label.Data(), label.Data(), btagBins, btagL, btagU, nBinsPt, pTmin, pTMax );
    H2_.back()[label.Data()]->setAxisTitle("pT",2);
    H2_.back()[label.Data()]->setAxisTitle("disc",1);
	}

} /// for mc.size()

} /// for hltPathNames_.size()


std::cout<<"Booking of flavour-dependent plots's been finished."<<std::endl;
   
   triggerConfChanged_ = false;  
}




// ------------ method called once each job just after ending the event loop  ------------
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
//	if (it_mem>=0) hltPathIndexs_.push_back(it_mem);
//	else  hltPathIndexs_.push_back(99999);
    hltPathIndexs_.push_back(it_mem);
}

   for ( size_t trgs=0; trgs<hltPathNames_.size(); trgs++) {

//   if ( hltPathIndexs_[trgs] == 99999 ) {
   if ( hltPathIndexs_[trgs] < 0 ) {
   std::cout << "Path " << hltPathNames_[trgs] << " does not exist" << std::endl;
   _isfoundHLTs.push_back(false);
   } 
	else {
	   std::cout << "Path " << hltPathNames_[trgs] << "  exist" << std::endl;
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
