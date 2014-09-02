// -*- C++ -*-
//
// Package:    ClusterAnalyzer
// Class:      ClusterAnalyzer
// 
/**\class ClusterAnalyzer ClusterAnalyzer.cc msegala/ClusterSummary/src/ClusterAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Michael Segala
//         Created:  Wed Feb 23 17:36:23 CST 2011
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TTree.h"
#include "FWCore/ServiceRegistry/interface/Service.h" 
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH1.h>
#include <TFile.h>
#include "DataFormats/TrackerCommon/interface/ClusterSummary.h"


//
// class declaration
//

class ClusterSummary;


using namespace std;

class ClusterAnalyzer : public edm::EDAnalyzer {
   public:
      explicit ClusterAnalyzer(const edm::ParameterSet&);
      ~ClusterAnalyzer();


   private:
      virtual void beginJob() override ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
   
   
      // ----------member data ---------------------------

      edm::EDGetTokenT<ClusterSummary> token;

      std::map<int, std::string> enumModules_;
      std::map< std::string, TH1D* > histos1D_;

      std::vector<int> nType_; 
      std::vector<float> clusterSize_;
      std::vector<float> clusterCharge_;

      std::map<int,std::string> allModules_;

      bool _firstPass;
      edm::Service<TFileService> fs;

      bool _verbose;

      bool doStrips;
      bool doPixels;
};


ClusterAnalyzer::ClusterAnalyzer(const edm::ParameterSet& iConfig)
{
  
  token = consumes<ClusterSummary>(iConfig.getParameter<edm::InputTag>("clusterSum"));

  _firstPass = true;
  _verbose = true;    //set to true to see the event by event summary info

}


ClusterAnalyzer::~ClusterAnalyzer(){}

void
ClusterAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   
   Handle< ClusterSummary  > class_;
   iEvent.getByToken( token, class_);
      
   if (_firstPass){
     //  Provenance Information
     const Provenance& prov = iEvent.getProvenance(class_.id());
     const edm::ParameterSet& pSet = parameterSet(prov);   

     doStrips = pSet.getParameter<bool>("doStrips");
     doPixels = pSet.getParameter<bool>("doPixels");

     std::vector<edm::ParameterSet> wantedsubdets_ps =  pSet.getParameter<std::vector<edm::ParameterSet> >("wantedSubDets");
     for(std::vector<edm::ParameterSet>::const_iterator wsdps = wantedsubdets_ps.begin();wsdps!=wantedsubdets_ps.end();++wsdps) {
       unsigned int             detsel    = wsdps->getParameter<unsigned int>("detSelection");
       std::string              detname   = wsdps->getParameter<std::string>("detLabel");

       if(ClusterSummary::checkSubDet(detsel)){
         if(doPixels)
           allModules_[detsel] =detname;
       } else{
         if(doStrips)allModules_[detsel] =detname;
       }
     }

     cout << "From provenance infomation the selected modules are = ";
     for( auto i = allModules_.begin(); i != allModules_.end(); ++i)
      cout << i->second <<" ";
     cout << endl;
   }

   if ( _firstPass ){ //only do on the first event

     for( auto i = allModules_.begin(); i != allModules_.end(); ++i){
       std::string tmpstr = i->second;
       histos1D_[ (tmpstr + "nclusters").c_str() ] = fs->make< TH1D >( (tmpstr + "nclusters").c_str() , (tmpstr + "nclusters").c_str() , 1000 , 0 , 3000   );
       histos1D_[ (tmpstr + "nclusters").c_str() ]->SetXTitle( ("number of Clusters in " + tmpstr).c_str() );

       histos1D_[ (tmpstr + "avgCharge").c_str() ] = fs->make< TH1D >( (tmpstr + "avgCharge").c_str() , (tmpstr + "avgCharge").c_str() , 500 , 0 , 1000   );
       histos1D_[ (tmpstr + "avgCharge").c_str() ]->SetXTitle( ("average cluster charge in " + tmpstr).c_str() );

       histos1D_[ (tmpstr + "avgSize").c_str() ] = fs->make< TH1D >( (tmpstr + "avgSize").c_str() , (tmpstr + "avgSize").c_str() , 30 , 0 , 10   );
       histos1D_[ (tmpstr + "avgSize").c_str() ]->SetXTitle( ("average cluster size in " + tmpstr).c_str() );
       
     }

     _firstPass = false;
   }
   
   for ( unsigned int iM  = 0; iM < class_->GetNumberOfModules(); iM++ ){
     auto iAllModules = allModules_.find(class_->GetModule(iM));
     if(iAllModules == allModules_.end()) continue;

     std::string tmpstr = iAllModules->second;

     histos1D_[ (tmpstr + "nclusters").c_str() ] -> Fill( class_ -> getNModulesByIndex  (iM ));
     histos1D_[ (tmpstr + "avgSize").c_str()   ] -> Fill( class_ -> getClusSizeByIndex  (iM )/class_ -> getNModulesByIndex(iM ));
     histos1D_[ (tmpstr + "avgCharge").c_str() ] -> Fill( class_ -> getClusChargeByIndex(iM )/class_ -> getNModulesByIndex(iM ));

     cout << "n"<<tmpstr <<", avg size, avg charge = "
                 << class_ -> getNModulesByIndex  (iM );
     cout << ", "<< class_ -> getClusSizeByIndex  (iM)  /class_ -> getNModulesByIndex(iM );
     cout << ", "<< class_ -> getClusChargeByIndex(iM)  /class_ -> getNModulesByIndex(iM ) <<  endl;
   }
   cout << "-------------------------------------------------------" << endl;
   

}


// ------------ method called once each job just before starting event loop  ------------
void 
ClusterAnalyzer::beginJob(){}

//define this as a plug-in
DEFINE_FWK_MODULE(ClusterAnalyzer);
