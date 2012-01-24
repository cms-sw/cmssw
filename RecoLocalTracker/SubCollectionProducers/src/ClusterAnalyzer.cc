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
// $Id: ClusterAnalyzer.cc,v 1.5 2011/10/31 17:15:18 msegala Exp $
//
//


// system include files
#include <memory>

// user include files
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
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   
   
      // ----------member data ---------------------------

      edm::InputTag _class;

      std::map<int, std::string> enumModules_;
      std::map< std::string, TH1D* > histos1D_;
  
      std::vector<int>   modules_;  
  
      std::vector<int> nType_; 
      std::vector<double> clusterSize_; 
      std::vector<double> clusterCharge_;

      std::vector<std::string>  v_moduleTypes;
      std::vector<std::string>  v_variables;

      std::vector< std::vector<double> > genericVariables_; 

      bool _firstPass;
      edm::Service<TFileService> fs;
      std::vector<string> maps;

      bool _verbose;

      std::string ProvInfo;
      std::string ProvInfo_vars;

};


ClusterAnalyzer::ClusterAnalyzer(const edm::ParameterSet& iConfig)
{
  
  _class    = iConfig.getParameter<edm::InputTag>("clusterSum");

  _firstPass = true;
  _verbose = true;    //set to true to see the event by event summary info

}


ClusterAnalyzer::~ClusterAnalyzer(){}

void
ClusterAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   
   Handle< ClusterSummary  > class_;
   iEvent.getByLabel( _class, class_);
      
   nType_ = class_ -> GetNumberOfModules();		
   //int nTypeTOB_ = class_ -> GetNumberOfModules( ClusterSummary::TOB  );
   //int nTypeTIB_ = class_ -> GetNumberOfModules( ClusterSummary::TIB  );
   clusterSize_ = class_ -> GetClusterSize();
   //double clusterSizeTOB_ = class_ -> GetClusterSize( ClusterSummary::TOB  );
   //double clusterSizeTIB_ = class_ -> GetClusterSize( ClusterSummary::TIB  );
   clusterCharge_ = class_ -> GetClusterCharge();
   //double clusterChargeTOB_ = class_ -> GetClusterCharge( ClusterSummary::TOB  );
   //double clusterChargeTIB_ = class_ -> GetClusterCharge( ClusterSummary::TIB  );
   
   

   if (_firstPass){
   
     modules_ . clear();
     modules_ = class_ -> GetUserModules();

     //  Provenance Information
     const Provenance& prov = iEvent.getProvenance(class_.id());
     edm::ParameterSet pSet=getParameterSet( prov.psetID() );   
     ProvInfo = pSet.getParameter<string>("Module");
     cout << "From provenance infomation the selected modules are = "<< ProvInfo << endl;

     ProvInfo_vars = pSet.getParameter<string>("Variables");
     cout << "From provenance infomation the avaliable variables are = "<< ProvInfo_vars << endl;
     

     // Define the Modules to get the summary info out of  
     v_moduleTypes = class_ -> DecodeProvInfo( ProvInfo );
     v_variables = class_ -> DecodeProvInfo( ProvInfo_vars );

   }
   
   class_ -> SetUserContent( v_variables );

   /*
   genericVariables_ = class_ -> GetGenericVariable();   
   cout <<" 1st way " << endl;
   cout << genericVariables_.size() << endl;
   cout << genericVariables_[0][1] << endl;
   cout << genericVariables_[1][1] << endl;
   cout << genericVariables_[2][2] << endl;
   cout <<" 2ns way " << endl;
   cout << class_ -> GetGenericVariable("cHits", ClusterSummary::TIB) << endl;
   cout << class_ -> GetGenericVariable("cSize", ClusterSummary::TIB) << endl;
   cout << class_ -> GetGenericVariable("cCharge", ClusterSummary::TOB) << endl;
   */

   
   
   if ( _firstPass ){ //only do on the first event
 
     // Loop over all the modules to create a Map of Histograms to fill
     int n = 0;
     for ( vector<int>::const_iterator mod = modules_ . begin(); mod != modules_ . end(); mod++ ){

       //cout << "Creating histograms for " << *mod << endl;
       cout << "Creating histograms for " << v_moduleTypes.at(n) << endl;
       
       std::string tmpstr = v_moduleTypes.at(n);

       histos1D_[ (tmpstr + "nclusters").c_str() ] = fs->make< TH1D >( (tmpstr + "nclusters").c_str() , (tmpstr + "nclusters").c_str() , 1000 , 0 , 3000   ); 
       histos1D_[ (tmpstr + "nclusters").c_str() ]->SetXTitle( ("number of Clusters in " + tmpstr).c_str() );    
  
       histos1D_[ (tmpstr + "avgCharge").c_str() ] = fs->make< TH1D >( (tmpstr + "avgCharge").c_str() , (tmpstr + "avgCharge").c_str() , 500 , 0 , 1000   ); 
       histos1D_[ (tmpstr + "avgCharge").c_str() ]->SetXTitle( ("average cluster charge in " + tmpstr).c_str() );    
  
       histos1D_[ (tmpstr + "avgSize").c_str() ] = fs->make< TH1D >( (tmpstr + "avgSize").c_str() , (tmpstr + "avgSize").c_str() , 30 , 0 , 10   ); 
       histos1D_[ (tmpstr + "avgSize").c_str() ]->SetXTitle( ("average cluster size in " + tmpstr).c_str() );    

       maps.push_back( (tmpstr + "nclusters").c_str() );
       maps.push_back( (tmpstr + "avgSize").c_str() );
       maps.push_back( (tmpstr + "avgCharge").c_str() );
       
       ++n;

     }

     _firstPass = false;
   
   }

   

   int n = 0;
   for ( vector<int>::const_iterator mod = modules_ . begin(); mod != modules_ . end(); mod++ ){
  
     std::string tmpstr = v_moduleTypes.at(n);

     histos1D_[ (tmpstr + "nclusters").c_str() ] -> Fill( nType_.at(n)  );
     histos1D_[ (tmpstr + "avgSize").c_str()   ] -> Fill( clusterSize_.at(n)  );
     histos1D_[ (tmpstr + "avgCharge").c_str() ] -> Fill( clusterCharge_.at(n)  );

     ++n;

   }
   

}


// ------------ method called once each job just before starting event loop  ------------
void 
ClusterAnalyzer::beginJob(){}

//define this as a plug-in
DEFINE_FWK_MODULE(ClusterAnalyzer);
