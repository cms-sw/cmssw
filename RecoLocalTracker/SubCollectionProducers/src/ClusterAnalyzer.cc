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
// $Id: ClusterAnalyzer.cc,v 1.4 2012/10/03 13:27:28 msegala Exp $
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
      std::string ProvInfoPixels;
      std::string ProvInfoPixels_vars;

      bool doStrips;
      bool doPixels;
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
      
   if (_firstPass){
   
     modules_ . clear();
     modules_ = class_ -> GetUserModules();

     //  Provenance Information
     const Provenance& prov = iEvent.getProvenance(class_.id());
     edm::ParameterSet pSet=getParameterSet( prov.psetID() );   

     ProvInfo = "";
     ProvInfo_vars = "";
     ProvInfoPixels = "";
     ProvInfoPixels_vars = "";

     std::string ProvString = "";
     std::string VarString = "";
       
     doStrips = pSet.getParameter<bool>("doStrips");
     doPixels = pSet.getParameter<bool>("doPixels");

     if (doStrips){
       ProvInfo = pSet.getParameter<string>("stripModule");
       cout << "From provenance infomation the selected strip modules are = "<< ProvInfo << endl;
       ProvInfo_vars = pSet.getParameter<string>("stripVariables");
       cout << "From provenance infomation the avaliable strip variables are = "<< ProvInfo_vars << endl;

     }
     if (doPixels){
       ProvInfoPixels = pSet.getParameter<string>("pixelModule");
       cout << "From provenance infomation the selected pixel modules are = "<< ProvInfoPixels << endl;
       ProvInfoPixels_vars = pSet.getParameter<string>("pixelVariables");
       cout << "From provenance infomation the avaliable pixel variables are = "<< ProvInfoPixels_vars << endl;
     }


     if (doStrips && doPixels) {
       ProvString = ProvInfo + "," + ProvInfoPixels;
       VarString = ProvInfo_vars + "," + ProvInfoPixels_vars;
     }
     else if (doStrips && !doPixels) {
       ProvString = ProvInfo;
       VarString = ProvInfo_vars;
     }
     else if (!doStrips && doPixels) {
       ProvString = ProvInfoPixels;
       VarString = ProvInfoPixels_vars;
     }
     
     // Define the Modules to get the summary info out of  
     v_moduleTypes = class_ -> DecodeProvInfo( ProvString );
     v_variables = class_ -> DecodeProvInfo( VarString );

   }
   
   class_ -> SetUserContent( v_variables );   
	
   genericVariables_ = class_ -> GetGenericVariable();   
   
   //cout << class_ -> GetGenericVariable("cHits", ClusterSummary::TIB) << endl;

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

     //Trick to see if it comes from a strip or pixel variable. If the first digit is < 6 then it is from the strips.
     int mod_tmp = *mod;
     while (mod_tmp > 9 ){
       mod_tmp /= 10;
     }
     
     if ( mod_tmp < 5 ){

       histos1D_[ (tmpstr + "nclusters").c_str() ] -> Fill( class_ -> GetGenericVariable("cHits", *mod) );
       histos1D_[ (tmpstr + "avgSize").c_str()   ] -> Fill( class_ -> GetGenericVariable("cSize", *mod)  /class_ -> GetGenericVariable("cHits", *mod) );
       histos1D_[ (tmpstr + "avgCharge").c_str() ] -> Fill( class_ -> GetGenericVariable("cCharge", *mod)/class_ -> GetGenericVariable("cHits", *mod) );

       cout << "n"<<tmpstr <<", avg size, avg charge = "<< class_ -> GetGenericVariable( "cHits",*mod ); 
       cout << ", "<< class_ -> GetGenericVariable( "cSize",*mod )  /class_ -> GetGenericVariable( "cHits",*mod ); 
       cout << ", "<< class_ -> GetGenericVariable( "cCharge",*mod )/class_ -> GetGenericVariable( "cHits",*mod ) <<  endl;

     }
     else{
       histos1D_[ (tmpstr + "nclusters").c_str() ] -> Fill( class_ -> GetGenericVariable("pHits", *mod) );
       histos1D_[ (tmpstr + "avgSize").c_str()   ] -> Fill( class_ -> GetGenericVariable("pSize", *mod)/class_ -> GetGenericVariable("pHits", *mod) );
       histos1D_[ (tmpstr + "avgCharge").c_str() ] -> Fill( class_ -> GetGenericVariable("pCharge", *mod)/class_ -> GetGenericVariable("pHits", *mod) );
       
       cout << "n"<<tmpstr <<", avg size, avg charge = "<< class_ -> GetGenericVariable( "pHits",*mod );
       cout << ", "<< class_ -> GetGenericVariable( "pSize",*mod )  /class_ -> GetGenericVariable( "pHits",*mod );
       cout << ", "<< class_ -> GetGenericVariable( "pCharge",*mod )/class_ -> GetGenericVariable( "pHits",*mod ) <<  endl;

     }
     
     ++n;

   }
   
   cout << "-------------------------------------------------------" << endl;
   

}


// ------------ method called once each job just before starting event loop  ------------
void 
ClusterAnalyzer::beginJob(){}

//define this as a plug-in
DEFINE_FWK_MODULE(ClusterAnalyzer);
