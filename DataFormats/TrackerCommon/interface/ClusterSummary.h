// -*- C++ -*-
//
// Package:    ClusterSummary
// Class:      ClusterSummary
// 
/**\class ClusterSummary ClusterSummary.cc msegala/ClusterSummary/src/ClusterSummary.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Michael Segala
//         Created:  Wed Feb 23 17:36:23 CST 2011
// $Id: ClusterSummary.h,v 1.8 2011/10/31 17:15:05 msegala Exp $
//
//


#ifndef CLUSTERSUMMARY
#define CLUSTERSUMMARY

// system include files
#include <memory>
#include <string>
#include <map>
#include <vector>
#include<iostream>
#include <string.h>
#include <sstream>
#include "FWCore/Utilities/interface/Exception.h"

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"


#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h" 
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

/*****************************************************************************************

How to use ClusterSummary class:

ClusterSummary provides summary inforation for a number of cluster dependent variables.
All the variables are stored within variables_
The modules selected are stored within modules_
The number of variables for each module is stored within iterator_

[If putting ClusterSummary into producer]

1) In beginJob() of Producer, set the method   

  Cluster.SetUserContent(v_userContent);
  Cluster.SetUserIterator();

  where v_userContent is a vector of strings of the varibales you are getting info on
   

2) In produce() of Prodcuer, fill the output vectors

   // Tell ClusterSummary which Tracker Partitions you want summary information from
   Cluster.SetUserModules( mod ) 

   // Fill Summary ARRAYS with the summary information
   Cluster.SetNType( mod );
   Cluster.SetClusterSize( mod, Summaryinfo.clusterSize() );
   Cluster.SetClusterCharge( mod, Summaryinfo.charge() );

   --or--

   //Fill generic vector to hold any variables. You can fill the vector based on the name of the variables or the location of the variable within userContent
   cCluster.SetGenericVariable( "cHits", mod_pair2, 1 );
   cCluster.SetGenericVariable( "cSize", mod_pair2, Summaryinfo.clusterSize() );
   cCluster.SetGenericVariable( "cCharge", mod_pair2, Summaryinfo.charge() );


   // Once the loop over all detIds have finsihed fill the Output vectors
   Cluster.SetUserVariables( mod );


  // Dont forget to clear all the vectors and arrays at end of each event




[If putting reading back ClusterSummary from anlayzer]

   You can access all the summary vectors in the following way

   Handle< ClusterSummary  > class_;
   iEvent.getByLabel( _class, class_);
      
   nType_ = class_ -> GetNumberOfModules();
   clusterSize_ = class_ -> GetClusterSize();
   clusterCharge_ = class_ -> GetClusterCharge();

   GetNumberOfModules(), GetClusterSize(), GetClusterCharge() looks into the vector variables_ and unfolds it to get the proper information 




********************************************************************************************/


class ClusterSummary {

 public:
  
  ClusterSummary():genericVariables_(3, std::vector<double>(5000,0) ){}

  // Enum for each partition within Tracer
  enum CMSTracker { TRACKER = 0,
		    TIB = 1,
		    TIB_1 = 11, TIB_2 = 12, TIB_3 = 13, TIB_4 = 14,
		    TOB = 2,
		    TOB_1 = 21, TOB_2 = 22, TOB_3 = 23, TOB_4 = 24, TOB_5 = 25, TOB_6 = 26,
		    TID = 3,
		    TIDM = 31, TIDP = 32, 
		    TIDM_1 = 311, TIDM_2 = 312, TIDM_3 = 313,
		    TIDP_1 = 321, TIDP_2 = 322, TIDP_3 = 323,
		    TIDMR_1 = 3110, TIDMR_2 = 3120, TIDMR_3 = 3130,
		    TIDPR_1 = 3210, TIDPR_2 = 3220, TIDPR_3 = 3230,
		    TEC = 4,
		    TECM = 41, TECP = 42, 
		    TECM_1 = 411, TECM_2 = 412, TECM_3 = 413, TECM_4 = 414, TECM_5 = 415, TECM_6 = 416, TECM_7 = 417, TECM_8 = 418, TECM_9 = 419,
		    TECP_1 = 421, TECP_2 = 422, TECP_3 = 423, TECP_4 = 424, TECP_5 = 425, TECP_6 = 426, TECP_7 = 427, TECP_8 = 428, TECP_9 = 429, 
		    TECMR_1 = 4110, TECMR_2 = 4120, TECMR_3 = 4130, TECMR_4 = 4140, TECMR_5 = 4150, TECMR_6 = 4160, TECMR_7 = 4170, 
		    TECPR_1 = 4210, TECPR_2 = 4220, TECPR_3 = 4230, TECPR_4 = 4240, TECPR_5 = 4250, TECPR_6 = 4260, TECPR_7 = 4270,  
		    PIXELS = 5 };

  // Enum which describes the ordering of the summary variables inside vector variables_
  enum VariablePlacement{ NMODULES = 0,
			  CLUSTERSIZE = 1,
			  CLUSTERCHARGE = 2 };


  // Setter and Getter and Clear for number of cluster in a given module
  int GetNType( int module ){ return nType[module]; } 
  void SetNType( int module ){ nType[module]++; } 
  void SetNType( int module, int val ){ nType[module] = val; } 
  void ClearNType( int module ){ nType[module] = 0; } 

  // Setter and Getter and Clear for cluster size
  double GetClusterSize( int module ){ return ClusterSize[module]; } 
  double GetAverageClusterSize( int module ){ return (ClusterSize[module]/nType[module]); } 
  void SetClusterSize( int module, double size ){ ClusterSize[module] += size; } 
  void ClearClusterSize( int module ){ ClusterSize[module] = 0; } 

  // Setter and Getter and Clear for cluster charge
  double GetClusterCharge( int module ){ return ClusterCharge[module]; } 
  double GetAverageClusterCharge( int module ){ return (ClusterCharge[module]/nType[module]); } 
  void SetClusterCharge( int module, double charge ){ ClusterCharge[module] += charge; } 
  void ClearClusterCharge( int module ){ ClusterCharge[module] = 0; } 

  // Setter and Getter for the User Content. You can also return the size and what is stored in the UserContent 
  void SetUserContent(std::vector<std::string> Content)  const { userContent = Content;}
  std::vector<std::string> GetUserContent()  { return userContent;}
  int GetUserContentSize()  { return userContent.size(); }
  void  GetUserContentInfo()  { 
    std::cout << "Saving info for " ;
    for (unsigned int i = 0; i < userContent.size(); ++i){ std::cout << userContent.at(i) << " " ;}
    std::cout << std::endl;
  }



  // Setter and Getter for generic variable container
  double GetGenericVariable( int variableLocation, int module ) const { return genericVariables_[variableLocation][module]; }


  double GetGenericVariable( std::string variableName, int module ) const { 
    int position = -1;
    for (unsigned int i = 0; i < userContent.size(); ++i){
      if (variableName == userContent[i]) position = i; 
    }

    return genericVariables_[position][module];
  }
  
  
  std::vector< std::vector<double> > GetGenericVariable() const { return genericVariables_; }
  
  void SetGenericVariable( int variableLocation, int module, double value ) { genericVariables_[variableLocation][module] += value; } 


  void SetGenericVariable( std::string variableName, int module, double value ) { 
    int position = -1;
    for (unsigned int i = 0; i < userContent.size(); ++i){
      if (variableName == userContent[i]) position = i; 
    }
    genericVariables_[position][module] += value;    
  } 
 
 
  void ClearGenericVariable() { 
    
    //genericVariables_.clear();
    //genericVariables_[0].clear();
    //genericVariables_[1].clear();
    //genericVariables_[2].clear();

    ///\\\\ Temperary until figure out how to clear correctly

    for (int i = 0; i < 5000; ++i){
      genericVariables_[0][i] = 0;
      genericVariables_[1][i] = 0;
      genericVariables_[2][i] = 0;
    }
  } 



  /* 
     Fill methods for three output vectors (modules_, iterator_, variables_)
     The vectors are filled by accessing the proper location within appropriate array
     User fills the vectors by calling method 
     SetUserVariables( int module ) 
     where module is from the ENUM CMSTracker
  */

  //Set and Get modules_
  void SetUserModules( int value ) { modules_.push_back( value ); }
  std::vector<int> GetUserModules( ) const { return modules_;  }
  void ClearUserModules( ) { modules_.clear(); }

  //Set and Get iterator_
  void SetUserIterator() { iterator_.push_back( GetUserContentSize()  ); }
  std::vector<int> GetUserIterator() const { return iterator_; }
  void ClearUserIterator() { iterator_.clear(); }
 
  //Set and Get variables_
  void SetUserVariables( int module ){ 
    variables_.push_back(GetNType(module));
    variables_.push_back(GetAverageClusterSize(module));
    variables_.push_back(GetAverageClusterCharge(module));  
  }
  std::vector<double> GetUserVariables() const { return variables_; }
  void ClearUserVariables(){ variables_.clear(); }
  void ClearAllVariables(){variables_.clear(); }
  

   
  // Return the location of desired module within modules_. 
  int GetModuleLocation ( int mod ) const;

  // Return a vector of the number of clusters per module. This method looks into variables_ and collects the correct information
  std::vector<int> GetNumberOfModules() const; 

  // Return number of clusters for a given module. An example of the input for the method should be ClusterSummary::TIB
  int GetNumberOfModules( int mod ) const;
  
  // Return a vector of the average cluster size per module. This method looks into variables_ and collects the correct information
  std::vector<double> GetClusterSize() const;

  // Return average cluster size for a given module. An example of the input for the method should be ClusterSummary::TIB
  double GetClusterSize( int mod ) const;

  // Return a vector of the average cluster charge per module. This method looks into variables_ and collects the correct information
  std::vector<double> GetClusterCharge() const;

  // Return average cluster charge for a given module. An example of the input for the method should be ClusterSummary::TIB
  double GetClusterCharge( int mod ) const;  
  
  // Return a vector of the modules that summary infomation was requested for. This should come from the provenance information. 
  std::vector<std::string> DecodeProvInfo(std::string ProvInfo) const;

 
  // Class which determines if a detId is part of a desired partition
  class ModuleSelection{
  public:
    ModuleSelection(std::string gs){
      geosearch = gs;
    };
    virtual std::pair<int,int> IsSelected (int DetId);
  private:
    std::string geosearch; // string of selected modules	
  };




 private:
  
  //mutable int nType[64];
  //mutable double ClusterSize[64];
  //mutable double ClusterCharge[64];


  mutable int nType[5000];
  mutable double ClusterSize[5000];
  mutable double ClusterCharge[5000];


  // String which stores the name of the variables the user is getting the summary info for
  mutable std::vector<std::string>        userContent;

  std::vector<int>   iterator_;   // <number of varibale for Module1, number of varibale for Module2 ...>
  std::vector<int>   modules_;    // <Module1, Module2 ...>
  std::vector<double> variables_;  // <nClusters Module1, avg cluster size Module1, avg charge Module1, nClusters Module2 ...>


  std::vector< std::vector<double> > genericVariables_; 

};


#endif







