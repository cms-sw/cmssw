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
//
//

#ifndef CLUSTERSUMMARY
#define CLUSTERSUMMARY

// system include files
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#endif
#include <memory>
#include <string>
#include <map>
#include <vector>
#include<iostream>
#include <string.h>
#include <sstream>
#include "FWCore/Utilities/interface/Exception.h"

// user include files

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"


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
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"

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


   //Fill generic vector to hold any variables. You can fill the vector based on the name of the variables or the location of the variable within userContent
   cCluster.SetGenericVariable( "sHits", mod_pair2, 1 );
   cCluster.SetGenericVariable( "sSize", mod_pair2, Summaryinfo.clusterSize() );
   cCluster.SetGenericVariable( "sCharge", mod_pair2, Summaryinfo.charge() );


   // Once the loop over all detIds have finsihed fill the Output vectors
   Cluster.SetUserVariables( mod );


  // Dont forget to clear all the vectors and arrays at end of each event




[If putting reading back ClusterSummary from anlayzer]

   You can access all the summary vectors in the following way

   Handle< ClusterSummary  > class_;
   iEvent.getByLabel( _class, class_);
      
   genericVariables_ = class_ -> GetGenericVariable();   

   //You can access the variables by genericVariables_[i][j] where 'i' is the order in which the variable was stored, see enum VariablePlacement
   cout << genericVariables_[0][1] << endl;
   cout << genericVariables_[1][1]/genericVariables_[0][1] << endl;
   cout << genericVariables_[2][2]/genericVariables_[0][2] << endl;

   --or--
   
   //You can access the variables by the variable and partition name.
   cout << class_ -> GetGenericVariable("cHits", ClusterSummary::TIB) << endl;
   cout << class_ -> GetGenericVariable("cSize", ClusterSummary::TIB)/class_ -> GetGenericVariable("cHits", ClusterSummary::TIB) << endl;
   cout << class_ -> GetGenericVariable("cCharge", ClusterSummary::TOB)/class_ -> GetGenericVariable("cHits", ClusterSummary::TOB) << endl;




********************************************************************************************/


class ClusterSummary {

 public:
  
  ClusterSummary();
  ~ClusterSummary();
  // copy ctor
  ClusterSummary(const ClusterSummary& src);
  // copy assingment operator
  ClusterSummary& operator=(const ClusterSummary& rhs);
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  ClusterSummary(ClusterSummary&& other);
#endif

  // Enum for each partition within Tracker
  enum CMSTracker { TRACKER = 0,
		    TIB = 1,
		    TIB_1 = 11, TIB_2 = 12, TIB_3 = 13, TIB_4 = 14, //TIB layer 1-4
		    TOB = 2,
		    TOB_1 = 21, TOB_2 = 22, TOB_3 = 23, TOB_4 = 24, TOB_5 = 25, TOB_6 = 26,  //TOB layer 1-6
		    TID = 3,
		    TIDM = 31, TIDP = 32,  //TID minus and plus
		    TIDM_1 = 311, TIDM_2 = 312, TIDM_3 = 313, //TID minus layer 1-3
		    TIDP_1 = 321, TIDP_2 = 322, TIDP_3 = 323, //TID plus layer 1-3
		    TIDMR_1 = 3110, TIDMR_2 = 3120, TIDMR_3 = 3130, //TID minus ring 1-3
		    TIDPR_1 = 3210, TIDPR_2 = 3220, TIDPR_3 = 3230, //TID plus ring 1-3
		    TEC = 4,
		    TECM = 41, TECP = 42,  //TEC minus and plus
		    TECM_1 = 411, TECM_2 = 412, TECM_3 = 413, TECM_4 = 414, TECM_5 = 415, TECM_6 = 416, TECM_7 = 417, TECM_8 = 418, TECM_9 = 419, //TEC minus layer 1-9
		    TECP_1 = 421, TECP_2 = 422, TECP_3 = 423, TECP_4 = 424, TECP_5 = 425, TECP_6 = 426, TECP_7 = 427, TECP_8 = 428, TECP_9 = 429, //TEC plus layer 1-9 
		    TECMR_1 = 4110, TECMR_2 = 4120, TECMR_3 = 4130, TECMR_4 = 4140, TECMR_5 = 4150, TECMR_6 = 4160, TECMR_7 = 4170, //TEC minus ring 1-9
		    TECPR_1 = 4210, TECPR_2 = 4220, TECPR_3 = 4230, TECPR_4 = 4240, TECPR_5 = 4250, TECPR_6 = 4260, TECPR_7 = 4270, //TEC plus ring 1-9
		    //PIXELS
		    PIXEL = 5,
		    FPIX = 6, // Pixel endcaps
		    FPIX_1 = 61,FPIX_2 = 62,FPIX_3 = 63, // Endcaps disks 1-3
		    FPIXM = 611, FPIXP = 612,  // Pixel endcaps minus and plus side
		    FPIXM_1 = 6110, FPIXM_2 = 6120, FPIXM_3 = 6130, // Endcap minus disk 1-3  
		    FPIXP_1 = 6210, FPIXP_2 = 6220, FPIXP_3 = 6230, // Endcap plus disk 1-3  
		    BPIX = 7, //Pixel barrel
		    BPIX_1 = 71, BPIX_2 = 72, BPIX_3 = 73 //Pixel barrel layer 1-3
		
  };

  // Enum which describes the ordering of the summary variables inside vector variables_
  enum VariablePlacement{
        NMODULES = 0,
			  CLUSTERSIZE = 1,
			  CLUSTERCHARGE = 2,
			  NVARIABLES = 3
  };

  static bool checkSubDet(const int input); //Returns true for pixel, throws an exception if it does not exist
  static std::string getSubDetName(const CMSTracker subdet); //Returns the string form of the subdet
  static std::string getVarName(const VariablePlacement var); //Returns the string form of the variable


  //===================+++++++++++++========================
  //
  //                 Main methods to fill 
  //                      Variables
  //
  //===================+++++++++++++========================
  
  //These functions are broken into two categories. The standard versions take the enums as input and find the locations in the vector.
  //The ones labeled "byIndex" take the vector location as input

 private:
  void checkModule(const int moduleLocation,const unsigned int vSize) const {if(moduleLocation >= int(vSize)) throw cms::Exception( "Missing module") << moduleLocation;}
 public:
  int   getNModulesByIndex  (const int mod) const {checkModule(mod,nModules  .size()); return nModules  [mod];}
  int   getClusSizeByIndex  (const int mod) const {checkModule(mod,clusSize  .size()); return clusSize  [mod];}
  float getClusChargeByIndex(const int mod) const {checkModule(mod,clusCharge.size()); return clusCharge[mod];}

  int   getNModules  (const CMSTracker mod) const {int pos = GetModuleLocation(mod); return pos < 0 ? 0. : nModules  [pos];}
  int   getClusSize  (const CMSTracker mod) const {int pos = GetModuleLocation(mod); return pos < 0 ? 0. : clusSize  [pos];}
  float getClusCharge(const CMSTracker mod) const {int pos = GetModuleLocation(mod); return pos < 0 ? 0. : clusCharge[pos];}

  std::vector<int>   getNModulesVector()   const {return nModules;}
  std::vector<int>   getClusSizeVector()   const {return clusSize;}
  std::vector<float> getClusChargeVector() const {return clusCharge;}

  void setNModulesByIndex  (const int mod, const int   val) const {checkModule(mod,nModules_tmp  .size()); nModules_tmp  [mod]+=val;}
  void setClusSizeByIndex  (const int mod, const int   val) const {checkModule(mod,clusSize_tmp  .size()); clusSize_tmp  [mod]+=val;}
  void setClusChargeByIndex(const int mod, const float val) const {checkModule(mod,clusCharge_tmp.size()); clusCharge_tmp[mod]+=val;}

  void setNModules  (const CMSTracker mod, const int   val) const {int pos = GetModuleLocation(mod); nModules_tmp  [pos]+=val;}
  void setClusSize  (const CMSTracker mod, const int   val) const {int pos = GetModuleLocation(mod); clusSize_tmp  [pos]+=val;}
  void setClusCharge(const CMSTracker mod, const float val) const {int pos = GetModuleLocation(mod); clusCharge_tmp[pos]+=val;}

  //Prepair the final vector to be put into the producer. Remove any remaining 0's and copy the Tmp to the vector over to genericVariables_. Must be done at the end of each event.
  void PrepairGenericVariable();

  //Clear genericVariablesTmp_. Must be done at the end of each event.
  void ClearGenericVariable() { 
    for(unsigned int i = 0; i < nModules_tmp.size(); ++i) nModules_tmp[i] = 0;
    for(unsigned int i = 0; i < clusSize_tmp.size(); ++i) clusSize_tmp[i] = 0;
    for(unsigned int i = 0; i < clusCharge_tmp.size(); ++i) clusCharge_tmp[i] = 0;
  } 

  //Set and Get modules_
  void SetUserModules( const CMSTracker value ) { modules.push_back( value ); }
  std::vector<int> GetUserModules() const { return modules;  }
  void ClearUserModules( ) { modules.clear(); }
  // Return the location of desired module within modules_. If warn is set to true, a warnign will be outputed in case no module was found
  int GetModuleLocation ( int mod, bool warn = true ) const;
  unsigned int GetNumberOfModules() const {return modules.size();}
  int GetModule(const int index) const { return modules[index];}
    
  // Return a vector of the modules that summary infomation was requested for. This should come from the provenance information. 
  std::vector<std::string> DecodeProvInfo(std::string ProvInfo) const;

 private:
  std::vector<int>   modules;    // <Module1, Module2 ...>
  std::vector<int>   nModules;
  std::vector<int>   clusSize;
  std::vector<float> clusCharge;

  // CMS-THREADSAFE: this mutable member data used in non-const functions
  mutable std::vector<int>   nModules_tmp;
  mutable std::vector<int>   clusSize_tmp;
  mutable std::vector<float> clusCharge_tmp;

};


#endif







