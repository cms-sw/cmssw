#ifndef HCALConfigDBTools_XMLTools_XMLHTRPatternLoader_h
#define HCALConfigDBTools_XMLTools_XMLHTRPatternLoader_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLHTRPatternLoader
// 
/**\class XMLHTRPatternLoader XMLHTRPatternLoader.h CaloOnlineTools/HcalOnlineDb/interface/XMLHTRPatternLoader.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 23 14:30:33 CDT 2007
// $Id: XMLHTRPatternLoader.h,v 1.3 2007/12/06 02:26:09 kukartse Exp $
//

// system include files

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"

// forward declarations

class XMLHTRPatternLoader : public XMLDOMBlock
{
  
 public:
  
  typedef struct _loaderBaseConfig : public XMLProcessor::loaderBaseConfig
  {
    _loaderBaseConfig();
  } loaderBaseConfig;
  
  typedef struct _datasetDBConfig : public XMLProcessor::DBConfig
  {
    _datasetDBConfig();
    string kind_of_part;
    string name_label;
    string htr_data_patterns_data_file;
    int crate;
  } datasetDBConfig;
  
  typedef struct _checksumsDBConfig : public XMLProcessor::DBConfig
  {
    _checksumsDBConfig();
    string comment_description;
    string name_label;
    string htr_data_patterns_data_file;
    int crate;
  } checksumsDBConfig;
  
  XMLHTRPatternLoader();
  XMLHTRPatternLoader( XMLProcessor::loaderBaseConfig * config, string templateBase = "HCAL_HTR_DATA_PATTERNS.XMLloader.template" );
  virtual ~XMLHTRPatternLoader();
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  int addPattern( datasetDBConfig * config,
  	      string templateFileName = "HCAL_HTR_DATA_PATTERNS.dataset.template" );
  
  int addChecksums( checksumsDBConfig * config,
  		    string templateFileName = "HCAL_HTR_DATA_PATTERNS.checksums.template" );
  
  int createLoader( vector<int> crate_number, vector<string> file_name );
  
 private:
  XMLHTRPatternLoader(const XMLHTRPatternLoader&); // stop default
  
  const XMLHTRPatternLoader& operator=(const XMLHTRPatternLoader&); // stop default
  
  // ---------- member data --------------------------------
  
};


#endif
