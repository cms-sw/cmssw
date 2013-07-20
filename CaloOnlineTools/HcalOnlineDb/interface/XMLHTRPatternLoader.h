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
// $Id: XMLHTRPatternLoader.h,v 1.4 2013/05/23 15:17:36 gartung Exp $
//

// system include files

// user include files
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"

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
    std::string kind_of_part;
    std::string name_label;
    std::string htr_data_patterns_data_file;
    int crate;
  } datasetDBConfig;
  
  typedef struct _checksumsDBConfig : public XMLProcessor::DBConfig
  {
    _checksumsDBConfig();
    std::string comment_description;
    std::string name_label;
    std::string htr_data_patterns_data_file;
    int crate;
  } checksumsDBConfig;
  
  XMLHTRPatternLoader();
  XMLHTRPatternLoader( XMLProcessor::loaderBaseConfig * config, std::string templateBase = "HCAL_HTR_DATA_PATTERNS.XMLloader.template" );
  virtual ~XMLHTRPatternLoader();
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  int addPattern( datasetDBConfig * config,
  	      std::string templateFileName = "HCAL_HTR_DATA_PATTERNS.dataset.template" );
  
  int addChecksums( checksumsDBConfig * config,
  		    std::string templateFileName = "HCAL_HTR_DATA_PATTERNS.checksums.template" );
  
  int createLoader( const std::vector<int>& crate_number, const std::vector<std::string>& file_name );
  
 private:
  XMLHTRPatternLoader(const XMLHTRPatternLoader&); // stop default
  
  const XMLHTRPatternLoader& operator=(const XMLHTRPatternLoader&); // stop default
  
  // ---------- member data --------------------------------
  
};


#endif
