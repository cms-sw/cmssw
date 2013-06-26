#ifndef HCALConfigDBTools_XMLTools_XMLLUTLoader_h
#define HCALConfigDBTools_XMLTools_XMLLUTLoader_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLLUTLoader
// 
/**\class XMLLUTLoader XMLLUTLoader.h CaloOnlineTools/HcalOnlineDb/interface/XMLLUTLoader.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 23 14:30:33 CDT 2007
// $Id: XMLLUTLoader.h,v 1.5 2013/05/23 15:17:36 gartung Exp $
//

// system include files

// user include files
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"

// forward declarations

class XMLLUTLoader : public XMLDOMBlock
{
  
 public:
  
  typedef struct _loaderBaseConfig : public XMLProcessor::loaderBaseConfig
  {

  } loaderBaseConfig;
  
  typedef struct _lutDBConfig : public XMLProcessor::DBConfig
  {
    _lutDBConfig();
    std::string kind_of_part;
    std::string name_label;
    std::string trig_prim_lookuptbl_data_file;
    int crate;
  } lutDBConfig;
  
  typedef struct _checksumsDBConfig : public XMLProcessor::DBConfig
  {
    _checksumsDBConfig();
    std::string comment_description;
    std::string name_label;
    std::string trig_prim_lookuptbl_data_file;
    int crate;
  } checksumsDBConfig;
  
  XMLLUTLoader();
  XMLLUTLoader( XMLProcessor::loaderBaseConfig * config, std::string templateBase = "HCAL_TRIG_PRIM_LOOKUP_TABLE.XMLloader.template" );
  virtual ~XMLLUTLoader();
  
  int addLUT( lutDBConfig * config,
  	      std::string templateFileName = "HCAL_TRIG_PRIM_LOOKUP_TABLE.dataset.template" );
  
  int addChecksums( checksumsDBConfig * config,
  		    std::string templateFileName = "HCAL_TRIG_PRIM_LOOKUP_TABLE.checksums.template" );
  
  int createLoader( const std::vector<int>& crate_number, const std::vector<std::string>& file_name );
  
 private:
  XMLLUTLoader(const XMLLUTLoader&); // stop default
  
  const XMLLUTLoader& operator=(const XMLLUTLoader&); // stop default
  
  
};


#endif
