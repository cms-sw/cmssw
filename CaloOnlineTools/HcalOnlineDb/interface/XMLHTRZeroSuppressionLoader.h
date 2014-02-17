#ifndef HCALConfigDBTools_XMLTools_XMLHTRZeroSuppressionLoader_h
#define HCALConfigDBTools_XMLTools_XMLHTRZeroSuppressionLoader_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLHTRZeroSuppressionLoader
// 
/**\class XMLHTRZeroSuppressionLoader XMLHTRZeroSuppressionLoader.h CaloOnlineTools/HcalOnlineDb/interface/XMLHTRZeroSuppressionLoader.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Nov 06 14:30:33 CDT 2007
// $Id: XMLHTRZeroSuppressionLoader.h,v 1.4 2010/08/06 20:24:10 wmtan Exp $
//

// system include files

// user include files
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"

// forward declarations

class XMLHTRZeroSuppressionLoader : public XMLDOMBlock
{
  
 public:
  
  typedef struct _loaderBaseConfig
  {
    _loaderBaseConfig();
    std::string extention_table_name;
    std::string name;
    std::string run_type;
    long long int run_number;
    time_t run_begin_timestamp;
    std::string comment_description;
    std::string data_set_id;
    std::string iov_id;
    long long int iov_begin;
    long long int iov_end;
    std::string tag_id;
    std::string tag_mode;
    std::string tag_name;
    std::string detector_name;
    std::string elements_comment_description;
  } loaderBaseConfig;
  
  typedef struct _datasetDBConfig : public XMLProcessor::DBConfig
  {
    _datasetDBConfig();
    std::string comment_description;
    std::string extention_table_name;
    int eta, phi, depth;
    int z;
    long long int hcal_channel_id;
    std::string detector_name;
    int zero_suppression;
  } datasetDBConfig;
  
  XMLHTRZeroSuppressionLoader();
  XMLHTRZeroSuppressionLoader( loaderBaseConfig * config, std::string templateBase = "HCAL_HTR_ZERO_SUPPRESSION.XMLloader.template" );
  virtual ~XMLHTRZeroSuppressionLoader();

  // deprecated - to be removed
  //int createLoader( void );
  
  int addZS( datasetDBConfig * config,
	     std::string templateFileName = "HCAL_HTR_ZERO_SUPPRESSION.dataset.template" );
  
 private:
  XMLHTRZeroSuppressionLoader(const XMLHTRZeroSuppressionLoader&); // stop default
  
  const XMLHTRZeroSuppressionLoader& operator=(const XMLHTRZeroSuppressionLoader&); // stop default
  
  // ---------- member data --------------------------------
  
};


#endif
