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
// $Id: XMLHTRZeroSuppressionLoader.h,v 1.1 2008/02/12 17:01:59 kukartse Exp $
//

// system include files

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"

// forward declarations

class XMLHTRZeroSuppressionLoader : public XMLDOMBlock
{
  
 public:
  
  typedef struct _loaderBaseConfig
  {
    _loaderBaseConfig();
    string extention_table_name;
    string name;
    string run_type;
    long long int run_number;
    time_t run_begin_timestamp;
    string comment_description;
    string data_set_id;
    string iov_id;
    long long int iov_begin;
    long long int iov_end;
    string tag_id;
    string tag_mode;
    string tag_name;
    string detector_name;
    string elements_comment_description;
  } loaderBaseConfig;
  
  typedef struct _datasetDBConfig : public XMLProcessor::DBConfig
  {
    _datasetDBConfig();
    string comment_description;
    string extention_table_name;
    int eta, phi, depth;
    int z;
    long long int hcal_channel_id;
    string detector_name;
    int zero_suppression;
  } datasetDBConfig;
  
  XMLHTRZeroSuppressionLoader();
  XMLHTRZeroSuppressionLoader( loaderBaseConfig * config, string templateBase = "HCAL_HTR_ZERO_SUPPRESSION.XMLloader.template" );
  virtual ~XMLHTRZeroSuppressionLoader();

  // deprecated - to be removed
  //int createLoader( void );
  
  int addZS( datasetDBConfig * config,
	     string templateFileName = "HCAL_HTR_ZERO_SUPPRESSION.dataset.template" );
  
 private:
  XMLHTRZeroSuppressionLoader(const XMLHTRZeroSuppressionLoader&); // stop default
  
  const XMLHTRZeroSuppressionLoader& operator=(const XMLHTRZeroSuppressionLoader&); // stop default
  
  // ---------- member data --------------------------------
  
};


#endif
