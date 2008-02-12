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
// $Id: XMLLUTLoader.h,v 1.3 2007/12/06 02:26:12 kukartse Exp $
//

// system include files

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"

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
    string kind_of_part;
    string name_label;
    string trig_prim_lookuptbl_data_file;
    int crate;
  } lutDBConfig;
  
  typedef struct _checksumsDBConfig : public XMLProcessor::DBConfig
  {
    _checksumsDBConfig();
    string comment_description;
    string name_label;
    string trig_prim_lookuptbl_data_file;
    int crate;
  } checksumsDBConfig;
  
  XMLLUTLoader();
  XMLLUTLoader( XMLProcessor::loaderBaseConfig * config, string templateBase = "HCAL_TRIG_PRIM_LOOKUP_TABLE.XMLloader.template" );
  virtual ~XMLLUTLoader();
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  int addLUT( lutDBConfig * config,
  	      string templateFileName = "HCAL_TRIG_PRIM_LOOKUP_TABLE.dataset.template" );
  
  int addChecksums( checksumsDBConfig * config,
  		    string templateFileName = "HCAL_TRIG_PRIM_LOOKUP_TABLE.checksums.template" );
  
  int createLoader( vector<int> crate_number, vector<string> file_name );
  
 private:
  XMLLUTLoader(const XMLLUTLoader&); // stop default
  
  const XMLLUTLoader& operator=(const XMLLUTLoader&); // stop default
  
  // ---------- member data --------------------------------
  
};


#endif
