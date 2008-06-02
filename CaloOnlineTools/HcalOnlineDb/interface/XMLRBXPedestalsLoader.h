#ifndef HCALConfigDBTools_XMLTools_XMLRBXPedestalsLoader_h
#define HCALConfigDBTools_XMLTools_XMLRBXPedestalsLoader_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLRBXPedestalsLoader
// 
/**\class XMLRBXPedestalsLoader XMLRBXPedestalsLoader.h CaloOnlineTools/HcalOnlineDb/interface/XMLRBXPedestalsLoader.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Nov 06 14:30:33 CDT 2007
// $Id: XMLRBXPedestalsLoader.h,v 1.1 2007/12/06 02:26:12 kukartse Exp $
//

// system include files

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"

// forward declarations

class XMLRBXPedestalsLoader : public XMLDOMBlock
{
  
 public:
  
  typedef struct _loaderBaseConfig
  {
    _loaderBaseConfig();
    string extention_table_name;
    string name;
    string run_mode;
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
    string name_label;
    string kind_of_part;
  } datasetDBConfig;
  
  //XMLRBXPedestalsLoader();
  XMLRBXPedestalsLoader( loaderBaseConfig * config, string templateBase = "HCAL_RBX_PEDESTALS_TYPE01.XMLloader.template" );
  virtual ~XMLRBXPedestalsLoader();

  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  int addRBXSlot( datasetDBConfig * config, string brickFileName,
	     string templateFileName = "HCAL_RBX_PEDESTALS_TYPE01.dataset.template" );
  
 protected:
  int init( void );
  int fixRbxName( string & );

  MemBufInputSource * _data; // a container for the XML template for a single pedestal

 private:
  XMLRBXPedestalsLoader(const XMLRBXPedestalsLoader&); // stop default
  
  const XMLRBXPedestalsLoader& operator=(const XMLRBXPedestalsLoader&); // stop default
  
  // ---------- member data --------------------------------
  
};

#endif
