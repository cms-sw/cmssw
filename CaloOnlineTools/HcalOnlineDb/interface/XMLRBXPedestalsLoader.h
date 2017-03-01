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
//

// system include files

// user include files
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"

// forward declarations

class XMLRBXPedestalsLoader : public XMLDOMBlock
{
  
 public:
  
  typedef struct _loaderBaseConfig
  {
    _loaderBaseConfig();
    std::string extention_table_name;
    std::string name;
    std::string run_mode;
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
    std::string name_label;
    std::string kind_of_part;
  } datasetDBConfig;
  
  //XMLRBXPedestalsLoader();
  XMLRBXPedestalsLoader( loaderBaseConfig * config, std::string templateBase = "HCAL_RBX_PEDESTALS_TYPE01.XMLloader.template" );
  virtual ~XMLRBXPedestalsLoader();

  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  int addRBXSlot( datasetDBConfig * config, std::string brickFileName,
		  std::string rbx_config_type = "pedestals",
		  std::string templateFileName = "HCAL_RBX_PEDESTALS_TYPE01.dataset.template" ); // pedestals, gols, delays
  
 protected:
  int init( void );
  int fixRbxName( std::string & );

  XERCES_CPP_NAMESPACE::MemBufInputSource * _data_ped_delay; // a container for the XML template for a single pedestal or zero delay
  XERCES_CPP_NAMESPACE::MemBufInputSource * _data_gol; // a container for the XML template for a single gol current
  XERCES_CPP_NAMESPACE::MemBufInputSource * _data_led; // a container for the XML template for a single led data unit

 private:
  XMLRBXPedestalsLoader(const XMLRBXPedestalsLoader&); // stop default
  
  const XMLRBXPedestalsLoader& operator=(const XMLRBXPedestalsLoader&); // stop default
  
  // ---------- member data --------------------------------
  
};

#endif
