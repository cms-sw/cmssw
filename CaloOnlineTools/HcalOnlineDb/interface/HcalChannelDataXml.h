#ifndef CaloOnlineTools_HcalOnlineDb_HcalChannelDataXml_h
#define CaloOnlineTools_HcalOnlineDb_HcalChannelDataXml_h
// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     HcalChannelDataXml
// 
/**\class HcalChannelDataXml HcalChannelDataXml.h CaloOnlineTools/HcalOnlineDb/interface/HcalChannelDataXml.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Wed Jul 01 06:42:00 CDT 2009
//

#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalAssistant.h"

class HcalChannelDataXml : public XMLDOMBlock
{
  
 public:
      
  HcalChannelDataXml();
  virtual ~HcalChannelDataXml();
  
  int init_data( void );

  //
  //_____ methods to set basic tags
  //
  XERCES_CPP_NAMESPACE::DOMNode * set_header_table_name(std::string name);
  XERCES_CPP_NAMESPACE::DOMNode * set_header_type(std::string type);
  XERCES_CPP_NAMESPACE::DOMNode * set_header_run_number(int run);
  XERCES_CPP_NAMESPACE::DOMNode * set_header_channel_map(std::string name);
  XERCES_CPP_NAMESPACE::DOMNode * set_elements_dataset_id(int id);
  XERCES_CPP_NAMESPACE::DOMNode * set_elements_iov_id(int id);
  XERCES_CPP_NAMESPACE::DOMNode * set_elements_iov_begin(int value);
  XERCES_CPP_NAMESPACE::DOMNode * set_elements_iov_end(int value);
  XERCES_CPP_NAMESPACE::DOMNode * set_elements_tag_id(int value);
  XERCES_CPP_NAMESPACE::DOMNode * set_elements_tag_mode(std::string value);
  XERCES_CPP_NAMESPACE::DOMNode * set_elements_tag_name(std::string value);
  XERCES_CPP_NAMESPACE::DOMNode * set_elements_detector_name(std::string value);
  XERCES_CPP_NAMESPACE::DOMNode * set_elements_comment(std::string value);
  XERCES_CPP_NAMESPACE::DOMNode * set_maps_tag_idref(int value);
  XERCES_CPP_NAMESPACE::DOMNode * set_maps_iov_idref(int value);
  XERCES_CPP_NAMESPACE::DOMNode * set_maps_dataset_idref(int value);

  //
  //_____ add data
  //
  XERCES_CPP_NAMESPACE::DOMNode * add_dataset( void );
  XERCES_CPP_NAMESPACE::DOMNode * add_hcal_channel( XERCES_CPP_NAMESPACE::DOMNode * _dataset, int ieta, int iphi, int depth, std::string subdetector );

  //
  //_____ DATA_SET getter methods
  //
  XERCES_CPP_NAMESPACE::DOMElement * get_data_element( XERCES_CPP_NAMESPACE::DOMNode * _dataset );
  XERCES_CPP_NAMESPACE::DOMElement * get_channel_element( XERCES_CPP_NAMESPACE::DOMNode * _dataset );

  //
  //_____ tester methods ________________________________________________
  //


 protected:  
  XERCES_CPP_NAMESPACE::MemBufInputSource * _root; // a container for the XML template;
  XERCES_CPP_NAMESPACE::MemBufInputSource * _dataset; // a container for the XML template;
  
  //
  //_____ HEADER
  std::string extension_table_name;
  std::string type_name;
  int run_number;
  std::string channel_map;
  //
  //_____ ELEMENTS
  int data_set_id;
  int iov_id;
  int iov_begin;
  int iov_end;
  int tag_id;
  std::string tag_mode;
  std::string tag_name;
  std::string detector_name;
  std::string comment;
  //
  //_____MAPS
  int tag_idref;
  int iov_idref;
  int data_set_idref;
  //
  //_____ DATA_SET
  std::string username;
  std::string dataset_comment;

  HcalAssistant hcal_ass;
  int dataset_count;
  time_t global_timestamp;
};


#endif
