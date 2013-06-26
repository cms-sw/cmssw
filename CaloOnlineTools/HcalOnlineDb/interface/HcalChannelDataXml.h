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
// $Id: HcalChannelDataXml.h,v 1.6 2010/08/06 20:24:09 wmtan Exp $
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
  DOMNode * set_header_table_name(std::string name);
  DOMNode * set_header_type(std::string type);
  DOMNode * set_header_run_number(int run);
  DOMNode * set_header_channel_map(std::string name);
  DOMNode * set_elements_dataset_id(int id);
  DOMNode * set_elements_iov_id(int id);
  DOMNode * set_elements_iov_begin(int value);
  DOMNode * set_elements_iov_end(int value);
  DOMNode * set_elements_tag_id(int value);
  DOMNode * set_elements_tag_mode(std::string value);
  DOMNode * set_elements_tag_name(std::string value);
  DOMNode * set_elements_detector_name(std::string value);
  DOMNode * set_elements_comment(std::string value);
  DOMNode * set_maps_tag_idref(int value);
  DOMNode * set_maps_iov_idref(int value);
  DOMNode * set_maps_dataset_idref(int value);

  //
  //_____ add data
  //
  DOMNode * add_dataset( void );
  DOMNode * add_hcal_channel( DOMNode * _dataset, int ieta, int iphi, int depth, std::string subdetector );

  //
  //_____ DATA_SET getter methods
  //
  DOMElement * get_data_element( DOMNode * _dataset );
  DOMElement * get_channel_element( DOMNode * _dataset );

  //
  //_____ tester methods ________________________________________________
  //


 protected:  
  MemBufInputSource * _root; // a container for the XML template;
  MemBufInputSource * _dataset; // a container for the XML template;
  
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
