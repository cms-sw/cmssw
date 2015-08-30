#ifndef CaloOnlineTools_HcalOnlineDb_LutXml_h
#define CaloOnlineTools_HcalOnlineDb_LutXml_h
// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     LutXml
// 
/**\class LutXml LutXml.h CalibCalorimetry/HcalTPGAlgos/interface/LutXml.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Mar 18 14:30:33 CDT 2008
//

#include <vector>
#include <map>
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include <stdint.h>

class LutXml : public XMLDOMBlock
{
  
 public:
    
  typedef struct _Config
  {
    _Config();
    int ieta, iphi, depth, crate, slot, topbottom, fiber, fiberchan, lut_type;
    std::string creationtag;
    std::string creationstamp;
    std::string formatrevision;
    std::string targetfirmware;
    int generalizedindex;
    std::vector<unsigned int> lut;
  } Config;
  
  LutXml();
  LutXml( XERCES_CPP_NAMESPACE::InputSource & _source );
  LutXml( std::string filename );
  virtual ~LutXml();
  
  void init( void );
  void addLut( Config & _config, XMLDOMBlock * checksums_xml = 0 );
  std::string & getCurrentBrick( void );
  
  std::vector<unsigned int> * getLutFast( uint32_t det_id );
  //
  //_____ following removed as a xalan-c component_____________________
  //
  //std::vector<unsigned int> getLut( int lut_type, int crate, int slot, int topbottom, int fiber, int fiber_channel );

  HcalSubdetector subdet_from_crate(int crate, int eta, int depth);
  int a_to_i(char * inbuf);
  int create_lut_map( void );

  static std::string get_checksum( std::vector<unsigned int> & lut );

  //
  //_____ following removed as a xalan-c component_____________________
  //
  //int test_xpath( std::string filename );
  int test_access( std::string filename );

  //LutXml & operator+=( const LutXml & other);

  //Iterators and find
  typedef std::map<uint32_t,std::vector<unsigned int> >::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator find(uint32_t) const;

 protected:  

  XMLCh * root;
  XMLCh * brick;
  XERCES_CPP_NAMESPACE::DOMElement * addParameter( std::string _name, std::string _type, std::string _value );
  XERCES_CPP_NAMESPACE::DOMElement * addParameter( std::string _name, std::string _type, int _value );
  XERCES_CPP_NAMESPACE::DOMElement * addData( std::string _elements, std::string _encoding, const std::vector<unsigned int>& _lut );

  XERCES_CPP_NAMESPACE::DOMElement * add_checksum(XERCES_CPP_NAMESPACE::DOMDocument * parent, Config & config );

  XERCES_CPP_NAMESPACE::DOMElement * brickElem;

  //std::map<uint32_t,std::vector<unsigned int> > * lut_map;
  std::map<uint32_t,std::vector<unsigned int> > lut_map;

};


#endif
