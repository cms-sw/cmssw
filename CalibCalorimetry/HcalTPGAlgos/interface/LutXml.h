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
// $Id: LutXml.h,v 1.2 2009/05/08 23:26:51 elmer Exp $
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
    string creationtag;
    string creationstamp;
    string formatrevision;
    string targetfirmware;
    int generalizedindex;
    std::vector<unsigned int> lut;
  } Config;
  
  LutXml();
  LutXml( InputSource & _source );
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
  typedef std::map<uint32_t,vector<unsigned int> >::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator find(uint32_t) const;

 protected:  

  XMLCh * root;
  XMLCh * brick;
  DOMElement * addParameter( string _name, string _type, string _value );
  DOMElement * addParameter( string _name, string _type, int _value );
  DOMElement * addData( string _elements, string _encoding, std::vector<unsigned int> _lut );

  DOMElement * add_checksum( DOMDocument * parent, Config & config );

  DOMElement * brickElem;

  //std::map<uint32_t,vector<unsigned int> > * lut_map;
  std::map<uint32_t,vector<unsigned int> > lut_map;

};


#endif
