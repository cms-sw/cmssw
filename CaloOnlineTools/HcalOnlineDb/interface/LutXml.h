#ifndef CaloOnlineTools_HcalOnlineDb_LutXml_h
#define CaloOnlineTools_HcalOnlineDb_LutXml_h
// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     LutXml
// 
/**\class LutXml LutXml.h CaloOnlineTools/HcalOnlineDb/interface/LutXml.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Mar 18 14:30:33 CDT 2008
// $Id: LutXml.h,v 1.1 2008/02/12 17:02:00 kukartse Exp $
//

#include <vector>
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"




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
  virtual ~LutXml();
  
  void init( void );
  void addLut( Config & _config );
  std::string & getCurrentBrick( void );

 protected:  

  DOMElement * addParameter( string _name, string _type, string _value );
  DOMElement * addParameter( string _name, string _type, int _value );
  DOMElement * addData( string _elements, string _encoding, std::vector<unsigned int> _lut );

  DOMElement * brickElem;

};


#endif
