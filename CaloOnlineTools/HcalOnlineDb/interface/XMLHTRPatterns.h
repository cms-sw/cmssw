#ifndef HCALConfigDBTools_XMLTools_XMLHTRPatterns_h
#define HCALConfigDBTools_XMLTools_XMLHTRPatterns_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLHTRPatterns
// 
/**\class XMLHTRPatterns XMLHTRPatterns.h CaloOnlineTools/HcalOnlineDb/interface/XMLHTRPatterns.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 23 14:30:33 CDT 2007
// $Id: XMLHTRPatterns.h,v 1.3 2007/12/06 02:26:09 kukartse Exp $
//

// system include files
#include<map>

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"

// forward declarations

class HTRPatternID
{
 public:
  HTRPatternID(){};
  ~HTRPatternID(){};
  int crate;
  int slot;
  int topbottom;
  int fiber;

  bool operator<(const HTRPatternID & _id) const
    {
      if ( crate < _id . crate ) return true;
      else if ( crate == _id . crate && slot < _id . slot ) return true;
      else if ( crate == _id . crate && slot == _id . slot && topbottom < _id . topbottom ) return true;
      else if ( crate == _id . crate && slot == _id . slot && topbottom == _id . topbottom && fiber < _id . fiber ) return true;
      else return false;
    }

 private:

};

class XMLHTRPatterns : public XMLDOMBlock
{
  
 public:
  
  typedef struct _loaderBaseConfig : public XMLProcessor::loaderBaseConfig
  {

  } loaderBaseConfig;
  
  typedef struct _HTRPatternConfig
  {
    _HTRPatternConfig();
    string CFGBrickSet;
    int crate;
    int slot;
    int topbottom;
    int fiber;
    int generalizedindex;
    string creationtag;
    string creationstamp;
    string pattern_spec_name;
  } HTRPatternConfig;
  
  XMLHTRPatterns();
  XMLHTRPatterns( string templateBase );
  virtual ~XMLHTRPatterns();
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  int addPattern( HTRPatternConfig * config,
  	      string templateFileName = "HCAL_HTR_DATA_PATTERNS.brickset.template" );
  int createByCrate( void );

 private:
  XMLHTRPatterns(const XMLHTRPatterns&); // stop default
  
  const XMLHTRPatterns& operator=(const XMLHTRPatterns&); // stop default
  
  int fillPatternInfo( const HTRPatternConfig & );

  // ---------- member data --------------------------------
  map< HTRPatternID, int > configByCrate;

  string data_elements;
};


#endif
