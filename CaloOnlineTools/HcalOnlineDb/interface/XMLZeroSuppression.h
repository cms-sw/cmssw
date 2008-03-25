#ifndef HCALConfigDBTools_XMLTools_XMLZeroSuppression_h
#define HCALConfigDBTools_XMLTools_XMLZeroSuppression_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLZeroSuppression
// 
/**\class XMLZeroSuppression XMLZeroSuppression.h CaloOnlineTools/HcalOnlineDb/interface/XMLZeroSuppression.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 23 14:30:33 CDT 2007
// $Id: XMLZeroSuppression.h,v 1.2 2007/11/08 20:24:19 kukartse Exp $
//

// system include files
#include<map>

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"

// forward declarations

class ZeroSuppressionID
{
 public:
  ZeroSuppressionID(){};
  ~ZeroSuppressionID(){};
  int crate;
  int slot;
  int topbottom;
  int fiber;

  bool operator<(const ZeroSuppressionID & _id) const
    {
      if ( crate < _id . crate ) return true;
      else if ( crate == _id . crate && slot < _id . slot ) return true;
      else if ( crate == _id . crate && slot == _id . slot && topbottom < _id . topbottom ) return true;
      else if ( crate == _id . crate && slot == _id . slot && topbottom == _id . topbottom && fiber < _id . fiber ) return true;
      else return false;
    }

 private:

};

class XMLZeroSuppression : public XMLDOMBlock
{
  
 public:
  
  typedef struct _loaderBaseConfig : public XMLProcessor::loaderBaseConfig
  {

  } loaderBaseConfig;
  
  typedef struct _ZeroSuppressionConfig
  {
    _ZeroSuppressionConfig();
    string CFGBrickSet;
    int crate;
    int slot;
    int topbottom;
    int fiber;
    int generalizedindex;
    string creationtag;
    string creationstamp;
    string pattern_spec_name;
  } ZeroSuppressionConfig;
  
  XMLZeroSuppression();
  XMLZeroSuppression( string templateBase );
  virtual ~XMLZeroSuppression();
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  int addPattern( ZeroSuppressionConfig * config,
  	      string templateFileName = "HCAL_HTR_DATA_PATTERNS.brickset.template" );
  int createByCrate( void );

 private:
  XMLZeroSuppression(const XMLZeroSuppression&); // stop default
  
  const XMLZeroSuppression& operator=(const XMLZeroSuppression&); // stop default
  
  int fillPatternInfo( const ZeroSuppressionConfig & );

  // ---------- member data --------------------------------
  map< ZeroSuppressionID, int > configByCrate;

  string data_elements;
};


#endif
