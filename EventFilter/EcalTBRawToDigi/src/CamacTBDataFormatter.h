#ifndef CamacTBDataFormatter_H
#define CamacTBDataFormatter_H
/** \class CamacTBDataFormatter
 *
 *  $Id: CamacTBDataFormatter.h,v 1.7 2010/10/12 13:11:35 eulisse Exp $
 *
 *  \author G. Franzoni
 */


#include <vector> 
#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopePlaneRawHits.h>
#include <TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRawInfo.h>
#include <TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h>
#include <TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>

#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCSample.h"




class CamacTBDataFormatter   {

 public:

  CamacTBDataFormatter();
  virtual ~CamacTBDataFormatter(){LogDebug("EcalTBRawToDigi") << "@SUB=CamacTBDataFormatter" << "\n"; };

  
  void  interpretRawData( const FEDRawData & data, EcalTBEventHeader& tbEventHeader, 
			  EcalTBHodoscopeRawInfo& hodoRaw, EcalTBTDCRawInfo& tdcRawInfo );
  
  // for tests based on standalone file
  /*   void  interpretRawData(unsigned long * buffer, unsigned long bufferSize, */
  /* 			 EcalTBEventHeader& tbEventHeader,  */
  /* 			 EcalTBHodoscopeRawInfo & hodo, */
  /* 			 EcalTBTDCRawInfo & tdc); */
  


 private:

  bool checkStatus(unsigned long word, int wordNumber);

  int nWordsPerEvent;    // Number of fibers per hodoscope plane   
  
  static const int nHodoFibers        = 64;    // Number of fibers per hodoscope plane   
  static const int nHodoscopes      = 2;      // Number of different mappings between fiber and electronics     
  static const int nHodoPlanes       = 4;      // Number of hodoscopes along the beam
  static const int hodoRawLen       = 4;      // The raw data is stored as 4 integers for each hodo plane

  int nHodoHits[nHodoPlanes];
  int hodoHits[nHodoPlanes][nHodoFibers];
  int hodoAll[nHodoPlanes*nHodoFibers];
  bool statusWords[148+4];

  std::vector<int> scalers_;
};
#endif
