#ifndef DATAFORMATS_HCALDETID_HCALSUBDETECTOR_H
#define DATAFORMATS_HCALDETID_HCALSUBDETECTOR_H

enum HcalSubdetector { HcalEmpty=0, HcalBarrel=1, HcalEndcap=2, HcalOuter=3, HcalForward=4, HcalTriggerTower=5, HcalOther=7 };

enum HcalOtherSubdetector { HcalOtherEmpty=0, HcalCalibration=2 };

#endif

#ifndef CaloOnlineTools_HcalOnlineDb_HcalAssistant_h
#define CaloOnlineTools_HcalOnlineDb_HcalAssistant_h
// -*- C++ -*-
//
// Package:     HcalOnlineDb
// Class  :     HcalAssistant
// 
/**\class HcalAssistant HcalAssistant.h CaloOnlineTools/HcalOnlineDb/interface/HcalAssistant.h

 Description: Various helper functions

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Thu Jul 16 11:39:31 CEST 2009
// $Id: HcalAssistant.h,v 1.1 2009/07/16 16:29:34 kukartse Exp $
//

#include <iostream>
#include <string>
#include <vector>
#include <map>


class HcalAssistant
{
  
 public:
  friend class HcalChannelQualityXml;  
  
  HcalAssistant();
  virtual ~HcalAssistant();
  
  int addQuotes();
  std::string getRandomQuote(void);
  
  std::string getUserName(void);
  
  HcalSubdetector getSubdetector(std::string _det);
  std::string getSubdetectorString(HcalSubdetector _det);
  
  int getListOfChannelsFromDb();
  int getSubdetector(int _rawid);
  int getIeta(int _rawid);
  int getIphi(int _rawid);
  int getDepth(int _rawid);
  int getRawId(HcalSubdetector _det, int _ieta, int _iphi, int _depth);
  
 private:
  std::vector<std::string> quotes;
  std::map<int,int> geom_to_rawid; // geom hash is the hey
  std::map<int,int> rawid_to_geom; // rawId is the key
  bool listIsRead;  // were channels read from OMDS?
  
  //
  //_____ encode HCAL geometry channel in a single integer hash
  //      not relying on HcalDetId
  int getGeomId(HcalSubdetector _det, int _ieta, int _iphi, int _depth);
  int getHcalIeta(int _geomId);
  int getHcalIphi(int _geomId);
  int getHcalDepth(int _geomId);
  HcalSubdetector getHcalSubdetector(int _geomId);
  int getRawId(int _geomId);
  int getGeomId(int _rawid);
};


#endif
