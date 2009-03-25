// -*- C++ -*-
//
// Original Author:  Gena Kukartsev Mar 11, 2009
// Adapted from HcalDbOmds.h
// $Id: HcalDbOmds.h,v 1.3 2009/03/16 01:43:24 kukartse Exp $
//
//
#ifndef HcalDbOmds_h
#define HcalDbOmds_h

#include <iostream>

#include "xgi/Utils.h"
#include "toolbox/string.h"
#include "occi.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"

using namespace oracle::occi;


/**
   \class HcalDbOmds
   \brief IO for OMDS instances of Hcal Calibrations
   \author Gena Kukartsev March 11, 2009
   $Id: HcalDbOmds.h,v 1.3 2009/03/16 01:43:24 kukartse Exp $
   
Text file formats for different data types is as following:
- # in first column comments the line
- HcalPedestals, HcalGains, HcalGainWidths have identical formats:
  eta(int)  phi(int) depth(int) det(HB,HE,HF) cap1_value(float) cap2_value(float) cap3_value(float) cap4_value(float)  HcalDetId(int,optional)
- HcalPedestalWidths:
  eta(int)  phi(int) depth(int) det(HB,HE,HF) sigma_1_1(float) sigma_2_1 sigma_2_2 sigma_3_1 sigma_3_2 sigma_3_3 sigma_4_1 sigma_4_2 sigma_4_3 sigma_4_4
- HcalQIEShape:
  33 x floats - low edges for first 33 channels of ideal QIE
- HcalQIEData:
  eta phi depth det 4x offsets_cap1 4x offsets_cap2 4x offsets_cap3 4x offsets_cap4 4x slopes_cap1 4x slopes_cap2 4x slopes_cap3 4x slopes_cap4
- HcalChannelQuality:
  eta phi depth det status(GOOD/BAD/HOT/DEAD)
- HcalElectronicsMap:
  line#  crate HTR_slot top_bottom(t/b) dcc# dcc_spigot fiber fiberchan subdet(HB/HE/HF/HO/HT) eta phi depth
  line#  crate HTR_slot top_bottom(t/b) dcc# dcc_spigot fiber fiberchan "CBOX" 
                                 sector(HBM/HBP/HEM/HEP/HO0/HO1P/HO1M/HO2P/HO2M/HFP/HFM) rbx#(wage) channel
  calibration channel type association see HcalCalibDetId.h
  if electronics channel is known to be unconnected, either "subdet" or "eta" should be NA
*/
namespace HcalDbOmds {
  bool getObject (oracle::occi::Connection * connection, 
		  const std::string & fTag, 
		  const std::string & fVersion,
		  const int fSubversion,
		  const std::string & fQuery,
		  HcalPedestals* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalPedestals& fObject);
  bool getObject (oracle::occi::Connection * connection, 
		  const std::string & fTag, 
		  const std::string & fVersion,
		  const int fSubversion,
		  const std::string & fQuery,
		  HcalPedestalWidths* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalPedestalWidths& fObject);
  bool getObject (oracle::occi::Connection * connection, 
		  const std::string & fTag, 
		  const std::string & fVersion,
		  const int fSubversion,
		  const std::string & fQuery,
		  HcalGains* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalGains& fObject);
  bool getObject (oracle::occi::Connection * connection, 
		  const std::string & fTag, 
		  const std::string & fVersion,
		  const int fSubversion,
		  const std::string & fQuery,
		  HcalGainWidths* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalGainWidths& fObject);
  bool getObject (oracle::occi::Connection * connection, 
		  const std::string & fTag, 
		  const std::string & fVersion,
		  const int fSubversion,
		  const std::string & fQuery,
		  HcalQIEData* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalQIEData& fObject);
  bool getObject (oracle::occi::Connection * connection, 
		  const std::string & fTag, 
		  const std::string & fVersion,
		  const int fSubversion,
		  const std::string & fQuery,
		  HcalCalibrationQIEData* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalCalibrationQIEData& fObject);
  bool getObject (oracle::occi::Connection * connection, 
		  const std::string & fTag, 
		  const std::string & fVersion,
		  const int fSubversion,
		  const std::string & fQuery,
		  HcalElectronicsMap* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalElectronicsMap& fObject);
  bool getObject (oracle::occi::Connection * connection, 
		  const std::string & fTag, 
		  const std::string & fVersion,
		  const int fSubversion,
		  const std::string & fQuery,
		  HcalChannelQuality* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalChannelQuality& fObject);
  bool getObject (oracle::occi::Connection * connection, 
		  const std::string & fTag, 
		  const std::string & fVersion,
		  const int fSubversion,
		  const std::string & fQuery,
		  HcalRespCorrs* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalRespCorrs& fObject);
  bool getObject (oracle::occi::Connection * connection, 
		  const std::string & fTag, 
		  const std::string & fVersion,
		  const int fSubversion,
		  const std::string & fQuery,
		  HcalZSThresholds* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalZSThresholds& fObject);
  bool getObject (oracle::occi::Connection * connection, 
		  const std::string & fTag, 
		  const std::string & fVersion,
		  const int fSubversion,
		  const std::string & fQuery,
		  HcalL1TriggerObjects* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalL1TriggerObjects& fObject);

  HcalSubdetector get_subdetector( std::string _det );
} 
#endif
