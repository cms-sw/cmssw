
//
// F.Ratnikov (UMd), Oct 28, 2005
// $Id: HcalDbXml.cc,v 1.1 2005/11/02 21:31:24 fedor Exp $
//
#include <vector>
#include <string>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbXml.h"

namespace {
  void dumpHeader (std::ostream& fOutput, unsigned fRun, const std::string& fTag, const std::string& fTableName, const std::string& fTypeName) {
    char buffer [1024];
    fOutput << "<?xml version='1.0' encoding='UTF-8'?>" << std::endl;
    fOutput << "<!DOCTYPE root []>" << std::endl;
    fOutput << "<ROOT>" << std::endl;
    fOutput << "  <HEADER>" << std::endl;
    fOutput << "    <TYPE>" << std::endl;
    fOutput << "      <EXTENSION_TABLE_NAME>" << fTableName << "</EXTENSION_TABLE_NAME>" << std::endl;
    fOutput << "      <NAME>" << fTypeName << "</NAME>" << std::endl;
    fOutput << "    </TYPE>" << std::endl;
    fOutput << "    <RUN>" << std::endl;
    sprintf (buffer, "TAG=%s RUN=%d", fTag.c_str (), fRun);
    fOutput << "      <RUN_NAME>" << buffer << "</RUN_NAME>" << std::endl;
    fOutput << "      <RUN_BEGIN_TIMESTAMP>2003-10-31 00:00:01.0</RUN_BEGIN_TIMESTAMP>" << std::endl; //Taffy's birthday
    fOutput << "      <COMMENT_DESCRIPTION>Generated automatically by HcalDbXml::dumpObject</COMMENT_DESCRIPTION>" << std::endl;
    fOutput << "    </RUN>" << std::endl;
    fOutput << "  </HEADER>" << std::endl;
  }

  void dumpFooter (std::ostream& fOutput) {
    fOutput << "</ROOT>" << std::endl;
  }

  void dumpChannelId (std::ostream& fOutput, HcalDetId fChannel) {
    fOutput << "      ";
    fOutput << "<CHANNEL> ";
    fOutput << "<EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME> ";
    fOutput << "<ETA>" << fChannel.ietaAbs() << "</ETA>";
    fOutput << "<PHI>" << fChannel.iphi() << "</PHI> ";
    fOutput << "<DEPTH>" << fChannel.depth() << "</DEPTH> ";
    fOutput << "<Z>" << (fChannel.zside() > 0 ? '+' : '-') << "</Z> ";
    fOutput << "<DETECTOR_NAME>" << (fChannel.subdet() == HcalBarrel ? "HB" : fChannel.subdet() == HcalEndcap ? "HE" : "HF") << "</DETECTOR_NAME> ";
    fOutput << "<HCAL_CHANNEL_ID>" << fChannel << "</HCAL_CHANNEL_ID> ";
    fOutput << "</CHANNEL>";
    fOutput << std::endl;
  }

  void dumpData (std::ostream& fOutput, const float* fValues, const float* fErrors) {
    fOutput << "      ";
    fOutput << "<DATA> ";
    fOutput << "<CAPACITOR_0_VALUE>" << fValues [0] << "</CAPACITOR_0_VALUE> ";
    fOutput << "<CAPACITOR_1_VALUE>" << fValues [1] << "</CAPACITOR_1_VALUE> ";
    fOutput << "<CAPACITOR_2_VALUE>" << fValues [2] << "</CAPACITOR_2_VALUE> ";	
    fOutput << "<CAPACITOR_3_VALUE>" << fValues [3] << "</CAPACITOR_3_VALUE> ";
    fOutput << "<CAPACITOR_0_ERROR>" << fErrors [0] << "</CAPACITOR_0_ERROR> ";	
    fOutput << "<CAPACITOR_1_ERROR>" << fErrors [1] << "</CAPACITOR_1_ERROR> ";	
    fOutput << "<CAPACITOR_2_ERROR>" << fErrors [2] << "</CAPACITOR_2_ERROR> ";	
    fOutput << "<CAPACITOR_3_ERROR>" << fErrors [3] << "</CAPACITOR_3_ERROR> ";	
    fOutput << "</DATA>";
    fOutput << std::endl;
  }

  void dumpDataset (std::ostream& fOutput, const std::string& fFileName, const std::string& fDescription) {
    fOutput << "    <DATA_SET>" << std::endl;
    fOutput << "      <COMMENT_DESCRIPTION>" << fDescription << "</COMMENT_DESCRIPTION>" << std::endl;
    fOutput << "      <DATA_FILE_NAME>" << fFileName << "</DATA_FILE_NAME>" << std::endl;
  }

  void endDataset (std::ostream& fOutput) {
    fOutput << "    </DATA_SET>" << std::endl;
  }
}


bool HcalDbXml::dumpObject (std::ostream& fOutput, unsigned fRun, const std::string& fTag, const HcalPedestals& fObject, const HcalPedestalWidths& fError) {
  float dummyErrors [4] = {0.0001, 0.0001, 0.0001, 0.0001};

  dumpHeader (fOutput, fRun, fTag, "WSLED_PEDESTAL_CLBRTN", "HCAL Pedestals");

  std::vector<HcalDetId> channels = fObject.getAllChannels ();
  for (std::vector<HcalDetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    HcalDetId chId = *channel;
    const float* values = fObject.getValues (chId)->getValues ();
    const float* errors = fError.getValues (chId)->getValues ();
    if (!values) {
      std::cerr << "HcalDbXml::dumpObject-> Can not get data for channel " << chId << std::endl;
      continue;
    }
    if (!errors) {
      std::cerr << "HcalDbXml::dumpObject-> Can not get errors for channel " << chId <<  ". Use defaults" << std::endl;
      errors = dummyErrors;
    }
    dumpDataset (fOutput, "na", "na");
    dumpChannelId (fOutput,chId.rawId ()); 
    dumpData (fOutput, values, errors);
    endDataset (fOutput);
  }
  dumpFooter (fOutput);
  return true;
}

bool HcalDbXml::dumpObject (std::ostream& fOutput, unsigned fRun, const std::string& fTag, const HcalGains& fObject, const HcalGainWidths& fError) {
  float dummyErrors [4] = {0.0001, 0.0001, 0.0001, 0.0001};

  dumpHeader (fOutput, fRun, fTag, "WSLED_GAIN_CLBRTN", "HCAL Gains");

  std::vector<HcalDetId> channels = fObject.getAllChannels ();
  for (std::vector<HcalDetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    HcalDetId chId = *channel;
    const float* values = fObject.getValues (chId)->getValues ();
    const float* errors = fError.getValues (chId)->getValues ();
    if (!values) {
      std::cerr << "HcalDbXml::dumpObject-> Can not get data for channel " << chId << std::endl;
      continue;
    }
    if (!errors) {
      std::cerr << "HcalDbXml::dumpObject-> Can not get errors for channel " << chId <<  ". Use defaults" << std::endl;
      errors = dummyErrors;
    }
    dumpDataset (fOutput, "na", "na");
    dumpChannelId (fOutput,chId); 
    dumpData (fOutput, values, errors);
    endDataset (fOutput);
  }
  dumpFooter (fOutput);
  return true;
}
