// S. Won 27 Apr 2009
//
#include <vector>
#include <string>

#include "CalibFormats/HcalObjects/interface/HcalText2DetIdConverter.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CalibCalorimetry/HcalStandardModules/interface/HcalZSXML.h"

namespace {
  void dumpHeader (std::ostream& fOutput, unsigned fRun, const std::string& fTableName, const std::string& fTypeName) {
    fOutput << "<?xml version='1.0' encoding='UTF-8' standalone='no'?>" << std::endl;
//    fOutput << "<!DOCTYPE root []>" << std::endl;
    fOutput << "<ROOT>" << std::endl;
    fOutput << "  <HEADER>" << std::endl;
    fOutput << "    <TYPE>" << std::endl;
    fOutput << "      <EXTENSION_TABLE_NAME>HCAL_ZERO_SUPPRESSION_TYPE01</EXTENSION_TABLE_NAME>" << std::endl;
    fOutput << "      <NAME>HCAL zero suppression [type 1]</NAME>" << std::endl;
    fOutput << "    </TYPE>" << std::endl;
    fOutput << "    <RUN>"<< std::endl;
    fOutput << "      <RUN_TYPE>" << "HcalCalibrations" << "</RUN_TYPE>"<< std::endl;
    fOutput << "      <RUN_NUMBER>" << fRun << "</RUN_NUMBER>"<< std::endl;
    fOutput << "    </RUN>" << std::endl;
    fOutput << "   </HEADER>" << std::endl;
  }

  void dumpFooter (std::ostream& fOutput) {
    fOutput << "</ROOT>" << std::endl;
  }

  void dumpChannelId (std::ostream& fOutput, DetId fChannel) {
    HcalText2DetIdConverter converter (fChannel);
    fOutput << "<CHANNEL> "<< std::endl;
    fOutput << "   <EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME> "<< std::endl;
    fOutput << "   <ETA>" << abs (converter.getField (1)) << "</ETA>"<< std::endl;
    fOutput << "   <PHI>" << converter.getField (2) << "</PHI> "<< std::endl;
    fOutput << "   <DEPTH>" << converter.getField (3) << "</DEPTH> "<< std::endl;
    fOutput << "   <Z>" << (converter.getField (1) > 0 > 0 ? "1" : "-1") << "</Z> "<< std::endl;
    fOutput << "   <DETECTOR_NAME>" << converter.getFlavor () << "</DETECTOR_NAME> "<< std::endl;
    fOutput << "   <HCAL_CHANNEL_ID>" << converter.getId().rawId () << "</HCAL_CHANNEL_ID> "<< std::endl;
    fOutput << "</CHANNEL>"<< std::endl;
    fOutput << std::endl;
  }

  void dumpData (std::ostream& fOutput, int fValue) {
    fOutput << "<DATA> "<< std::endl;
    fOutput << "   <ZERO_SUPPRESSION>" << fValue << "</ZERO_SUPPRESSION> "<< std::endl;
    fOutput << "</DATA> " << std::endl;
  }

  void dumpDataset (std::ostream& fOutput, unsigned fVersion = 0) {
    fOutput << "<DATA_SET>" << std::endl;
    fOutput << "   <VERSION>" << fVersion << "</VERSION>" << std::endl;
  }

  void endDataset (std::ostream& fOutput) {
    fOutput << "</DATA_SET>" << std::endl;
  }

  void dumpMapping (std::ostream& fOutput, unsigned fRun, const std::string& fKind, 
		    unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, 
		    const std::string& fTag, unsigned fVersion, const std::vector<DetId>& fChannels) {
    const std::string IOV_ID = "IOV_ID";
    const std::string TAG_ID = "TAG_ID";
    fOutput << "<ELEMENTS>" << std::endl;
    // set channels affected
/*    int i = fChannels.size ();
    while (--i >= 0) {
      fOutput << "<DATA_SET id=\"" << i << "\">" << std::endl;
      fOutput << "<KIND_OF_CONDITION><NAME>" << fKind << "</NAME></KIND_OF_CONDITION>" << std::endl;
      dumpChannelId (fOutput, fChannels[i]);
      fOutput << "<VERSION>" << fVersion << "</VERSION>" << std::endl;
      fOutput << "</DATA_SET>" << std::endl;
    }*/
    // set IOV
    fOutput << "<DATA_SET id=\"1\"/>" << std::endl;
    fOutput << "   <IOV id=\"" << IOV_ID << "\">" << std::endl;
    fOutput << "      <INTERVAL_OF_VALIDITY_BEGIN>" << fGMTIOVBegin << "</INTERVAL_OF_VALIDITY_BEGIN>"<< std::endl;
    fOutput << "      <INTERVAL_OF_VALIDITY_END>" << fGMTIOVEnd << "</INTERVAL_OF_VALIDITY_END>"<< std::endl;
    fOutput << "   </IOV>" << std::endl;
    // set TAG
    fOutput << "   <TAG id=\"2\" mode=\"auto\">"<< std::endl;
    fOutput << "      <TAG_NAME>" << fTag << "</TAG_NAME>"<< std::endl;
    fOutput << "      <DETECTOR_NAME>HCAL</DETECTOR_NAME>"<< std::endl;
    fOutput << "      <COMMENT_DESCRIPTION>Automatically created by HcalZSXML</COMMENT_DESCRIPTION>" << std::endl;
    fOutput << "   </TAG>" << std::endl;

    fOutput << "</ELEMENTS>" << std::endl;

    // mapping itself
    fOutput << "<MAPS>" << std::endl;
    fOutput << "   <TAG idref=\"2\">" << std::endl;
    fOutput << "      <IOV idref=\"1\">" << std::endl;
//    i = fChannels.size ();
//    while (--i >= 0) {
    fOutput << "         <DATA_SET idref=\"-1\"/>" << std::endl;
//    }
    fOutput << "      </IOV>" << std::endl;
    fOutput << "   </TAG>" << std::endl;
    fOutput << "</MAPS>" << std::endl;
  }
}

bool HcalZSXML::dumpObject (std::ostream& fOutput,
                            unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion,
                            const HcalZSThresholds& fObject) {
  const std::string KIND = "HCAL zero suppression [type 1]";
  const std::string TABLE = "HCAL_ZERO_SUPPRESSION_TYPE01";

  dumpHeader (fOutput, fRun, TABLE, KIND);

  std::vector<DetId> channels = fObject.getAllChannels ();
  dumpMapping (fOutput, fRun, KIND, fGMTIOVBegin, fGMTIOVEnd, fTag, fVersion, channels);
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    DetId chId = *channel;
    int value = -999;
    value = fObject.getValues (chId)->getValue ();
    if (value==-999) {
      std::cerr << "HcalZSXML::dumpObject-> Can not get data for channel " << HcalText2DetIdConverter(chId).toString () << std::endl;
      continue;
    }
    dumpDataset (fOutput, fVersion);
    dumpChannelId (fOutput,chId);
    dumpData (fOutput, value);
    endDataset (fOutput);
  }

  dumpFooter (fOutput);
  return true;
}

