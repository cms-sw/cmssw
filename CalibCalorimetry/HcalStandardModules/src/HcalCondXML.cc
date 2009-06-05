// S. Won, Northwestern University, 5 June 2009

#include <vector>
#include <string>

#include "CalibFormats/HcalObjects/interface/HcalText2DetIdConverter.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CalibCalorimetry/HcalStandardModules/interface/HcalCondXML.h"
namespace {
  void dumpHeader (std::ostream& fOutput, unsigned fRun, const std::string& fTableName, const std::string& fTypeName) {
    fOutput << "<?xml version='1.0' encoding='UTF-8' standalone='no'?>" << std::endl;
//    fOutput << "<!DOCTYPE root []>" << std::endl;
    fOutput << "<ROOT>" << std::endl;
    fOutput << "  <HEADER>" << std::endl;
    fOutput << "    <TYPE>" << std::endl;
    fOutput << "      <EXTENSION_TABLE_NAME>" << fTableName << "</EXTENSION_TABLE_NAME>" << std::endl;
    fOutput << "      <NAME>" << fTypeName << "</NAME>" << std::endl;
    fOutput << "    </TYPE>" << std::endl;
    fOutput << "    <RUN>"<< std::endl;
    fOutput << "      <RUN_TYPE>" << "HcalCalibrations" << "</RUN_TYPE>"<< std::endl;
    fOutput << "      <RUN_NUMBER>" << fRun << "</RUN_NUMBER>"<< std::endl;
    fOutput << "    </RUN>" << std::endl;
    fOutput << "    <HINTS channelmap='HCAL_CHANNELS'/>" << std::endl;
    fOutput << "   </HEADER>" << std::endl;
  }

  void dumpMapping (std::ostream& fOutput, unsigned fRun, const std::string& fKind,
                    unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd,
                    const std::string& fTag, unsigned fVersion, const std::vector<DetId>& fChannels) {
    const std::string IOV_ID = "IOV_ID";
    const std::string TAG_ID = "TAG_ID";
    fOutput << "<ELEMENTS>" << std::endl;
    // set IOV
    fOutput << "<DATA_SET id=\"-1\"/>" << std::endl;
    fOutput << "   <IOV id=\"1\">" << std::endl;
    fOutput << "      <INTERVAL_OF_VALIDITY_BEGIN>" << fGMTIOVBegin << "</INTERVAL_OF_VALIDITY_BEGIN>"<< std::endl;
    fOutput << "      <INTERVAL_OF_VALIDITY_END>" << fGMTIOVEnd << "</INTERVAL_OF_VALIDITY_END>"<< std::endl;
    fOutput << "   </IOV>" << std::endl;
    // set TAG
    fOutput << "   <TAG id=\"2\" mode=\"auto\">"<< std::endl;
    fOutput << "      <TAG_NAME>" << fTag << "</TAG_NAME>"<< std::endl;
    fOutput << "      <DETECTOR_NAME>HCAL</DETECTOR_NAME>"<< std::endl;
    fOutput << "      <COMMENT_DESCRIPTION>Automatically created by HcalCondXML</COMMENT_DESCRIPTION>" << std::endl;
    fOutput << "   </TAG>" << std::endl;

    fOutput << "</ELEMENTS>" << std::endl;

    fOutput << "<MAPS>" << std::endl;
    fOutput << "   <TAG idref=\"2\">" << std::endl;
    fOutput << "      <IOV idref=\"1\">" << std::endl;
    fOutput << "         <DATA_SET idref=\"-1\"/>" << std::endl;
    fOutput << "      </IOV>" << std::endl;
    fOutput << "   </TAG>" << std::endl;
    fOutput << "</MAPS>" << std::endl;
  }

  void dumpFooter (std::ostream& fOutput) {
    fOutput << "</ROOT>" << std::endl;
  }

  void dumpDataset (std::ostream& fOutput, unsigned fVersion = 0) {
    fOutput << "<DATA_SET>" << std::endl;
    fOutput << "   <VERSION>" << fVersion << "</VERSION>" << std::endl;
  }

  void endDataset (std::ostream& fOutput) {
    fOutput << "</DATA_SET>" << std::endl;
  }

  void dumpChannelIdOld (std::ostream& fOutput, DetId fChannel) {
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

  void dumpChannelId (std::ostream& fOutput, DetId fChannel) {
    HcalText2DetIdConverter converter (fChannel);
    fOutput << "<CHANNEL> "<< std::endl;
    fOutput << "   <EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME> "<< std::endl;
    fOutput << "   <SUBDET>" << converter.getFlavor () << "</SUBDET> "<< std::endl;
    fOutput << "   <IETA>" << converter.getField (1) << "</IETA>"<< std::endl;
    fOutput << "   <IPHI>" << converter.getField (2) << "</IPHI> "<< std::endl;
    fOutput << "   <DEPTH>" << converter.getField (3) << "</DEPTH> "<< std::endl;
//    fOutput << "   <HCAL_CHANNEL_ID>" << converter.getId().rawId () << "</HCAL_CHANNEL_ID> "<< std::endl;
    fOutput << "</CHANNEL>"<< std::endl;
    fOutput << std::endl;
  }

  void dumpZSData (std::ostream& fOutput, HcalZSThreshold fValue) {
    fOutput << "<DATA> "<< std::endl;
    fOutput << "   <ZERO_SUPPRESSION>" << fValue.getValue() << "</ZERO_SUPPRESSION> "<< std::endl;
    fOutput << "</DATA> " << std::endl;
  }

  void dumpPedData (std::ostream& fOutput, HcalPedestal fValue, HcalPedestalWidth fValue2, int isADC) {
    fOutput << "<DATA> "<< std::endl;
    fOutput << "   <CAP0>" << fValue.getValue(0) << "</CAP0> "<< std::endl;
    fOutput << "   <CAP1>" << fValue.getValue(1) << "</CAP1> "<< std::endl;
    fOutput << "   <CAP2>" << fValue.getValue(2) << "</CAP2> "<< std::endl;
    fOutput << "   <CAP3>" << fValue.getValue(3) << "</CAP3> "<< std::endl;
    fOutput << "   <SIGMA_00>" << fValue2.getWidth(0) << "</SIGMA_00> "<< std::endl;
    fOutput << "   <SIGMA_11>" << fValue2.getWidth(1) << "</SIGMA_11> "<< std::endl;
    fOutput << "   <SIGMA_22>" << fValue2.getWidth(2) << "</SIGMA_22> "<< std::endl;
    fOutput << "   <SIGMA_33>" << fValue2.getWidth(3) << "</SIGMA_33> "<< std::endl;
    fOutput << "   <SIGMA_01>" << fValue2.getSigma(0,1) << "</SIGMA_01> "<< std::endl;
    fOutput << "   <SIGMA_02>" << fValue2.getSigma(0,2) << "</SIGMA_02> "<< std::endl;
    fOutput << "   <SIGMA_03>" << fValue2.getSigma(0,3) << "</SIGMA_03> "<< std::endl;
    fOutput << "   <SIGMA_12>" << fValue2.getSigma(1,2) << "</SIGMA_12> "<< std::endl;
    fOutput << "   <SIGMA_13>" << fValue2.getSigma(1,3) << "</SIGMA_13> "<< std::endl;
    fOutput << "   <SIGMA_23>" << fValue2.getSigma(2,3) << "</SIGMA_23> "<< std::endl;
    fOutput << "   <IS_ADC_COUNTS>" << isADC << "</IS_ADC_COUNTS>" << std::endl;
    fOutput << "</DATA> " << std::endl;
  }

  void dumpGainData (std::ostream& fOutput, HcalGain fValue) {
    fOutput << "<DATA> "<< std::endl;
    fOutput << "   <CAP0>" << fValue.getValue(0) << "</CAP0> "<< std::endl;
    fOutput << "   <CAP1>" << fValue.getValue(1) << "</CAP1> "<< std::endl;
    fOutput << "   <CAP2>" << fValue.getValue(2) << "</CAP2> "<< std::endl;
    fOutput << "   <CAP3>" << fValue.getValue(3) << "</CAP3> "<< std::endl;
    fOutput << "</DATA> " << std::endl;
  }

  void dumpGainWidthData (std::ostream& fOutput, HcalGain fValue) {
    fOutput << "<DATA> "<< std::endl;
    fOutput << "   <CAP0>" << fValue.getValue(0) << "</CAP0> "<< std::endl;
    fOutput << "   <CAP1>" << fValue.getValue(1) << "</CAP1> "<< std::endl;
    fOutput << "   <CAP2>" << fValue.getValue(2) << "</CAP2> "<< std::endl;
    fOutput << "   <CAP3>" << fValue.getValue(3) << "</CAP3> "<< std::endl;
    fOutput << "</DATA> " << std::endl;
  }

  void dumpRespCorrData (std::ostream& fOutput, HcalRespCorr fValue) {
    fOutput << "<DATA> "<< std::endl;
    fOutput << "   <VALUE>" << fValue.getValue() << "</VALUE> "<< std::endl;
    fOutput << "</DATA> " << std::endl;
  }

  void dumpQIEData (std::ostream& fOutput, HcalQIECoder fValue) {
    fOutput << "<DATA> "<< std::endl;
    fOutput << "   <CAP0_RANGE0_SLOPE>" << fValue.slope(0,0) << "</CAP0_RANGE0_SLOPE>" << std::endl; 
    fOutput << "   <CAP0_RANGE1_SLOPE>" << fValue.slope(0,1) << "</CAP0_RANGE1_SLOPE>" << std::endl;
    fOutput << "   <CAP0_RANGE2_SLOPE>" << fValue.slope(0,2) << "</CAP0_RANGE2_SLOPE>" << std::endl;
    fOutput << "   <CAP0_RANGE3_SLOPE>" << fValue.slope(0,3) << "</CAP0_RANGE3_SLOPE>" << std::endl;
    fOutput << "   <CAP1_RANGE0_SLOPE>" << fValue.slope(1,0) << "</CAP1_RANGE0_SLOPE>" << std::endl;
    fOutput << "   <CAP1_RANGE1_SLOPE>" << fValue.slope(1,1) << "</CAP1_RANGE1_SLOPE>" << std::endl;
    fOutput << "   <CAP1_RANGE2_SLOPE>" << fValue.slope(1,2) << "</CAP1_RANGE2_SLOPE>" << std::endl;
    fOutput << "   <CAP1_RANGE3_SLOPE>" << fValue.slope(1,3) << "</CAP1_RANGE3_SLOPE>" << std::endl;
    fOutput << "   <CAP2_RANGE0_SLOPE>" << fValue.slope(2,0) << "</CAP2_RANGE0_SLOPE>" << std::endl;
    fOutput << "   <CAP2_RANGE1_SLOPE>" << fValue.slope(2,1) << "</CAP2_RANGE1_SLOPE>" << std::endl;
    fOutput << "   <CAP2_RANGE2_SLOPE>" << fValue.slope(2,2) << "</CAP2_RANGE2_SLOPE>" << std::endl;
    fOutput << "   <CAP2_RANGE3_SLOPE>" << fValue.slope(2,3) << "</CAP2_RANGE3_SLOPE>" << std::endl;
    fOutput << "   <CAP3_RANGE0_SLOPE>" << fValue.slope(3,0) << "</CAP3_RANGE0_SLOPE>" << std::endl;
    fOutput << "   <CAP3_RANGE1_SLOPE>" << fValue.slope(3,1) << "</CAP3_RANGE1_SLOPE>" << std::endl;
    fOutput << "   <CAP3_RANGE2_SLOPE>" << fValue.slope(3,2) << "</CAP3_RANGE2_SLOPE>" << std::endl;
    fOutput << "   <CAP3_RANGE3_SLOPE>" << fValue.slope(3,3) << "</CAP3_RANGE3_SLOPE>" << std::endl;
    fOutput << "   <CAP0_RANGE0_OFFSET>" << fValue.offset(0,0) << "</CAP0_RANGE0_OFFSET>" << std::endl;
    fOutput << "   <CAP0_RANGE1_OFFSET>" << fValue.offset(0,1) << "</CAP0_RANGE1_OFFSET>" << std::endl;
    fOutput << "   <CAP0_RANGE2_OFFSET>" << fValue.offset(0,2) << "</CAP0_RANGE2_OFFSET>" << std::endl;
    fOutput << "   <CAP0_RANGE3_OFFSET>" << fValue.offset(0,3) << "</CAP0_RANGE3_OFFSET>" << std::endl;
    fOutput << "   <CAP1_RANGE0_OFFSET>" << fValue.offset(1,0) << "</CAP1_RANGE0_OFFSET>" << std::endl;
    fOutput << "   <CAP1_RANGE1_OFFSET>" << fValue.offset(1,1) << "</CAP1_RANGE1_OFFSET>" << std::endl;
    fOutput << "   <CAP1_RANGE2_OFFSET>" << fValue.offset(1,2) << "</CAP1_RANGE2_OFFSET>" << std::endl;
    fOutput << "   <CAP1_RANGE3_OFFSET>" << fValue.offset(1,3) << "</CAP1_RANGE3_OFFSET>" << std::endl;
    fOutput << "   <CAP2_RANGE0_OFFSET>" << fValue.offset(2,0) << "</CAP2_RANGE0_OFFSET>" << std::endl;
    fOutput << "   <CAP2_RANGE1_OFFSET>" << fValue.offset(2,1) << "</CAP2_RANGE1_OFFSET>" << std::endl;
    fOutput << "   <CAP2_RANGE2_OFFSET>" << fValue.offset(2,2) << "</CAP2_RANGE2_OFFSET>" << std::endl;
    fOutput << "   <CAP2_RANGE3_OFFSET>" << fValue.offset(2,3) << "</CAP2_RANGE3_OFFSET>" << std::endl;
    fOutput << "   <CAP3_RANGE0_OFFSET>" << fValue.offset(3,0) << "</CAP3_RANGE0_OFFSET>" << std::endl;
    fOutput << "   <CAP3_RANGE1_OFFSET>" << fValue.offset(3,1) << "</CAP3_RANGE1_OFFSET>" << std::endl;
    fOutput << "   <CAP3_RANGE2_OFFSET>" << fValue.offset(3,2) << "</CAP3_RANGE2_OFFSET>" << std::endl;
    fOutput << "   <CAP3_RANGE3_OFFSET>" << fValue.offset(3,3) << "</CAP3_RANGE3_OFFSET>" << std::endl;
    fOutput << "</DATA> " << std::endl;
  }
}
bool HcalCondXML::dumpObject (std::ostream& fOutput,
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
    if(fObject.exists(chId))
    {
       const HcalZSThreshold * item = fObject.getValues(chId);
       dumpDataset (fOutput, fVersion);
       dumpChannelId (fOutput,chId);
       dumpZSData (fOutput, *item);
       endDataset (fOutput);
    }
  }

  dumpFooter (fOutput);
  return true;
}
bool HcalCondXML::dumpObject (std::ostream& fOutput,
                            unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion,
                            const HcalPedestals& fObject, const HcalPedestalWidths& fObject2) { 
  const std::string KIND = "HCAL Pedestals [V3]";
  const std::string TABLE = "HCAL_PEDESTALS_V3";

  dumpHeader (fOutput, fRun, TABLE, KIND);
  int isADC = 0;
  if(fObject.isADC()) isADC = 1;

  std::vector<DetId> channels = fObject.getAllChannels ();
  dumpMapping (fOutput, fRun, KIND, fGMTIOVBegin, fGMTIOVEnd, fTag, fVersion, channels);
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    DetId chId = *channel;
    if(fObject.exists(chId))
    {
       const HcalPedestal * item = fObject.getValues (chId);
       const HcalPedestalWidth * item2 = fObject2.getValues (chId);

       dumpDataset (fOutput, fVersion);
       dumpChannelId (fOutput,chId);
       dumpPedData (fOutput, *item, *item2, isADC);
       endDataset (fOutput);
    }
  }

  dumpFooter (fOutput);
  return true;
}


