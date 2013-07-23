// F.Ratnikov (UMd), Oct 28, 2005
// Modified by S. Won 6 May 2008
// $Id: CastorDbXml.cc,v 1.2 2009/12/10 10:29:00 elmer Exp $
//
#include <vector>
#include <string>


#include "CalibFormats/CastorObjects/interface/CastorText2DetIdConverter.h"

#include "CondFormats/CastorObjects/interface/AllObjects.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorDbXml.h"

namespace {
  void dumpProlog (std::ostream& fOutput) {
    fOutput << "<?xml version='1.0' encoding='UTF-8'?>" << std::endl;
    fOutput << "<!DOCTYPE root []>" << std::endl;
    fOutput << "<ROOT>" << std::endl;
  }

  void dumpRun (std::ostream& fOutput, unsigned fRun) {
    fOutput << "<RUN>"<< std::endl;
    fOutput << "   <RUN_TYPE>" << "CastorDbXml" << "</RUN_TYPE>"<< std::endl;
    fOutput << "   <RUN_NUMBER>" << fRun << "</RUN_NUMBER>"<< std::endl;
    fOutput << "</RUN>" << std::endl;
  }

  void dumpHeader (std::ostream& fOutput, unsigned fRun, const std::string& fTableName, const std::string& fTypeName) {
    fOutput << "  <HEADER>" << std::endl;
    fOutput << "    <TYPE>" << std::endl;
    fOutput << "      <EXTENSION_TABLE_NAME>" << fTableName << "</EXTENSION_TABLE_NAME>" << std::endl;
    fOutput << "      <NAME>" << fTypeName << "</NAME>" << std::endl;
    fOutput << "    </TYPE>" << std::endl;
    dumpRun (fOutput, fRun);
    fOutput << "  </HEADER>" << std::endl;
  }

  void dumpFooter (std::ostream& fOutput) {
    fOutput << "</ROOT>" << std::endl;
  }

  void dumpChannelId (std::ostream& fOutput, DetId fChannel) {
    CastorText2DetIdConverter converter (fChannel);
    fOutput << "<CHANNEL> "<< std::endl;
    fOutput << "   <EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME> "<< std::endl;
    fOutput << "   <ETA>" << abs (converter.getField (1)) << "</ETA>"<< std::endl;
    fOutput << "   <PHI>" << converter.getField (2) << "</PHI> "<< std::endl;
    fOutput << "   <DEPTH>" << converter.getField (3) << "</DEPTH> "<< std::endl;
    fOutput << "   <Z>" << (converter.getField (1) > 0 ? "1" : "-1") << "</Z> "<< std::endl;
    fOutput << "   <DETECTOR_NAME>" << converter.getFlavor () << "</DETECTOR_NAME> "<< std::endl;
    fOutput << "   <HCAL_CHANNEL_ID>" << converter.getId().rawId () << "</HCAL_CHANNEL_ID> "<< std::endl;
    fOutput << "</CHANNEL>"<< std::endl;
    fOutput << std::endl;
  }

  void dumpData (std::ostream& fOutput, const float* fValues, const CastorPedestalWidth& fErrors) {
    fOutput << "<DATA> "<< std::endl;
    fOutput << "   <CAPACITOR_0_VALUE>" << fValues [0] << "</CAPACITOR_0_VALUE> "<< std::endl;
    fOutput << "   <CAPACITOR_1_VALUE>" << fValues [1] << "</CAPACITOR_1_VALUE> "<< std::endl;
    fOutput << "   <CAPACITOR_2_VALUE>" << fValues [2] << "</CAPACITOR_2_VALUE> "<< std::endl;	
    fOutput << "   <CAPACITOR_3_VALUE>" << fValues [3] << "</CAPACITOR_3_VALUE> "<< std::endl;
    fOutput << "   <SIGMA_0_0>" << fErrors.getSigma (0,0) << "</SIGMA_0_0> "<< std::endl;	
    fOutput << "   <SIGMA_1_1>" << fErrors.getSigma (1,1) << "</SIGMA_1_1> "<< std::endl;	
    fOutput << "   <SIGMA_2_2>" << fErrors.getSigma (2,2) << "</SIGMA_2_2> "<< std::endl;	
    fOutput << "   <SIGMA_3_3>" << fErrors.getSigma (3,3) << "</SIGMA_3_3> "<< std::endl;	
    fOutput << "   <SIGMA_0_1>" << fErrors.getSigma (1,0) << "</SIGMA_0_1> "<< std::endl;	
    fOutput << "   <SIGMA_0_2>" << fErrors.getSigma (2,0) << "</SIGMA_0_2> "<< std::endl;	
    fOutput << "   <SIGMA_0_3>" << fErrors.getSigma (3,0) << "</SIGMA_0_3> "<< std::endl;	
    fOutput << "   <SIGMA_1_2>" << fErrors.getSigma (2,1) << "</SIGMA_1_2> "<< std::endl;	
    fOutput << "   <SIGMA_1_3>" << fErrors.getSigma (3,1) << "</SIGMA_1_3> "<< std::endl;	
    fOutput << "   <SIGMA_2_3>" << fErrors.getSigma (3,2) << "</SIGMA_2_3> "<< std::endl;	
    fOutput << "</DATA> " << std::endl;
  }

  void dumpData (std::ostream& fOutput, const float* fValues, const float* fErrors) {
    fOutput << "<DATA> "<< std::endl;
    fOutput << "   <CAPACITOR_0_VALUE>" << fValues [0] << "</CAPACITOR_0_VALUE> "<< std::endl;
    fOutput << "   <CAPACITOR_1_VALUE>" << fValues [1] << "</CAPACITOR_1_VALUE> "<< std::endl;
    fOutput << "   <CAPACITOR_2_VALUE>" << fValues [2] << "</CAPACITOR_2_VALUE> "<< std::endl;
    fOutput << "   <CAPACITOR_3_VALUE>" << fValues [3] << "</CAPACITOR_3_VALUE> "<< std::endl;
    fOutput << "   <CAPACITOR_0_ERROR>" << fErrors [0] << "</CAPACITOR_0_ERROR> "<< std::endl;
    fOutput << "   <CAPACITOR_1_ERROR>" << fErrors [1] << "</CAPACITOR_1_ERROR> "<< std::endl;
    fOutput << "   <CAPACITOR_2_ERROR>" << fErrors [2] << "</CAPACITOR_2_ERROR> "<< std::endl;
    fOutput << "   <CAPACITOR_3_ERROR>" << fErrors [3] << "</CAPACITOR_3_ERROR> "<< std::endl;
    fOutput << "</DATA> " << std::endl;
  }

  void dumpDataset (std::ostream& fOutput, unsigned fVersion = 0, const std::string& fFileName = "", const std::string& fDescription = "") {
    fOutput << "<DATA_SET>" << std::endl;
    fOutput << "   <VERSION>" << fVersion << "</VERSION>" << std::endl;
    if (!fFileName.empty ()) 
      fOutput << "      <DATA_FILE_NAME>" << fFileName << "</DATA_FILE_NAME>" << std::endl;
    if (!fDescription.empty ())
      fOutput << "      <COMMENT_DESCRIPTION>" << fDescription << "</COMMENT_DESCRIPTION>" << std::endl;
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
    int i = fChannels.size ();
    while (--i >= 0) {
      fOutput << "<DATA_SET id=\"" << i << "\">" << std::endl;
      dumpRun (fOutput, fRun);
      fOutput << "<KIND_OF_CONDITION><NAME>" << fKind << "</NAME></KIND_OF_CONDITION>" << std::endl;
      dumpChannelId (fOutput, fChannels[i]);
      fOutput << "<VERSION>" << fVersion << "</VERSION>" << std::endl;
      fOutput << "</DATA_SET>" << std::endl;
    }
    // set IOV
    fOutput << "<IOV id=\"" << IOV_ID << "\">";
    fOutput << "   <INTERVAL_OF_VALIDITY_BEGIN>" << fGMTIOVBegin << "</INTERVAL_OF_VALIDITY_BEGIN>"<< std::endl;
    fOutput << "   <INTERVAL_OF_VALIDITY_END>" << fGMTIOVEnd << "</INTERVAL_OF_VALIDITY_END>"<< std::endl;
    fOutput << "</IOV>" << std::endl;
    // set TAG
    fOutput << "<TAG id=\"" << TAG_ID << "\" mode=\"create\">"<< std::endl;
    fOutput << "   <TAG_NAME>" << fTag << "</TAG_NAME>"<< std::endl;
    fOutput << "   <DETECTOR_NAME>HCAL</DETECTOR_NAME>"<< std::endl;
    fOutput << "   <COMMENT_DESCRIPTION>Automatically created by CastorDbXml</COMMENT_DESCRIPTION>" << std::endl;
    fOutput << "</TAG>" << std::endl;

    fOutput << "</ELEMENTS>" << std::endl;

    // mapping itself
    fOutput << "<MAPS>" << std::endl;
    fOutput << "<TAG idref=\"" << TAG_ID << "\">" << std::endl;
    fOutput << "<IOV idref=\"" << IOV_ID << "\">" << std::endl;
    i = fChannels.size ();
    while (--i >= 0) {
      fOutput << "<DATA_SET idref=\"" << i << "\"/>" << std::endl;
    }
    fOutput << "</IOV>" << std::endl;
    fOutput << "</TAG>" << std::endl;
    fOutput << "</MAPS>" << std::endl;
  }
}



bool CastorDbXml::dumpObject (std::ostream& fOutput, 
			    unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion, 
			    const CastorPedestals& fObject) {
  float dummyError = 0.0001;
  std::cout << "CastorDbXml::dumpObject-> set default errors: 0.0001, 0.0001, 0.0001, 0.0001" << std::endl;
  CastorPedestalWidths widths(fObject.isADC() );
  std::vector<DetId> channels = fObject.getAllChannels ();
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {

    CastorPedestalWidth item(*channel);
    for (int iCapId = 1; iCapId <= 4; iCapId++) {
      item.setSigma (iCapId, iCapId, dummyError*dummyError);
    }
    widths.addValues(item);

  }
  return dumpObject (fOutput, fRun, fGMTIOVBegin, fGMTIOVEnd, fTag, fVersion, fObject, widths);
}

bool CastorDbXml::dumpObject (std::ostream& fOutput, 
			    unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion, 
			    const CastorPedestals& fObject, const CastorPedestalWidths& fError) {
  const std::string KIND = "HCAL_PEDESTALS_V2";

  dumpProlog (fOutput);
  dumpHeader (fOutput, fRun, KIND, KIND);

  std::vector<DetId> channels = fObject.getAllChannels ();
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    DetId chId = *channel;
    const float* values = fObject.getValues (chId)->getValues ();
    const CastorPedestalWidth* errors = fError.getValues (chId);
    if (!values) {
      std::cerr << "CastorDbXml::dumpObject-> Can not get data for channel " << CastorText2DetIdConverter(chId).toString () << std::endl;
      continue;
    }
    if (!errors) {
      std::cerr << "CastorDbXml::dumpObject-> Can not get errors for channel " << CastorText2DetIdConverter(chId).toString () <<  ". Use defaults" << std::endl;
      continue;
    }
    dumpDataset (fOutput, fVersion, "", "");
    dumpChannelId (fOutput,chId); 
    dumpData (fOutput, values, *errors);
    endDataset (fOutput);
  }
  dumpMapping (fOutput, fRun, KIND, fGMTIOVBegin, fGMTIOVEnd, fTag, fVersion, channels);

  dumpFooter (fOutput);
  return true;
}

bool CastorDbXml::dumpObject (std::ostream& fOutput, 
			    unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion,
			    const CastorGains& fObject) {
  float dummyErrors [4] = {0., 0., 0., 0.};
  std::cout << "CastorDbXml::dumpObject-> set default errors: 4 x 0.0" << std::endl;

  CastorGainWidths widths;
  std::vector<DetId> channels = fObject.getAllChannels ();
  for (std::vector<DetId>::iterator channel = channels.begin (); channel !=  channels.end (); channel++) 
    {
      CastorGainWidth item(*channel,dummyErrors[0],dummyErrors[1],dummyErrors[2],dummyErrors[3]);
      widths.addValues(item);
    }

  return dumpObject (fOutput, fRun, fGMTIOVBegin, fGMTIOVEnd, fTag, fVersion, fObject, widths);
}

bool CastorDbXml::dumpObject (std::ostream& fOutput, 
			    unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, unsigned fVersion,
			    const CastorGains& fObject, const CastorGainWidths& fError) {
  const std::string KIND = "HCAL Gains";
  const std::string TABLE = "HCAL_GAIN_PEDSTL_CALIBRATIONS";

  dumpProlog (fOutput);
  dumpHeader (fOutput, fRun, TABLE, KIND);

  std::vector<DetId> channels = fObject.getAllChannels ();
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {
    DetId chId = *channel;
    const float* values = fObject.getValues (chId)->getValues ();
    const float* errors = fError.getValues (chId)->getValues ();
    if (!values) {
      std::cerr << "CastorDbXml::dumpObject-> Can not get data for channel " << CastorText2DetIdConverter(chId).toString () << std::endl;
      continue;
    }
    if (!errors) {
      std::cerr << "CastorDbXml::dumpObject-> Can not get errors for channel " << CastorText2DetIdConverter(chId).toString () <<  ". Use defaults" << std::endl;
      continue;
    }
    dumpDataset (fOutput, fVersion, "", "");
    dumpChannelId (fOutput,chId); 
    dumpData (fOutput, values, errors);
    endDataset (fOutput);
  }
  dumpMapping (fOutput, fRun, KIND, fGMTIOVBegin, fGMTIOVEnd, fTag, fVersion, channels);

  dumpFooter (fOutput);
  return true;
}
