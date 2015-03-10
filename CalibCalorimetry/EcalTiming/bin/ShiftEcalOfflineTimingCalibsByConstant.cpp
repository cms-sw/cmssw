#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondTools/Ecal/interface/EcalTimeCalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <iostream>

void usage()
{
  std::cout << "Usage: ShiftEcalOfflineTimingCalibsByConstant [originalCalibsFile] [shiftInNs]"
    << std::endl
    << "\tshiftInNs will be added to each calibration in originalCalibsFile."
    << std::endl
    << "\tNote: The output will always have an entry for each crystal, even if the input did not!"
    << std::endl;
}

// ****************************************************************
int main(int argc, char* argv[])
{
  //
  // Binary to shift a timing XML file by a constant amount
  // Seth Cooper
  // November 8 2010
  // Usage: ShiftEcalOfflineTimingCalibsByConstant <originalCalibsFile> <shiftInNs>
  //   shiftInNs will be added to each calibration in the original file
  //
  // Note that the output ALWAYS has all crystals, even though some may be empty in the input!

  using namespace std;

  char* calibFile = argv[1];
  if(!calibFile)
  {
    cout << "Error: Missing input file." << endl;
    usage();
    return -1;
  }
  char* shiftChar = argv[2];
  float shiftInNs = atof(shiftChar);

  EcalCondHeader calibFileHeader;

  string calibFileStr(calibFile);

  EcalTimeCalibConstants calibConstants;

  // Populate the EcalTimeCalibConstants object by reading in the files
  int ret = EcalTimeCalibConstantsXMLTranslator::readXML(calibFileStr,calibFileHeader,calibConstants);
  if(ret)
  {
    cout << "Error reading calibration XML file" << endl;
    return -2;
  }

  // Loop over the calib constants and apply shift
  for(int hash = 0; hash < EBDetId::MAX_HASH; ++hash)
  {
    EBDetId thisDet = EBDetId::unhashIndex(hash);
    if(thisDet==EBDetId())
      continue;

    if(calibConstants.find(thisDet.rawId())==calibConstants.end())
      continue;
    if(calibConstants[thisDet.rawId()]==0) // 0 is a special case --> empty entry
      continue;

    float newCalib = calibConstants[thisDet.rawId()]+shiftInNs;
    //cout << "Crystal " << thisDet << " calibration now " << newCalib
    //  << " = " << calibConstants[thisDet.rawId()]
    //  << " + " << shiftConstants[thisDet.rawId()]
    //  << endl;
    calibConstants[thisDet.rawId()] = newCalib;
  }
  for(int hash = 0; hash < EEDetId::kSizeForDenseIndexing; ++hash)
  {
    EEDetId thisDet = EEDetId::unhashIndex(hash);
    if(thisDet==EEDetId())
      continue;

    if(calibConstants.find(thisDet.rawId())==calibConstants.end())
      continue;
    if(calibConstants[thisDet.rawId()]==0) // 0 is a special case --> empty entry
      continue;

    float newCalib = calibConstants[thisDet.rawId()]+shiftInNs;
    //cout << "Crystal " << thisDet << " calibration now " << newCalib
    //  << " = " << calibConstants[thisDet.rawId()]
    //  << " + " << shiftConstants[thisDet.rawId()]
    //  << endl;
    calibConstants[thisDet.rawId()] = newCalib;
  }

  // Write new XML file
  string outputFile = "offlineTimingCalibsShiftedByConst.xml";
  EcalTimeCalibConstantsXMLTranslator::writeXML(outputFile,calibFileHeader,calibConstants);
}


