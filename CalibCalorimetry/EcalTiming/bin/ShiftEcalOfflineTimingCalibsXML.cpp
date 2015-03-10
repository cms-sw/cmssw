#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondTools/Ecal/interface/EcalTimeCalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <iostream>

void usage()
{
  std::cout << "Usage: ShiftEcalOfflineTimingCalibsXML [originalCalibsFile] [fileWithChanges]"
    << std::endl
    << "\tThe values in fileWithChanges will be added to those in originalCalibsFile."
    << std::endl;
}

// ****************************************************************
int main(int argc, char* argv[])
{
  //
  // Binary to shift a timing XML file using an input XML file
  // Seth Cooper
  // October 29 2010
  // Usage: ShiftEcalOfflineTimingCalibsXML <originalCalibsFile> <fileWithChanges>
  //   The calibs in the file to shift will be added to those in the original file
  //

  using namespace std;

  char* calibFile = argv[1];
  if (!calibFile)
  {
    cout << "Error: Missing input file." << endl;
    usage();
    return -1;
  }

  char* shiftFile = argv[2];
  if (!shiftFile)
  {
    cout << "Error: Missing input file." << endl;
    usage();
    return -1;
  }

  EcalCondHeader calibFileHeader;
  EcalCondHeader shiftFileHeader;

  string calibFileStr(calibFile);
  string shiftFileStr(shiftFile);

  EcalTimeCalibConstants calibConstants;
  EcalTimeCalibConstants shiftConstants;

  // Populate the EcalTimeCalibConstants objects by reading in the files
  int ret = EcalTimeCalibConstantsXMLTranslator::readXML(calibFileStr,calibFileHeader,calibConstants);
  if(ret)
  {
    cout << "Error reading calibration XML file" << endl;
    return -2;
  }
  ret = EcalTimeCalibConstantsXMLTranslator::readXML(shiftFileStr,shiftFileHeader,shiftConstants);
  if(ret)
  {
    cout << "Error reading shift XML file" << endl;
    return -2;
  }

  // Loop over the calib constants and apply any necessary shifts
  for(int hash = 0; hash < EBDetId::MAX_HASH; ++hash)
  {
    EBDetId thisDet = EBDetId::unhashIndex(hash);
    if(thisDet==EBDetId())
      continue;

    if(calibConstants.find(thisDet.rawId())==calibConstants.end())
      continue;
    if(shiftConstants.find(thisDet.rawId())==shiftConstants.end())
      continue;
    if(shiftConstants[thisDet.rawId()]==0)
      continue;

    // Calib found and there is one to shift as well, make new calib and store it
    float newCalib = calibConstants[thisDet.rawId()]+shiftConstants[thisDet.rawId()];
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
    if(shiftConstants.find(thisDet.rawId())==shiftConstants.end())
      continue;
    if(shiftConstants[thisDet.rawId()]==0)
      continue;

    // Calib found and there is one to shift as well, make new calib and store it
    float newCalib = calibConstants[thisDet.rawId()]+shiftConstants[thisDet.rawId()];
    //cout << "Crystal " << thisDet << " calibration now " << newCalib
    //  << " = " << calibConstants[thisDet.rawId()]
    //  << " + " << shiftConstants[thisDet.rawId()]
    //  << endl;
    calibConstants[thisDet.rawId()] = newCalib;
  }

  // Write new XML file
  string outputFile = "offlineTimingCalibsShifted.xml";
  EcalTimeCalibConstantsXMLTranslator::writeXML(outputFile,calibFileHeader,calibConstants);


}


