#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibErrors.h"
#include "CondTools/Ecal/interface/EcalTimeCalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalTimeCalibErrorsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <iostream>

void usage()
{
  std::cout
    << "Usage: CombineEcalOfflineTimingCalibsXML [originalCalibsFile] [fileWithNewCalibs]"
    << std::endl
    << "\tThe values in fileWithNewCalibs will replace the corresponding ones in originalCalibsFile; all other calibs in originalCalibsFile will be untouched."
    << std::endl;
}

// ****************************************************************
int main(int argc, char* argv[])
{
  //
  // Binary to combine 2 sets of timing calibrations
  // Seth Cooper
  // Updated July 6 2011: No longer uses the error files (we ignore them for now anyway)
  //                      Just replaces the calibrations in the original file with those passed in
  // Usage: ShiftEcalOfflineTimingCalibsXML <originalCalibsFile> <fileWithChanges>
  //   The calibs in the fileWithChanges will replace those in the original file.
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

  // Loop over the calib constants and apply the replacements
  for(int hash = 0; hash < EBDetId::MAX_HASH; ++hash)
  {
    float newCalib = 0;
    EBDetId thisDet = EBDetId::unhashIndex(hash);
    if(thisDet==EBDetId())
      continue;

    // Look for constant to replace
    if(shiftConstants.find(thisDet.rawId())==shiftConstants.end())
      continue;

    newCalib = shiftConstants[thisDet.rawId()];
    //cout << "Crystal " << thisDet << " calibration now " << newCalib
    //  << "; was " << calibConstants[thisDet.rawId()]
    //  << endl;
    calibConstants[thisDet.rawId()] = newCalib;
  }
  for(int hash = 0; hash < EEDetId::kSizeForDenseIndexing; ++hash)
  {
    float newCalib = 0;
    EEDetId thisDet = EEDetId::unhashIndex(hash);
    if(thisDet==EEDetId())
      continue;

    if(shiftConstants.find(thisDet.rawId())==shiftConstants.end())
      continue;

    newCalib = calibConstants[thisDet.rawId()];
    //cout << "Crystal " << thisDet << " calibration now " << newCalib
    //  << "; was  " << calibConstants[thisDet.rawId()]
    //  << endl;
    calibConstants[thisDet.rawId()] = newCalib;
  }

  // Write new XML file
  string outputFile = "ecalOfflineTimingCalibsCombined.xml";
  EcalTimeCalibConstantsXMLTranslator::writeXML(outputFile,calibFileHeader,calibConstants);

}


