/*! \file testJets.cpp
 * \test file for testing the Jet Et calibration LUT class
 *
 *  
 *
 * \author Alex Tapper
 * \date July 2006
 */

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

#include "L1Trigger/GlobalCaloTrigger/test/produceTrivialCalibrationLut.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <exception>
#include <vector>


typedef std::vector<L1GctJet::lutPtr> lutPtrVector;

using namespace std;

int main()
{
  try {

    //create jet calibration lookup table
    produceTrivialCalibrationLut* lutProducer=new produceTrivialCalibrationLut();
    lutProducer->setOrcaStyleCorrectionType();

    // Instance of the class
    lutPtrVector lutVector = lutProducer->produce();

    delete lutProducer;

    // print it out
    for (lutPtrVector::const_iterator lut = lutVector.begin(); lut != lutVector.end(); lut++) {
      cout << "\n===Lookup table for eta=" << (*lut)->etaBin() << "===" << endl;
      cout << (**lut) << endl;
    }
  }
  catch (cms::Exception& e)
    {
      cerr << "CMS exception from " << e.what() << endl;
    }
  catch (std::exception& e)
    {
      cerr << "std exception from " << e.what() << endl;
    }
  catch (...) { // Catch anything unknown that goes wrong! 
    cerr << "yikes! something awful happened.... (unknown exception)" << endl; 
  } 
}
