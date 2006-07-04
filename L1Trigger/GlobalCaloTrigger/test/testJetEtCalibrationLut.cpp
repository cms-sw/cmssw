/*! \file testJets.cpp
 * \test file for testing the Jet Et calibration LUT class
 *
 *  
 *
 * \author Alex Tapper
 * \date July 2006
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h" 
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <exception>
#include <vector>

using namespace std;

int main()
{
  try {
    // Instance of the class
    L1GctJetEtCalibrationLut* lut = new L1GctJetEtCalibrationLut("data/testJetEtCalibrationLut.dat");

    // print it out
    cout << (*lut);
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
