/*! \file testJets.cpp
 * \test file for testing the Jet Et calibration LUT class
 *
 *  
 *
 * \author Alex Tapper
 * \date July 2006
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
 
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <exception>
#include <vector>

using namespace std;

int main()
{
  try {

    // Manually set up the gct configuration
    double lsb=1.0;
    static const unsigned nThresh=64;
    vector<double> thresh(nThresh);
    thresh.at(0) = 0.0;
    for (unsigned t=1; t<nThresh; ++t) {
      thresh.at(t) = t*16.0 - 8.0;
    }

    double threshold=5.0;
    vector< vector<double> > defaultCalib;
    L1CaloEtScale* myScale = new L1CaloEtScale(lsb, thresh);
    L1GctJetEtCalibrationFunction* myFun = new L1GctJetEtCalibrationFunction();
    myFun->setOutputEtScale(*myScale);
    myFun->setParams(lsb, threshold,
                     defaultCalib, defaultCalib);

    // Instance of the class
    L1GctJetEtCalibrationLut* lut = L1GctJetEtCalibrationLut::setupLut(myFun);

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
