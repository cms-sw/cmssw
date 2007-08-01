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

using namespace std;

int main()
{
  try {

    //create jet calibration lookup table
    produceTrivialCalibrationLut* lutProducer=new produceTrivialCalibrationLut();
    lutProducer->setOrcaStyleCorrectionType();

    // Instance of the class
    L1GctJetEtCalibrationLut* lut1 = lutProducer->produce();

    // Another instance with different parameters
    lutProducer->setOldOrcaStyleCorrectionType();
    L1GctJetEtCalibrationLut* lut2 = lutProducer->produce();

    delete lutProducer;

    for (unsigned eta=0; eta<11; eta++) {
      float sumDiffEt   = 0.0;
      float sumDiffRank = 0.0;
      for (uint16_t rawEt=0; rawEt<1024; rawEt++) {
        L1GctJet j(rawEt,eta,0,(eta>=7),true);

        unsigned calEt1 = j.calibratedEt(lut1);
        unsigned calEt2 = j.calibratedEt(lut2);

        unsigned jRank1 = j.jetCand(lut1).rank();
        unsigned jRank2 = j.jetCand(lut2).rank();

        sumDiffEt   += float(calEt1) - float(calEt2);
        sumDiffRank += float(jRank1) - float(jRank2);

      }
      cout << "eta = " << eta << " average Et difference " << sumDiffEt/1024.
                              << " average rank difference " << sumDiffRank/1024. << endl;
    }

    // print it out
    //cout << (*lut);
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
