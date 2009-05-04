//-------------------------------------------------
//
/**  \class DTExtLutTester
 *
 *   L1 DT Track Finder Extrapolation Lut Tester
 *
 *
 *   $Date: 2008/05/14 14:52:08 $
 *   $Revision: 1.2 $
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTExtLutTester_h
#define DTExtLutTester_h

#include <FWCore/Framework/interface/EDAnalyzer.h>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTExtLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTExtLutRcd.h"


class DTExtLutTester : public edm::EDAnalyzer {
 public:

  DTExtLutTester(const edm::ParameterSet& ps);

  ~DTExtLutTester();
  
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

 private:

};

#endif
