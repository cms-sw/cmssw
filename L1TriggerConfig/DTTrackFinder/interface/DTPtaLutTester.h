//-------------------------------------------------
//
/**  \class DTPtaLutTester
 *
 *   L1 DT Track Finder Pt Assignment Lut Tester
 *
 *
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTPtaLutTester_h
#define DTPtaLutTester_h

#include <FWCore/Framework/interface/EDAnalyzer.h>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTPtaLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTPtaLutRcd.h"

class DTPtaLutTester : public edm::EDAnalyzer {
public:
  DTPtaLutTester(const edm::ParameterSet& ps);

  ~DTPtaLutTester() override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
};

#endif
