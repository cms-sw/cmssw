//-------------------------------------------------
//
/**  \class DTExtLutTester
 *
 *   L1 DT Track Finder Extrapolation Lut Tester
 *
 *
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTExtLutTester_h
#define DTExtLutTester_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTExtLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTExtLutRcd.h"

class DTExtLutTester : public edm::one::EDAnalyzer<> {
public:
  DTExtLutTester(const edm::ParameterSet& ps);

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::ESGetToken<L1MuDTExtLut, L1MuDTExtLutRcd> token_;
};

#endif
