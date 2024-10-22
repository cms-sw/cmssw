//-------------------------------------------------
//
/**  \class DTEtaPatternLutTester
 *
 *   L1 DT Track Finder Eta Pattern Lut Tester
 *
 *
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTEtaPatternLutTester_h
#define DTEtaPatternLutTester_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTEtaPatternLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTEtaPatternLutRcd.h"

class DTEtaPatternLutTester : public edm::one::EDAnalyzer<> {
public:
  explicit DTEtaPatternLutTester(const edm::ParameterSet& ps);

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::ESGetToken<L1MuDTEtaPatternLut, L1MuDTEtaPatternLutRcd> token_;
};

#endif
