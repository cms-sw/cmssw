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

#include <FWCore/Framework/interface/EDAnalyzer.h>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTEtaPatternLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTEtaPatternLutRcd.h"

class DTEtaPatternLutTester : public edm::EDAnalyzer {
public:
  DTEtaPatternLutTester(const edm::ParameterSet& ps);

  ~DTEtaPatternLutTester() override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
};

#endif
