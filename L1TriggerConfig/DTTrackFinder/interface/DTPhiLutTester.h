//-------------------------------------------------
//
/**  \class DTPhiLutTester
 *
 *   L1 DT Track Finder Phi Assignment Lut Tester
 *
 *
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTPhiLutTester_h
#define DTPhiLutTester_h

#include <FWCore/Framework/interface/EDAnalyzer.h>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTPhiLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTPhiLutRcd.h"

class DTPhiLutTester : public edm::EDAnalyzer {
public:
  DTPhiLutTester(const edm::ParameterSet& ps);

  ~DTPhiLutTester() override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
};

#endif
