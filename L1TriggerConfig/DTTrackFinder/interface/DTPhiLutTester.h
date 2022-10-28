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

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTPhiLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTPhiLutRcd.h"

class DTPhiLutTester : public edm::one::EDAnalyzer<> {
public:
  DTPhiLutTester(const edm::ParameterSet& ps);

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::ESGetToken<L1MuDTPhiLut, L1MuDTPhiLutRcd> token_;
};

#endif
