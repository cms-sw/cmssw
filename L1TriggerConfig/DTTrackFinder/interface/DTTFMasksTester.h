//-------------------------------------------------
//
/**  \class DTTFMasksTester
 *
 *   L1 DT Track Finder Parameters Tester
 *
 *
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTTFMasksTester_h
#define DTTFMasksTester_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFMasksRcd.h"

class DTTFMasksTester : public edm::one::EDAnalyzer<> {
public:
  DTTFMasksTester(const edm::ParameterSet& ps);

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::ESGetToken<L1MuDTTFMasks, L1MuDTTFMasksRcd> token_;
};

#endif
