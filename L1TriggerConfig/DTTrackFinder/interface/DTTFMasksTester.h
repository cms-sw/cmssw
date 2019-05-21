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

#include <FWCore/Framework/interface/EDAnalyzer.h>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFMasksRcd.h"

class DTTFMasksTester : public edm::EDAnalyzer {
public:
  DTTFMasksTester(const edm::ParameterSet& ps);

  ~DTTFMasksTester() override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
};

#endif
