//-------------------------------------------------
//
/**  \class DTTFParametersTester
 *
 *   L1 DT Track Finder Parameters Tester
 *
 *
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTTFParametersTester_h
#define DTTFParametersTester_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFParametersRcd.h"

class DTTFParametersTester : public edm::one::EDAnalyzer<> {
public:
  DTTFParametersTester(const edm::ParameterSet& ps);

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::ESGetToken<L1MuDTTFParameters, L1MuDTTFParametersRcd> token_;
};

#endif
