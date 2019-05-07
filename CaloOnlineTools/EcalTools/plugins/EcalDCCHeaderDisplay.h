/**
 * module  for displaying unpacked DCCHeader information
 *
 * \author A. Ghezzi
 * \author S. Cooper
 * \author G. Franzoni
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <iostream>
#include <vector>

class EcalDCCHeaderDisplay : public edm::EDAnalyzer {

public:
  EcalDCCHeaderDisplay(const edm::ParameterSet &ps);

protected:
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

private:
  edm::InputTag EcalDCCHeaderCollection_;
};
