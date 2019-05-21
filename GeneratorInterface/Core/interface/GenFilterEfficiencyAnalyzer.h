#ifndef GENFILTEREFFICIENCYANALYZER_H
#define GENFILTEREFFICIENCYANALYZER_H

// F. Cossutti
// $Revision://

// analyzer of a summary information product on filter efficiency for a user specified path
// meant for the generator filter efficiency calculation

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
//
// class declaration
//

class GenFilterEfficiencyAnalyzer : public edm::EDAnalyzer {
public:
  explicit GenFilterEfficiencyAnalyzer(const edm::ParameterSet&);
  ~GenFilterEfficiencyAnalyzer() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endJob() override;

  edm::EDGetTokenT<GenFilterInfo> genFilterInfoToken_;
  GenFilterInfo totalGenFilterInfo_;

  // ----------member data ---------------------------
};

#endif
