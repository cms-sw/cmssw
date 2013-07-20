#ifndef GENFILTEREFFICIENCYANALYZER_H
#define GENFILTEREFFICIENCYANALYZER_H

// F. Cossutti
// $Date: 2013/03/01 19:06:14 $
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
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
//
// class declaration
//

class GenFilterEfficiencyAnalyzer : public edm::EDAnalyzer {
public:
  explicit GenFilterEfficiencyAnalyzer(const edm::ParameterSet&);
  ~GenFilterEfficiencyAnalyzer();
  
  
private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  virtual void endJob() override;

  int nTota_;
  int nPass_;
  edm::InputTag genFilterInfoTag_;

  // ----------member data ---------------------------
  
};

#endif
