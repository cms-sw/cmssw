
#ifndef GENXSECANALYZER_H
#define GENXSECANALYZER_H

// $Revision://

// analyzer of a summary information product on filter efficiency for a user specified path
// meant for the generator filter efficiency calculation

// system include files
#include <memory>
#include <vector>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"

//
// class declaration
//

class GenXSecAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  explicit GenXSecAnalyzer(const edm::ParameterSet &);
  ~GenXSecAnalyzer() override;
  const double final_xsec_value() const { return xsec_.value(); }
  const double final_xsec_error() const { return xsec_.error(); }

private:
  void beginJob() override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;
  void endRun(edm::Run const &, edm::EventSetup const &) override;
  void endJob() override;
  // computation of cross section after matching and before HepcFilter and GenFilter
  GenLumiInfoProduct::XSec compute(const GenLumiInfoProduct &);
  // combination of cross section from different MCs after matching (could be either before or after HepcFilter and GenFilter)
  void combine(GenLumiInfoProduct::XSec &, double &, const GenLumiInfoProduct::XSec &, const double &);
  void combine(double &, double &, double &, const double &, const double &, const double &);

  edm::EDGetTokenT<GenFilterInfo> genFilterInfoToken_;
  edm::EDGetTokenT<GenFilterInfo> hepMCFilterInfoToken_;
  edm::EDGetTokenT<GenLumiInfoProduct> genLumiInfoToken_;
  edm::EDGetTokenT<LHERunInfoProduct> lheRunInfoToken_;

  // ----------member data --------------------------

  int nMCs_;

  int hepidwtup_;

  // for weight before GenFilter and HepMCFilter and before matching
  double totalWeightPre_;
  double thisRunWeightPre_;

  // for weight after GenFilter and HepMCFilter and after matching
  double totalWeight_;
  double thisRunWeight_;

  // combined cross sections before HepMCFilter and GenFilter
  GenLumiInfoProduct::XSec xsecPreFilter_;

  // final combined cross sections
  GenLumiInfoProduct::XSec xsec_;

  // GenLumiInfo before HepMCFilter and GenFilter, this is used
  // for computation
  GenLumiInfoProduct product_;

  // statistics from additional generator filter, for computation
  // reset for each run
  GenFilterInfo filterOnlyEffRun_;

  // statistics from HepMC filter, for computation
  GenFilterInfo hepMCFilterEffRun_;

  // statistics from additional generator filter, for print-out only
  GenFilterInfo filterOnlyEffStat_;

  // statistics from HepMC filter, for print-out only
  GenFilterInfo hepMCFilterEffStat_;

  // the vector/map size is the number of LHE processes + 1
  // needed only for printouts, not used for computation
  // only printed out when combining the same physics process
  // uncertainty-averaged cross sections before matching
  std::vector<GenLumiInfoProduct::XSec> xsecBeforeMatching_;
  // uncertainty-averaged cross sections after matching
  std::vector<GenLumiInfoProduct::XSec> xsecAfterMatching_;
  // statistics from jet matching
  std::map<int, GenFilterInfo> jetMatchEffStat_;

  // the following vectors all have the same size
  // LHE or Pythia/Herwig cross section of previous luminosity block
  // vector size = number of processes, used for computation
  std::map<int, GenLumiInfoProduct::XSec> previousLumiBlockLHEXSec_;

  // LHE or Pythia/Herwig combined cross section of current luminosity block
  // updated for each luminosity block, initialized in every run
  // used for computation
  std::map<int, GenLumiInfoProduct::XSec> currentLumiBlockLHEXSec_;
};

#endif
