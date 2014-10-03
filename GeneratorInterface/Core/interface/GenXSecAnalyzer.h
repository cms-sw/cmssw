#ifndef GENXSECANALYZER_H
#define GENXSECANALYZER_H

// $Revision://

// analyzer of a summary information product on filter efficiency for a user specified path
// meant for the generator filter efficiency calculation


// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
//
// class declaration
//

class GenXSecAnalyzer : public edm::EDAnalyzer {


public:
  explicit GenXSecAnalyzer(const edm::ParameterSet&);
  ~GenXSecAnalyzer();
  const double final_xsec_value() const {return xsec_.value();}
  const double final_xsec_error() const {return xsec_.error();}
  
private:

  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endJob();
  void compute();

  int hepidwtup_;
  GenLumiInfoProduct::XSec xsec_;
  GenFilterInfo  jetMatchEffStat_;
  GenFilterInfo  totalEffStat_;     // statistics from total filter
  // ----------member data ---------------------------
  std::vector<GenLumiInfoProduct> products_; // the size depends on the number of MC with different LHE information
};

#endif
