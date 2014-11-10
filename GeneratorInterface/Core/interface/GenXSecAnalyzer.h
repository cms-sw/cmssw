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
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

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

class GenXSecAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchLuminosityBlocks> {


public:
  explicit GenXSecAnalyzer(const edm::ParameterSet&);
  ~GenXSecAnalyzer();
  const double final_xsec_value() const {return xsec_.value();}
  const double final_xsec_error() const {return xsec_.error();}
  
private:

  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  virtual void endJob() override;
  void compute();

  edm::EDGetTokenT<GenFilterInfo> genFilterInfoToken_;
  edm::EDGetTokenT<GenFilterInfo> hepMCFilterInfoToken_;
  edm::EDGetTokenT<GenLumiInfoProduct> genLumiInfoToken_;
  
  // ----------member data --------------------------

  int hepidwtup_;
  unsigned int theProcesses_size;
  bool           hasHepMCFilterInfo_;

  // final cross sections
  GenLumiInfoProduct::XSec xsec_;
  // statistics from additional generator filter
  GenFilterInfo  filterOnlyEffStat_;     

  // statistics from HepMC filter
  GenFilterInfo  hepMCFilterEffStat_;   

  // statistics for event level efficiency, the size is the number of processes + 1 
  std::vector<GenFilterInfo>  eventEffStat_; 
  // statistics from jet matching, the size is the number of processes + 1 
  std::vector<GenFilterInfo>  jetMatchEffStat_; 
  // uncertainty-averaged cross sections before matching, the size is the number of processes + 1
  std::vector<GenLumiInfoProduct::XSec> xsecBeforeMatching_;
  // uncertainty-averaged cross sections after matching, the size is the number of processes + 1 
  std::vector<GenLumiInfoProduct::XSec> xsecAfterMatching_; 
  // the size depends on the number of MC with different LHE information
  std::vector<GenLumiInfoProduct> products_; 

};

#endif
