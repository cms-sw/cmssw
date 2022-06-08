#ifndef RCTDIGITORCTTEXT_H
#define RCTDIGITORCTTEXT_H

/*\class RctDigiToRctText
 *\description produces from RCT digis RCT data files
 *  format specified by Pam Klabbers
 *\author Nuno Leonardo (CERN)
 *\created Thu Mar 29 23:22:57 CEST 2007
 */

// system include files
#include <fstream>
#include <iostream>
#include <memory>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"  // Logger
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

const static unsigned NUM_RCT_CRATES = 18;

class RctDigiToRctText : public edm::one::EDAnalyzer<> {
public:
  explicit RctDigiToRctText(const edm::ParameterSet &);
  ~RctDigiToRctText() override;

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  /// label for RCT digis
  edm::InputTag m_rctInputLabel;

  /// basename for output files
  std::string m_textFileName;

  /// write upper case hex words
  bool m_hexUpperCase;

  /// handles for output files
  std::ofstream m_file[NUM_RCT_CRATES];

  /// handle for debug file
  std::ofstream fdebug;

  int nevt = -1;
};

#endif
