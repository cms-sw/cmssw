#ifndef GCTDIGITOPSBTEXT_H
#define GCTDIGITOPSBTEXT_H

/*\class GctDigiToPsbText
 *\description produces from GCT digis expected GT PSB files
 *\author Nuno Leonardo (CERN)
 *\date 08.08
 */

// system include files
#include <fstream>
#include <iostream>
#include <memory>
// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// gct
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

class GctDigiToPsbText : public edm::one::EDAnalyzer<> {
public:
  explicit GctDigiToPsbText(const edm::ParameterSet &);
  ~GctDigiToPsbText() override;

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  /// label for input digis
  edm::InputTag m_gctInputLabel;

  /// basename for output files
  std::string m_textFileName;

  /// write upper case hex words
  bool m_hexUpperCase;

  /// handles for output files
  std::ofstream m_file[4];

  /// handle for debug file
  std::ofstream fdebug;
};

#endif
