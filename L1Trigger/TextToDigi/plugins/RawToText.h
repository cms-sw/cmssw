#ifndef RAWTOTEXT_H
#define RAWTOTEXT_H

/*\class RawToText
 *\description conversion of GCT raw to text formats
               based on and reverses J.Brooke's TextToRaw
 *\usage trigger pattern tests
 *\author Nuno Leonardo (CERN)
 *\date 07.08
 */

// system include files
#include <fstream>
#include <memory>
#include <string>
// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

class RawToText : public edm::one::EDAnalyzer<> {
public:
  explicit RawToText(const edm::ParameterSet &);
  ~RawToText() override;

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

private:
  // FED collection label
  edm::InputTag inputLabel_;

  // ID of the FED to emulate
  int fedId_;

  // File to write
  std::string filename_;
  std::ofstream file_;

  int nevt_;
};

#endif
