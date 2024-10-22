// user file includes
#include "CalibTracker/SiStripCommon/interface/SiStripFedIdListReader.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// system includes
#include <string>

/**
   @class testSiStripFedIdListReader
   @author R.Bainbridge
*/

class testSiStripFedIdListReader : public edm::global::EDAnalyzer<> {
public:
  explicit testSiStripFedIdListReader(const edm::ParameterSet &);
  ~testSiStripFedIdListReader() = default;
  void analyze(edm::StreamID, const edm::Event &, const edm::EventSetup &) const override;

private:
  edm::FileInPath fileInPath_;
};

// -----------------------------------------------------------------------------
//
testSiStripFedIdListReader::testSiStripFedIdListReader(const edm::ParameterSet &pset)
    : fileInPath_(pset.getParameter<edm::FileInPath>("file")) {
  edm::LogVerbatim("Unknown") << "[testSiStripFedIdListReader::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
void testSiStripFedIdListReader::analyze(edm::StreamID, const edm::Event &, const edm::EventSetup &) const {
  SiStripFedIdListReader reader(fileInPath_.fullPath());
  edm::LogVerbatim("Unknown") << "[testSiStripFedIdListReader::" << __func__ << "]" << reader;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testSiStripFedIdListReader);
