
#include "L1Trigger/TextToDigi/plugins/RawToText.h"

// system
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
// framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
// raw data collection
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

RawToText::RawToText(const edm::ParameterSet &iConfig)
    : inputLabel_(iConfig.getParameter<edm::InputTag>("inputLabel")),
      fedId_(iConfig.getUntrackedParameter<int>("fedId", 745)),
      filename_(iConfig.getUntrackedParameter<std::string>("filename", "slinkOutput.txt")),
      nevt_(0) {
  edm::LogInfo("TextToDigi") << "Creating ASCII dump " << filename_ << std::endl;
}

RawToText::~RawToText() {}

void RawToText::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  nevt_++;

  // get raw data collection
  edm::Handle<FEDRawDataCollection> feds;
  iEvent.getByLabel(inputLabel_, feds);
  const FEDRawData &gctRcd = feds->FEDData(fedId_);

  edm::LogInfo("GCT") << "Upacking FEDRawData of size " << std::dec << gctRcd.size() << std::endl;

  // do a simple check of the raw data
  if (gctRcd.size() < 16) {
    edm::LogWarning("Invalid Data") << "Empty/invalid GCT raw data, size = " << gctRcd.size() << std::endl;
    return;
  }

  const unsigned char *data = gctRcd.data();

  int eventSize = gctRcd.size() / 4;

  unsigned long d = 0;
  for (int i = 0; i < eventSize; i++) {
    d = 0;
    // d  = data[i*4+0] + (data[i*4+1]<<8) + (data[i*4+2]<<16) +
    // (data[i*4+3]<<24);
    for (int j = 0; j < 4; j++) {
      d += ((data[i * 4 + j] & 0xff) << (8 * j));
    }
    file_ << std::setw(8) << std::setfill('0') << std::hex << d << std::endl;
  }
  file_ << std::flush << std::endl;
}

void RawToText::beginJob() {
  // open VME file
  file_.open(filename_.c_str(), std::ios::out);

  if (!file_.good()) {
    edm::LogInfo("RawToText") << "Failed to open ASCII file " << filename_ << std::endl;
  }
}

void RawToText::endJob() { file_.close(); }
