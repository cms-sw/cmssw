

#include "L1Trigger/TextToDigi/plugins/TextToRaw.h"

// system
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/Exception.h"

// Raw data collection
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

using std::cerr;
using std::cout;
using std::endl;
using std::ios;
using std::string;
using std::vector;

const unsigned TextToRaw::EVT_MAX_SIZE;

TextToRaw::TextToRaw(const edm::ParameterSet &iConfig)
    : fedId_(iConfig.getUntrackedParameter<int>("fedId", 745)),
      filename_(iConfig.getUntrackedParameter<std::string>("filename", "slinkOutput.txt")),
      fileEventOffset_(iConfig.getUntrackedParameter<int>("FileEventOffset", 0)),
      nevt_(0) {
  edm::LogInfo("TextToDigi") << "Reading ASCII dump from " << filename_ << std::endl;

  // register the products
  produces<FEDRawDataCollection>();
}

TextToRaw::~TextToRaw() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

/// Append empty digi collection
void TextToRaw::putEmptyDigi(edm::Event &iEvent) {
  std::unique_ptr<FEDRawDataCollection> rawColl(new FEDRawDataCollection());
  // FEDRawData& feddata=rawColl->FEDData(fedId_);
  // feddata.data()[0] = 0;
  iEvent.put(std::move(rawColl));
}

// ------------ method called to produce the data  ------------
void TextToRaw::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  // Skip event if required
  if (nevt_ < fileEventOffset_) {
    putEmptyDigi(iEvent);
    nevt_++;
    return;
  } else if (nevt_ == 0 && fileEventOffset_ < 0) {
    std::string line;
    // skip first fileEventOffset input crossings
    for (unsigned i = 0; i < (unsigned)abs(fileEventOffset_); i++) {
      unsigned iline = 0;
      while (getline(file_, line) && !line.empty()) {
        iline++;
        if (iline * 4 >= EVT_MAX_SIZE)
          throw cms::Exception("TextToRawEventSizeOverflow") << "TextToRaw::produce() : "
                                                             << " read too many lines (" << iline << ": " << line << ")"
                                                             << ", maximum event size is " << EVT_MAX_SIZE << std::endl;
      }
    }
  }

  nevt_++;

  // read file
  std::string line;
  unsigned i = 0;  // count 32-bit words

  // while not encountering dumb errors
  while (getline(file_, line) && !line.empty()) {
    // bail if we reached the EVT_MAX_SIZE
    if (i * 4 >= EVT_MAX_SIZE) {
      throw cms::Exception("TextToRaw") << "Read too many lines from file. Maximum event size is " << EVT_MAX_SIZE
                                        << " lines" << std::endl;
    }

    // convert string to int
    std::istringstream iss(line);
    unsigned long d;
    iss >> std::hex >> d;

    // copy data
    for (int j = 0; j < 4; j++) {
      if ((i * 4 + j) < EVT_MAX_SIZE) {
        char c = (d >> (8 * j)) & 0xff;
        data_[i * 4 + j] = c;
      }
    }

    ++i;

    // bail if we reached the EVT_MAX_SIZE
    if (i >= EVT_MAX_SIZE) {
      throw cms::Exception("TextToRaw") << "Read too many lines from file. Maximum event size is " << EVT_MAX_SIZE
                                        << " lines" << std::endl;
    }
  }

  unsigned evtSize = i * 4;

  // create the collection
  std::unique_ptr<FEDRawDataCollection> rawColl(new FEDRawDataCollection());
  // retrieve the target buffer
  FEDRawData &feddata = rawColl->FEDData(fedId_);
  // Allocate space for header+trailer+payload
  feddata.resize(evtSize);

  // fill FEDRawData object
  for (unsigned i = 0; i < evtSize; ++i) {
    feddata.data()[i] = data_[i];
  }

  // put the collection in the event
  iEvent.put(std::move(rawColl));
}

// ------------ method called once each job just before starting event loop
// ------------
void TextToRaw::beginJob() {
  // open VME file
  file_.open(filename_.c_str(), std::ios::in);
  if (!file_.good()) {
    edm::LogInfo("TextToDigi") << "Failed to open ASCII file " << filename_ << std::endl;
  }
}

// ------------ method called once each job just after ending the event loop
// ------------
void TextToRaw::endJob() { file_.close(); }
