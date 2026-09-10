/**
*  See header file for a description of this class.
*/

#include <sstream>

#include <HepMC3/GenEvent.h>
#include <HepMC3/Print.h>
#include <HepMC3/Reader.h>
#include <HepMC3/ReaderFactory.h>
#include <HepMC3/Units.h>

#include "IOMC/Input/interface/HepMC3FileReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

HepMC3FileReader::HepMC3FileReader(std::vector<std::string> fileNames)
    : fileNames_(std::move(fileNames)), nextFile_(0) {
  if (fileNames_.empty()) {
    throw cms::Exception("FileNotFound", "HepMC3FileReader") << "No input file was given.\n";
  }
  openNextFile();
}

HepMC3FileReader::~HepMC3FileReader() {
  if (reader_) {
    reader_->close();
  }
}

bool HepMC3FileReader::openNextFile() {
  if (reader_) {
    reader_->close();
    reader_.reset();
  }

  if (nextFile_ >= fileNames_.size()) {
    return false;
  }

  const std::string& fileName = fileNames_[nextFile_];
  ++nextFile_;

  edm::LogInfo("HepMC3FileReader") << "Opening file " << fileName << " using HepMC3::deduce_reader";
  reader_ = HepMC3::deduce_reader(fileName);

  if (!reader_ || reader_->failed()) {
    throw cms::Exception("FileNotFound", "HepMC3FileReader::openNextFile()")
        << "File " << fileName << " was not found or is not a readable HepMC3 file.\n";
  }

  return true;
}

const HepMC3::GenEvent* HepMC3FileReader::fillCurrentEventData() {
  while (reader_) {
    // a fresh event, so that no attribute of the previous one can survive
    auto evt = std::make_unique<HepMC3::GenEvent>(HepMC3::Units::GEV, HepMC3::Units::MM);

    if (reader_->read_event(*evt) && !reader_->failed()) {
      // the simulation expects GeV and mm, whatever the file uses
      evt->set_units(HepMC3::Units::GEV, HepMC3::Units::MM);
      evt_ = std::move(evt);
      return evt_.get();
    }

    // end of this file, continue with the next one if there is any
    if (!openNextFile()) {
      break;
    }
  }

  evt_.reset();
  return nullptr;
}

void HepMC3FileReader::printCurrentEvent() const {
  if (evt_) {
    std::ostringstream str;
    HepMC3::Print::content(str, *evt_);
    edm::LogVerbatim("HepMC3FileReader") << str.str();
  }
}
