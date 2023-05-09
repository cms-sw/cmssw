#include "EventFilter/Utilities/interface/DAQSource.h"
#include "EventFilter/Utilities/interface/DAQSourceModelsScouting.h"

#include <sstream>
#include <sys/types.h>
#include <vector>
#include <regex>

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"

using namespace scouting;

void DataModeScoutingRun2Muon::readEvent(edm::EventPrincipal& eventPrincipal) {
  edm::TimeValue_t time;
  timeval stv;
  gettimeofday(&stv, nullptr);
  time = stv.tv_sec;
  time = (time << 32) + stv.tv_usec;
  edm::Timestamp tstamp(time);

  std::unique_ptr<BXVector<l1t::Muon>> rawData(new BXVector<l1t::Muon>);
  //allow any bx
  rawData->setBXRange(0, 4000);

  unpackOrbit(rawData.get(), (char*)event_->payload(), event_->eventSize());

  uint32_t hdrEventID = event_->event();
  edm::EventID eventID = edm::EventID(daqSource_->eventRunNumber(), daqSource_->currentLumiSection(), hdrEventID);
  edm::EventAuxiliary aux(
      eventID, daqSource_->processGUID(), tstamp, event_->isRealData(), edm::EventAuxiliary::PhysicsTrigger);

  aux.setProcessHistoryID(daqSource_->processHistoryID());
  daqSource_->makeEventWrapper(eventPrincipal, aux);

  std::unique_ptr<edm::WrapperBase> edp(new edm::Wrapper<BXVector<l1t::Muon>>(std::move(rawData)));
  eventPrincipal.put(
      daqProvenanceHelpers_[0]->branchDescription(), std::move(edp), daqProvenanceHelpers_[0]->dummyProvenance());
}

void DataModeScoutingRun2Muon::unpackOrbit(BXVector<l1t::Muon>* muons, char* buf, size_t len) {
  using namespace scouting;
  size_t pos = 0;
  uint32_t o_test = 0;
  while (pos < len) {
    assert(pos + 4 <= len);
    uint32_t header = *((uint32*)(buf + pos));
    uint32_t mAcount = (header & header_masks::mAcount) >> header_shifts::mAcount;
    uint32_t mBcount = (header & header_masks::mBcount) >> header_shifts::mBcount;

    block* bl = (block*)(buf + pos + 4);

    pos += 12 + (mAcount + mBcount) * 8;
    assert(pos <= len);

    uint32_t bx = bl->bx;

    uint32_t orbit = bl->orbit;
    o_test = orbit;

    //should cuts should be applied
    bool excludeIntermediate = true;

    for (size_t i = 0; i < (mAcount + mBcount); i++) {
      //unpack new muon
      //variables: index, ietaext, ipt, qual, iphiext, iso, chrg, iphi, ieta

      // remove intermediate if required
      // index==0 and ietaext==0 are a necessary and sufficient condition
      uint32_t index = (bl->mu[i].s >> shifts::index) & masks::index;
      int32_t ietaext = ((bl->mu[i].f >> shifts::etaext) & masks::etaextv);
      if (((bl->mu[i].f >> shifts::etaext) & masks::etaexts) != 0)
        ietaext -= 256;

      if (excludeIntermediate && index == 0 && ietaext == 0)
        continue;

      //extract pt and quality and apply cut if required
      uint32_t ipt = (bl->mu[i].f >> shifts::pt) & masks::pt;
      //cuts??
      //        if((ipt-1)<ptcut) {discarded++; continue;}
      uint32_t qual = (bl->mu[i].f >> shifts::qual) & masks::qual;
      //        if(qual < qualcut) {discarded++; continue;}

      //extract integer value for extrapolated phi
      int32_t iphiext = ((bl->mu[i].f >> shifts::phiext) & masks::phiext);

      // extract iso bits and charge
      uint32_t iso = (bl->mu[i].s >> shifts::iso) & masks::iso;
      int32_t chrg = 0;
      if (((bl->mu[i].s >> shifts::chrgv) & masks::chrgv) == 1) {
        chrg = ((bl->mu[i].s >> shifts::chrg) & masks::chrg) == 1 ? -1 : 1;
      }

      // extract eta and phi at muon station
      int32_t iphi = ((bl->mu[i].s >> shifts::phi) & masks::phi);
      int32_t ieta = (bl->mu[i].s >> shifts::eta) & masks::etav;
      if (((bl->mu[i].s >> shifts::eta) & masks::etas) != 0)
        ieta -= 256;

      l1t::Muon muon(
          *dummyLVec_, ipt, ieta, iphi, qual, chrg, chrg != 0, iso, -1, 0, false, 0, 0, 0, 0, ietaext, iphiext);
      muons->push_back(bx, muon);
    }
  }
  std::cout << "end read ... " << o_test << std::endl << std::flush;
}  //unpackOrbit

std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& DataModeScoutingRun2Muon::makeDaqProvenanceHelpers() {
  //set FRD data collection
  daqProvenanceHelpers_.clear();
  daqProvenanceHelpers_.emplace_back(std::make_shared<const edm::DaqProvenanceHelper>(
      edm::TypeID(typeid(l1t::MuonBxCollection)), "l1t::MuonBxCollection", "l1tMuonBxCollection", "DAQSource"));
  return daqProvenanceHelpers_;
}

bool DataModeScoutingRun2Muon::nextEventView() {
  if (eventCached_)
    return true;
  event_ = std::make_unique<FRDEventMsgView>(dataBlockAddr_);
  if (event_->size() > dataBlockMax_) {
    throw cms::Exception("DAQSource::getNextEvent")
        << " event id:" << event_->event() << " lumi:" << event_->lumi() << " run:" << event_->run()
        << " of size:" << event_->size() << " bytes does not fit into a chunk of size:" << dataBlockMax_ << " bytes";
  }
  return true;
}

bool DataModeScoutingRun2Muon::checksumValid() { return true; }

std::string DataModeScoutingRun2Muon::getChecksumError() const { return std::string(); }

//
//2nd model: read multiple input files with different data type
//

std::pair<bool, std::vector<std::string>> DataModeScoutingRun2Multi::defineAdditionalFiles(
    std::string const& primaryName, bool fileListMode) const {
  std::vector<std::string> additionalFiles;

  auto fullpath = std::filesystem::path(primaryName);
  auto fullname = fullpath.filename();
  std::string stem = fullname.stem().string();
  std::string ext = fullname.extension().string();
  std::regex regexz("_");
  std::vector<std::string> nameTokens = {std::sregex_token_iterator(stem.begin(), stem.end(), regexz, -1),
                                         std::sregex_token_iterator()};

  if (nameTokens.size() < 3) {
    throw cms::Exception("DAQSource::getNextEvent")
        << primaryName << " name doesn't start with run#_ls#_index#_*.ext syntax";
  }

  //Can also filter out non-matching primary files (if detected by DaqDirector). false will tell source to skip the primary file.
  if (nameTokens.size() > 3 && nameTokens[3].rfind("secondary", 0) == 0)
    return std::make_pair(false, additionalFiles);

  //TODO: provisional, name syntax should be better defined

  additionalFiles.push_back(fullpath.parent_path().string() + "/" + nameTokens[0] + "_" + nameTokens[1] + "_" +
                            nameTokens[2] + "_secondary" + ext);
  //additionalFiles.push_back(fullpath.parent_path.string() + "/" + nameTokens[0] + "_" + nameTokens[1] + "_" + nameTokens[2] + "_tertiary" + ".raw");

  return std::make_pair(true, additionalFiles);
}

void DataModeScoutingRun2Multi::readEvent(edm::EventPrincipal& eventPrincipal) {
  edm::TimeValue_t time;
  timeval stv;
  gettimeofday(&stv, nullptr);
  time = stv.tv_sec;
  time = (time << 32) + stv.tv_usec;
  edm::Timestamp tstamp(time);

  std::unique_ptr<BXVector<l1t::Muon>> rawData(new BXVector<l1t::Muon>);
  //allow any bx
  rawData->setBXRange(0, 4000);

  unpackMuonOrbit(rawData.get(), (char*)events_[0]->payload(), events_[0]->eventSize());

  //TODO: implement here other object type (e.g. unpackCaloOrbit)
  //
  std::unique_ptr<BXVector<l1t::Muon>> rawDataSec(new BXVector<l1t::Muon>);
  //allow any bx
  rawDataSec->setBXRange(0, 4000);

  unpackMuonOrbit(rawDataSec.get(), (char*)events_[1]->payload(), events_[1]->eventSize());

  uint32_t hdrEventID = events_[0]->event();  //take from 1st file
  edm::EventID eventID = edm::EventID(daqSource_->eventRunNumber(), daqSource_->currentLumiSection(), hdrEventID);
  edm::EventAuxiliary aux(
      eventID, daqSource_->processGUID(), tstamp, events_[0]->isRealData(), edm::EventAuxiliary::PhysicsTrigger);

  aux.setProcessHistoryID(daqSource_->processHistoryID());
  daqSource_->makeEventWrapper(eventPrincipal, aux);

  std::unique_ptr<edm::WrapperBase> edp(new edm::Wrapper<BXVector<l1t::Muon>>(std::move(rawData)));
  eventPrincipal.put(
      daqProvenanceHelpers_[0]->branchDescription(), std::move(edp), daqProvenanceHelpers_[0]->dummyProvenance());

  //TODO: use other object and provenance helper (duplicate is just for demonstration)
  //  std::unique_ptr<edm::WrapperBase> edpSec(new edm::Wrapper<BXVector<l1t::Muon>>(std::move(rawDataSec)));
  //  eventPrincipal.put(daqProvenanceHelpers_[1]->branchDescription(), std::move(edpSec), daqProvenanceHelpers_[1]->dummyProvenance());

  eventCached_ = false;
}

void DataModeScoutingRun2Multi::unpackMuonOrbit(BXVector<l1t::Muon>* muons, char* buf, size_t len) {
  using namespace scouting;
  size_t pos = 0;
  //uint32_t o_test = 0;
  while (pos < len) {
    assert(pos + 4 <= len);
    uint32_t header = *((uint32*)(buf + pos));
    uint32_t mAcount = (header & header_masks::mAcount) >> header_shifts::mAcount;
    uint32_t mBcount = (header & header_masks::mBcount) >> header_shifts::mBcount;

    block* bl = (block*)(buf + pos + 4);

    pos += 12 + (mAcount + mBcount) * 8;
    assert(pos <= len);

    uint32_t bx = bl->bx;

    //uint32_t orbit = bl->orbit;
    //o_test = orbit;

    //should cuts should be applied
    bool excludeIntermediate = true;

    for (size_t i = 0; i < (mAcount + mBcount); i++) {
      //unpack new muon
      //variables: index, ietaext, ipt, qual, iphiext, iso, chrg, iphi, ieta

      // remove intermediate if required
      // index==0 and ietaext==0 are a necessary and sufficient condition
      uint32_t index = (bl->mu[i].s >> shifts::index) & masks::index;
      int32_t ietaext = ((bl->mu[i].f >> shifts::etaext) & masks::etaextv);
      if (((bl->mu[i].f >> shifts::etaext) & masks::etaexts) != 0)
        ietaext -= 256;

      if (excludeIntermediate && index == 0 && ietaext == 0)
        continue;

      //extract pt and quality and apply cut if required
      uint32_t ipt = (bl->mu[i].f >> shifts::pt) & masks::pt;
      //cuts??
      //        if((ipt-1)<ptcut) {discarded++; continue;}
      uint32_t qual = (bl->mu[i].f >> shifts::qual) & masks::qual;
      //        if(qual < qualcut) {discarded++; continue;}

      //extract integer value for extrapolated phi
      int32_t iphiext = ((bl->mu[i].f >> shifts::phiext) & masks::phiext);

      // extract iso bits and charge
      uint32_t iso = (bl->mu[i].s >> shifts::iso) & masks::iso;
      int32_t chrg = 0;
      if (((bl->mu[i].s >> shifts::chrgv) & masks::chrgv) == 1) {
        chrg = ((bl->mu[i].s >> shifts::chrg) & masks::chrg) == 1 ? -1 : 1;
      }

      // extract eta and phi at muon station
      int32_t iphi = ((bl->mu[i].s >> shifts::phi) & masks::phi);
      int32_t ieta = (bl->mu[i].s >> shifts::eta) & masks::etav;
      if (((bl->mu[i].s >> shifts::eta) & masks::etas) != 0)
        ieta -= 256;

      l1t::Muon muon(
          *dummyLVec_, ipt, ieta, iphi, qual, chrg, chrg != 0, iso, -1, 0, false, 0, 0, 0, 0, ietaext, iphiext);
      muons->push_back(bx, muon);
    }
  }
}  //unpackOrbit

std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& DataModeScoutingRun2Multi::makeDaqProvenanceHelpers() {
  //set FRD data collection
  daqProvenanceHelpers_.clear();
  daqProvenanceHelpers_.emplace_back(std::make_shared<const edm::DaqProvenanceHelper>(
      edm::TypeID(typeid(l1t::MuonBxCollection)), "l1t::MuonBxCollection", "l1tMuonBxCollection", "DAQSource"));
  //Note: two same kind of objects can not be put in the event from the source, so this example will be changed
  daqProvenanceHelpers_.emplace_back(std::make_shared<const edm::DaqProvenanceHelper>(
      edm::TypeID(typeid(l1t::MuonBxCollection)), "l1t::MuonBxCollection", "l1tMuonBxCollection", "DAQSource"));
  return daqProvenanceHelpers_;
}

bool DataModeScoutingRun2Multi::nextEventView() {
  blockCompleted_ = false;
  if (eventCached_)
    return true;
  for (unsigned int i = 0; i < events_.size(); i++) {
    //add last event length..
    dataBlockAddrs_[i] += events_[i]->size();
  }
  return makeEvents();
}

bool DataModeScoutingRun2Multi::makeEvents() {
  events_.clear();
  for (int i = 0; i < numFiles_; i++) {
    if (dataBlockAddrs_[i] >= dataBlockMaxAddrs_[i]) {
      blockCompleted_ = true;
      return false;
    }
    events_.emplace_back(std::make_unique<FRDEventMsgView>(dataBlockAddrs_[i]));
  }
  return true;
}

bool DataModeScoutingRun2Multi::checksumValid() { return true; }

std::string DataModeScoutingRun2Multi::getChecksumError() const { return std::string(); }
