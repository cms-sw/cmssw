#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.h"

#include <cassert>
#include <iostream>
#include <sstream>

#include "helper.h"  // assert_no_abort


PtAssignmentEngine::PtAssignmentEngine() :
    allowedModes_({3,5,9,6,10,12,7,11,13,14,15}),
    forests_(),
    ptlut_reader_(),
    ptLUTVersion_(0xFFFFFFFF)
{

}

PtAssignmentEngine::~PtAssignmentEngine() {

}

// Called by "produce" in plugins/L1TMuonEndCapForestESProducer.cc
// Runs over local XMLs if we are not running from the database
// void PtAssignmentEngine::read(const std::string& xml_dir, const unsigned xml_nTrees) {
void PtAssignmentEngine::read(int pt_lut_version, const std::string& xml_dir) {

  std::string xml_dir_full = "L1Trigger/L1TMuonEndCap/data/pt_xmls/" + xml_dir;
  unsigned xml_nTrees = 64; // 2016 XMLs
  if (pt_lut_version >= 6)
    xml_nTrees = 400;       // First 2017 XMLs

  std::cout << "EMTF emulator: attempting to read " << xml_nTrees << " pT LUT XMLs from local directory" << std::endl;
  std::cout << xml_dir_full << std::endl;
  std::cout << "Non-standard operation; if it fails, now you know why" << std::endl;

  for (unsigned i = 0; i < allowedModes_.size(); ++i) {
    int mode = allowedModes_.at(i); // For 2016, maps to "mode_inv"
    std::stringstream ss;
    ss << xml_dir_full << "/" << mode;
    forests_.at(mode).loadForestFromXML(ss.str().c_str(), xml_nTrees);
  }

  return;
}

void PtAssignmentEngine::load(int pt_lut_version, const L1TMuonEndCapForest *payload) {
  if (ptLUTVersion_ == pt_lut_version)  return;
  ptLUTVersion_ = pt_lut_version;

  edm::LogInfo("L1T") << "EMTF using pt_lut_ver: " << pt_lut_version;

  for (unsigned i = 0; i < allowedModes_.size(); ++i) {
    int mode = allowedModes_.at(i);

    L1TMuonEndCapForest::DForestMap::const_iterator index = payload->forest_map_.find(mode); // associates mode to index
    if (index == payload->forest_map_.end())  continue;

    forests_.at(mode).loadFromCondPayload(payload->forest_coll_[index->second]);

    double boostWeight_ = payload->forest_map_.find(mode+16)->second / 1000000.;
    // std::cout << "Loaded forest for mode " << mode << " with boostWeight_ = " << boostWeight_ << std::endl;
    // std::cout << "  * ptLUTVersion_ = " << ptLUTVersion_ << std::endl;
    forests_.at(mode).getTree(0)->setBoostWeight( boostWeight_ );

    if (not(boostWeight_ == 0 || ptLUTVersion_ >= 6))  // Check that XMLs and pT LUT version are consistent
      { edm::LogError("L1T") << "boostWeight_ = " << boostWeight_ << ", ptLUTVersion_ = " << ptLUTVersion_; return; }
    // Will catch user trying to run with Global Tag settings on 2017 data, rather than fakeEmtfParams. - AWB 08.06.17

    // // Code below can be used to save out trees in XML format
    // for (int t = 0; t < 64; t++) {
    //   emtf::Tree* tree = forests_.at(mode).getTree(t);
    //   std::stringstream ss;
    //   ss << mode << "/" << t << ".xml";
    //   tree->saveToXML( ss.str().c_str() );
    // }

  }

  return;
}

void PtAssignmentEngine::configure(
    int verbose,
    bool readPtLUTFile, bool fixMode15HighPt,
    bool bug9BitDPhi, bool bugMode7CLCT, bool bugNegPt
) {
  verbose_ = verbose;

  readPtLUTFile_   = readPtLUTFile;
  fixMode15HighPt_ = fixMode15HighPt;
  bug9BitDPhi_     = bug9BitDPhi;
  bugMode7CLCT_    = bugMode7CLCT;
  bugNegPt_        = bugNegPt;

  configure_details();
}

void PtAssignmentEngine::configure_details() {
  if (readPtLUTFile_) {
    std::stringstream ss;
    // Hardcoded - this 2 GB LUT file does not exist in CMSSW
    ss << "/afs/cern.ch/work/a/abrinke1/public/EMTF/PtAssign2017/LUTs/2017_06_07/LUT_v07_07June17.dat";
    std::string lut_full_path = ss.str();
    ptlut_reader_.read(lut_full_path);
  }
}

const PtAssignmentEngineAux& PtAssignmentEngine::aux() const {
  static const PtAssignmentEngineAux instance;
  return instance;
}

float PtAssignmentEngine::calculate_pt(const address_t& address) const {
  float pt = 0.;

  if (readPtLUTFile_) {
    pt = calculate_pt_lut(address);
  } else {
    pt = calculate_pt_xml(address);
  }

  return pt;
}

float PtAssignmentEngine::calculate_pt(const EMTFTrack& track) const {
  float pt = 0.;

  pt = calculate_pt_xml(track);

  return pt;
}

float PtAssignmentEngine::calculate_pt_lut(const address_t& address) const {
  // LUT outputs 'gmt_pt', so need to convert back to 'xmlpt'
  int gmt_pt = ptlut_reader_.lookup(address);
  float pt = aux().getPtFromGMTPt(gmt_pt);
  float xmlpt = pt;
  xmlpt *= unscale_pt(pt);

  return xmlpt;
}

