#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.h"

#include <cassert>
#include <iostream>
#include <sstream>


PtAssignmentEngine::PtAssignmentEngine() :
    allowedModes_({3,5,9,6,10,12,7,11,13,14,15}),
    forests_(),
    ptlut_reader_(),
    version_(0xFFFFFFFF)
{

}

PtAssignmentEngine::~PtAssignmentEngine() {

}

// Called by "produce" in plugins/L1TMuonEndCapForestESProducer.cc 
// Runs over local XMLs if we are not running from the database
void PtAssignmentEngine::read(const std::string& xml_dir) {

  std::string xml_dir_full = "/afs/cern.ch/work/a/abrinke1/public/EMTF/PtAssign2017/XMLs/2017_05_10_for_emulator";

  std::cout << "EMTF emulator: attempting to read pT LUT XMLs from local directory" << std::endl;
  std::cout << xml_dir_full << std::endl;
  std::cout << "Non-standard operation; if it fails, now you know why" << std::endl;

  for (unsigned i = 0; i < allowedModes_.size(); ++i) {
    int mode = allowedModes_.at(i);
    std::stringstream ss;

    if (mode != 12)
      ss << xml_dir_full << "/f_MODE_" << mode << "_invPtTarg_invPtWgt_bitCompr_RPC_BDTG_AWB_Sq.weights";
    else
      ss << xml_dir_full << "/f_MODE_" << mode << "_invPtTarg_invPtWgt_bitCompr_RPC_BDTG_AWB.weights";

    forests_.at(mode).loadForestFromXML(ss.str().c_str(), 400);
  }

  return;
}

void PtAssignmentEngine::load(const L1TMuonEndCapForest *payload) {
  unsigned pt_lut_version = payload->version_;
  if (version_ == pt_lut_version)  return;

  for (unsigned i = 0; i < allowedModes_.size(); ++i) {
    int mode = allowedModes_.at(i);

    L1TMuonEndCapForest::DForestMap::const_iterator index = payload->forest_map_.find(mode); // associates mode to index
    if (index == payload->forest_map_.end())  continue;

    forests_.at(mode).loadFromCondPayload(payload->forest_coll_[index->second]);
    
    // boostWeight set in plugins/L1TMuonEndCapForestESProducer.cc
    // For 2017 only - should make configurable based on "Era" - AWB 22.05.17
    double boostWeight_ = payload->forest_map_.find(mode+16)->second / 1000000.;  
    forests_.at(mode).getTree(0)->setBoostWeight( boostWeight_ );

    // // Code below can be used to save out trees in XML format
    // for (int t = 0; t < 64; t++) {
    //   emtf::Tree* tree = forests_.at(mode).getTree(t);
    //   std::stringstream ss;
    //   ss << mode << "/" << t << ".xml";
    //   tree->saveToXML( ss.str().c_str() );
    // }

  }

  version_ = pt_lut_version;
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
    ss << "/afs/cern.ch/work/a/abrinke1/public/EMTF/PtAssign2017/LUTs/2017_05_24/LUT_v6_24May17.dat";
    std::string lut_full_path = ss.str();

    std::cout << "EMTF emulator: attempting to read pT LUT binary file from local area" << std::endl;
    std::cout << lut_full_path << std::endl;
    std::cout << "Non-standard operation; if it fails, now you know why" << std::endl;

    ptlut_reader_.read(lut_full_path);
  }
}

const PtAssignmentEngineAux& PtAssignmentEngine::aux() const {
  static const PtAssignmentEngineAux instance;
  return instance;
}

float PtAssignmentEngine::scale_pt(const float pt, const int mode) const {
  return scale_pt(pt, mode);
}

float PtAssignmentEngine::unscale_pt(const float pt, const int mode) const {
  return unscale_pt(pt, mode);
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

