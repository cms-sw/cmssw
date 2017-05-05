#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.hh"

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

void PtAssignmentEngine::read(const std::string& xml_dir) {
  //std::string xml_dir_full = "L1Trigger/L1TMuon/data/emtf_luts/" + xml_dir + "/ModeVariables/trees";
  std::string xml_dir_full = "L1Trigger/L1TMuonEndCap/data/emtf_luts/" + xml_dir + "/ModeVariables/trees";

  for (unsigned i = 0; i < allowedModes_.size(); ++i) {
    int mode_inv = allowedModes_.at(i);  // inverted mode because reasons
    std::stringstream ss;
    ss << xml_dir_full << "/" << mode_inv;
    forests_.at(mode_inv).loadForestFromXML(ss.str().c_str(), 64);
  }
  return;
}

void PtAssignmentEngine::load(const L1TMuonEndCapForest *payload) {
  unsigned pt_lut_version = payload->version_;
  if (version_ == pt_lut_version)  return;

  for (unsigned i = 0; i < allowedModes_.size(); ++i) {
    int mode_inv = allowedModes_.at(i);

    L1TMuonEndCapForest::DForestMap::const_iterator index = payload->forest_map_.find(mode_inv); // associates mode to index
    if (index == payload->forest_map_.end())  continue;

    forests_.at(mode_inv).loadFromCondPayload(payload->forest_coll_[index->second]);

    //for(int t=0; t<64; t++){
    //  emtf::Tree* tree = forests_.at(mode_inv).getTree(t);
    //  std::stringstream ss;
    //  ss << mode_inv << "/" << t << ".xml";
    //  tree->saveToXML( ss.str().c_str() );
    //}
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
    ss << std::getenv("CMSSW_BASE") << "/" << "src/L1Trigger/L1TMuonEndCap/data/emtf_luts/v_16_02_21_ptlut/LUT_AndrewFix_25July16.dat";  // hardcoded, it does not exist in CMSSW
    std::string lut_full_path = ss.str();

    ptlut_reader_.read(lut_full_path);
  }
}

const PtAssignmentEngineAux& PtAssignmentEngine::aux() const {
  static const PtAssignmentEngineAux instance;
  return instance;
}

float PtAssignmentEngine::calculate_pt(const address_t& address) {
  float pt = 0.;

  if (readPtLUTFile_) {
    pt = calculate_pt_lut(address);
  } else {
    pt = calculate_pt_xml(address);
  }

  return pt;
}

float PtAssignmentEngine::calculate_pt_lut(const address_t& address) {
  // LUT outputs 'gmt_pt', so need to convert back to 'xmlpt'
  int gmt_pt = ptlut_reader_.lookup(address);
  float pt = aux().getPtFromGMTPt(gmt_pt);
  float xmlpt = pt;
  xmlpt /= 1.4;

  return xmlpt;
}

