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
  // std::string xml_dir_full = "L1Trigger/L1TMuon/data/emtf_luts/" + xml_dir + "/ModeVariables/trees";
  // std::string xml_dir_full = "L1Trigger/L1TMuonEndCap/data/emtf_luts/" + xml_dir + "/ModeVariables/trees";
  // std::string xml_dir_full = "/afs/cern.ch/work/a/abrinke1/public/EMTF/PtAssign2017/XMLs/2017_05_08_for_emulator";
  std::string xml_dir_full = "/afs/cern.ch/work/a/abrinke1/public/EMTF/PtAssign2017/XMLs/2017_05_10_for_emulator";

  for (unsigned i = 0; i < allowedModes_.size(); ++i) {
    int mode_inv = allowedModes_.at(i);  // inverted mode because reasons (Change for 2017? - AWB 20.05.17)
    std::stringstream ss;

    // ss << xml_dir_full << "/" << mode_inv;
    // forests_.at(mode_inv).loadForestFromXML(ss.str().c_str(), 64);

    // if (mode_inv > 12 || mode_inv == 11 || mode_inv == 7)
    //   ss << xml_dir_full << "/f_MODE_" << mode_inv << "_logPtTarg_invPtWgt_bitCompr_RPC";
    // else
    //   ss << xml_dir_full << "/f_MODE_" << mode_inv << "_logPtTarg_invPtWgt_bitCompr_noRPC";

    if (mode_inv != 12)
      ss << xml_dir_full << "/f_MODE_" << mode_inv << "_invPtTarg_invPtWgt_bitCompr_RPC_BDTG_AWB_Sq.weights";
    else
      ss << xml_dir_full << "/f_MODE_" << mode_inv << "_invPtTarg_invPtWgt_bitCompr_RPC_BDTG_AWB.weights";

    forests_.at(mode_inv).loadForestFromXML(ss.str().c_str(), 400);

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
    
    // boostWeight set in plugins/L1TMuonEndCapForestESProducer.cc
    // For 2017 only - should make configurable based on "Era" - AWB 22.05.17
    double boostWeight_ = payload->forest_map_.find(mode_inv+16)->second / 1000000.;  
    forests_.at(mode_inv).getTree(0)->setBoostWeight( boostWeight_ );

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
  xmlpt /= 1.4;

  return xmlpt;
}

