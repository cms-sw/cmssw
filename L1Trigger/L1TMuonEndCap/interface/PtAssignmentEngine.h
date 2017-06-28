#ifndef L1TMuonEndCap_PtAssignmentEngine_h
#define L1TMuonEndCap_PtAssignmentEngine_h

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <array>

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtLUTReader.h"
#include "L1Trigger/L1TMuonEndCap/interface/bdt/Forest.h"


class PtAssignmentEngine {
public:
  explicit PtAssignmentEngine();
  virtual ~PtAssignmentEngine();

  typedef uint64_t address_t;

  void read(const std::string& xml_dir);
  void load(const L1TMuonEndCapForest *payload);
  const std::array<emtf::Forest, 16>& getForests(void) const { return forests_; }
  const std::vector<int>& getAllowedModes(void) const { return allowedModes_; }

  void configure(
      int verbose,
      int ptLUTVersion, bool readPtLUTFile, bool fixMode15HighPt,
      bool bug9BitDPhi, bool bugMode7CLCT, bool bugNegPt
  );

  void configure_details();

  const PtAssignmentEngineAux& aux() const;

  virtual float scale_pt  (const float pt, const int mode = 15) const;
  virtual float unscale_pt(const float pt, const int mode = 15) const;

  virtual address_t calculate_address(const EMTFTrack& track) const { return 0; }

  virtual float calculate_pt(const address_t& address) const;
  virtual float calculate_pt(const EMTFTrack& track) const;

  virtual float calculate_pt_lut(const address_t& address) const;
  virtual float calculate_pt_xml(const address_t& address) const { return 0.; }
  virtual float calculate_pt_xml(const EMTFTrack& track) const { return 0.; }

  inline int  readPtLUTFile() { return readPtLUTFile_; }
  inline void set_ptLUTVersion( int ptLUTVersion ) { ptLUTVersion_ = ptLUTVersion; }

protected:
  std::vector<int> allowedModes_;
  std::array<emtf::Forest, 16> forests_;
  PtLUTReader ptlut_reader_;

  int verbose_;

  int ptLUTVersion_;
  bool readPtLUTFile_, fixMode15HighPt_;
  bool bug9BitDPhi_, bugMode7CLCT_, bugNegPt_;
};

#endif
