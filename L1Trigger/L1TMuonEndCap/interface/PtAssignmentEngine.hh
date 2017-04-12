#ifndef L1TMuonEndCap_PtAssignmentEngine_hh
#define L1TMuonEndCap_PtAssignmentEngine_hh

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <array>

#include "L1Trigger/L1TMuonEndCap/interface/Common.hh"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux.hh"
#include "L1Trigger/L1TMuonEndCap/interface/PtLUTReader.hh"
#include "L1Trigger/L1TMuonEndCap/interface/bdt/Forest.h"


class PtAssignmentEngine {
public:
  explicit PtAssignmentEngine();
  ~PtAssignmentEngine();

  typedef uint64_t address_t;

  void read(const std::string& xml_dir);

  void configure(
      int verbose,
      bool readPtLUTFile, bool fixMode15HighPt,
      bool bug9BitDPhi, bool bugMode7CLCT, bool bugNegPt
  );

  void configure_details();

  const PtAssignmentEngineAux& aux() const;

  address_t calculate_address(const EMTFTrack& track) const;

  float calculate_pt(const address_t& address);

  float calculate_pt_lut(const address_t& address);
  float calculate_pt_xml(const address_t& address);

private:
  std::vector<int> allowedModes_;
  std::array<emtf::Forest, 16> forests_;
  PtLUTReader ptlut_reader_;

  bool ok_;

  int verbose_;

  bool readPtLUTFile_, fixMode15HighPt_;
  bool bug9BitDPhi_, bugMode7CLCT_, bugNegPt_;
};

#endif
