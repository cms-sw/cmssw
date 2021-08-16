#ifndef L1Trigger_L1TMuonOverlap_OmtfName_H
#define L1Trigger_L1TMuonOverlap_OmtfName_H

#include <string>
#include <ostream>

#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

class OmtfName {
public:
  enum Board {
    OMTFn1 = -1,
    OMTFn2 = -2,
    OMTFn3 = -3,
    OMTFn4 = -4,
    OMTFn5 = -5,
    OMTFn6 = -6,
    OMTFp1 = 1,
    OMTFp2 = 2,
    OMTFp3 = 3,
    OMTFp4 = 4,
    OMTFp5 = 5,
    OMTFp6 = 6,
    UNKNOWN = 0
  };

public:
  OmtfName(Board board = UNKNOWN) : theBoard(board) {}

  OmtfName(const std::string& name);

  //by giving procesor id [0,5] and endcap position {+1,-1} as in uGMT.
  explicit OmtfName(unsigned int iProcesor, int endcap);

  //by giving procesor id [0,5] and endcap position as l1t::tftype of omtf_pos or omtf_neg.
  explicit OmtfName(unsigned int iProcesor, l1t::tftype endcap);

  //by giving procesor continous index [0,11].
  explicit OmtfName(unsigned int iProcesor);

  operator int() const { return theBoard; }
  bool operator==(const OmtfName& o) const { return theBoard == o.theBoard; }
  bool operator!=(const OmtfName& o) const { return !(*this == o); }

  unsigned int processor() const;
  int position() const;
  l1t::tftype tftype() const;

  std::string name() const;

private:
  Board theBoard;

  friend std::ostream& operator<<(std::ostream& out, const OmtfName& n) {
    out << n.name();
    return out;
  }
};
#endif
