/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Author:
 *   Laurent Forthomme
 *
 ****************************************************************************/

#ifndef DataFormats_TotemReco_TotemT2Digi_h
#define DataFormats_TotemReco_TotemT2Digi_h

class TotemT2Digi {
public:
  TotemT2Digi() = default;
  TotemT2Digi(unsigned short id, unsigned char marker, unsigned short le, unsigned short te, unsigned char status);

  void setLeadingEdge(unsigned short le) { lead_edge_ = le; }
  unsigned short leadingEdge() const { return lead_edge_; }
  void setTrailingEdge(unsigned short te) { trail_edge_ = te; }
  unsigned short trailingEdge() const { return trail_edge_; }
  unsigned char status() const { return status_ & 0xF; }
  bool hasLE() const { return marker_ & 0x1; }
  bool hasTE() const { return marker_ & 0x2; }
  bool hasManyLE() const { return marker_ & 0x4; }
  bool hasManyTE() const { return marker_ & 0x8; }

private:
  /// New HW ID in ver 2.2
  unsigned short id_{0};
  /// Channel marker
  unsigned char marker_{0};
  /// Leading edge time
  unsigned short lead_edge_{0};
  /// Trailing edge time
  unsigned short trail_edge_{0};
  /// Header status flags
  unsigned char status_{0};
};

bool operator<(const TotemT2Digi& lhs, const TotemT2Digi& rhs);

#endif
