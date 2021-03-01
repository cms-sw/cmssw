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
  explicit TotemT2Digi() = default;
  TotemT2Digi(unsigned char geo, unsigned char id, unsigned char marker, unsigned short le, unsigned short te);

  void setLeadingEdge(unsigned short le) { lead_edge_ = le; }
  unsigned short leadingEdge() const { return lead_edge_; }
  void setTrailingEdge(unsigned short te) { trail_edge_ = te; }
  unsigned short trailingEdge() const { return trail_edge_; }

private:
  /// Geo ID
  unsigned char geo_id_;
  /// Channel ID
  unsigned char channel_id_;
  /// Channel marker
  unsigned char marker_;
  /// Leading edge time
  unsigned short lead_edge_;
  /// Trailing edge time
  unsigned short trail_edge_;
};

bool operator<(const TotemT2Digi& lhs, const TotemT2Digi& rhs);

#endif
