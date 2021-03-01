/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Author:
 *   Laurent Forthomme
 *
 ****************************************************************************/

#ifndef DataFormats_TotemReco_TotemT2RecHit_h
#define DataFormats_TotemReco_TotemT2RecHit_h

class TotemT2RecHit {
public:
  explicit TotemT2RecHit() = default;

  void setLeadingEdge(unsigned short le) { lead_edge_ = le; }
  unsigned short leadingEdge() const { return lead_edge_; }
  void setTrailingEdge(unsigned short te) { trail_edge_ = te; }
  unsigned short trailingEdge() const { return trail_edge_; }

private:
  /// Leading edge time
  unsigned short lead_edge_;
  /// Trailing edge time
  unsigned short trail_edge_;
};

bool operator<(const TotemT2RecHit& lhs, const TotemT2RecHit& rhs) {
  if (lhs.leadingEdge() < rhs.leadingEdge())
    return true;
  if (lhs.leadingEdge() > rhs.leadingEdge())
    return false;
  if (lhs.trailingEdge() < rhs.trailingEdge())
    return true;
  if (lhs.trailingEdge() > rhs.trailingEdge())
    return false;
  return false;
}

#endif
