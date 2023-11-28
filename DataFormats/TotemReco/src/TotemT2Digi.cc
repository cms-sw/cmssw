#include "DataFormats/TotemReco/interface/TotemT2Digi.h"

TotemT2Digi::TotemT2Digi(
    unsigned short id, unsigned char marker, unsigned short le, unsigned short te, unsigned char status)
    : id_(id), marker_(marker), lead_edge_(le), trail_edge_(te), status_(status) {}

bool operator<(const TotemT2Digi& lhs, const TotemT2Digi& rhs) {
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
