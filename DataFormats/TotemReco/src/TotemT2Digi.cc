#include "DataFormats/TotemReco/interface/TotemT2Digi.h"

TotemT2Digi::TotemT2Digi(unsigned char geo, unsigned char id, unsigned char marker, unsigned short le, unsigned short te)
    : geo_id_(geo), channel_id_(id), marker_(marker), lead_edge_(le), trail_edge_(te) {}

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
