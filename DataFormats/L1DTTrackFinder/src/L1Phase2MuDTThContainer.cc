#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTThContainer.h"

L1Phase2MuDTThContainer::L1Phase2MuDTThContainer() {}

void L1Phase2MuDTThContainer::setContainer(const Segment_Container& inputSegments) { m_segments = inputSegments; }

L1Phase2MuDTThContainer::Segment_Container const* L1Phase2MuDTThContainer::getContainer() const { return &m_segments; }
