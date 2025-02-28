#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPartition.h"

using namespace l1t::me0;

bool l1t::me0::isGhost(const ME0StubPrimitive& seg, const ME0StubPrimitive& comp, bool checkIds, bool checkStrips) {
  /*
    takes in a segment and a list of segments to ensure that there aren't
    copies of the same data (ID value identical) or mirrors (ID value + 2 or - 2
    from each other)
    */

  bool ghost = (seg.quality() < comp.quality()) && (!checkStrips || (std::abs(seg.strip() - comp.strip()) < 2)) &&
               (!checkIds || ((seg.patternId() == comp.patternId()) || (seg.patternId() + 2 == comp.patternId()) ||
                              (seg.patternId() == comp.patternId() + 2)));
  return ghost;
}

bool l1t::me0::isAtEdge(int x, int groupWidth, int edgeDistance) {
  if (groupWidth > 0) {
    return ((x % groupWidth) < edgeDistance) || ((x % groupWidth) >= (groupWidth - edgeDistance));
  } else {
    return true;
  }
};

std::vector<ME0StubPrimitive> l1t::me0::cancelEdges(
    const std::vector<ME0StubPrimitive>& segments, int groupWidth, int ghostWidth, int edgeDistance, bool verbose) {
  /*
    Takes in a list of SEGMENTS and is designed to perform ghost
    cancellation on the edges of the "groups".

    during segment sorting, an initial step is that the partition is chunked into
    groups of width GROUP_WIDTH. The firmware selects just one segment from each group.

    Since segments can ghost (produce duplicates of the same segment on
    different strips), we do a ghost cancellation before this group filtering process.

    This is done by looking at the edges of the group and zeroing off a segment
    if it is of lower quality than its neighbors. Segments away from the edges
    of the group will not need to be de-duplicated since this is handled by the
    group filtering process itself. This is only needed to prevent duplicates
    from appearing in different groups.

    An implementation that cancels after the filtering is much simpler and less
    resource intensive but comes with the downside that we may lose segments.

    ghost_width = 0 means do not compare
    ghost_width = 1 means look 1 to the left and right
    ghost_width = 2 means look 2 to the left and right

    edge_distance specifies which strips will have ghost cancellation done on them
    edge_distance = 0 means to only look at strips directly on the edges (0 7 8 15 etc)
    edge_distance = 1 means to look one away from the edge (0 1 6 7 8 9 14 15 16 17 etc)

    etc
    */

  std::vector<ME0StubPrimitive> canceledSegments = segments;
  std::vector<int> comps;

  bool ghost;
  for (int i = 0; i < static_cast<int>(segments.size()); ++i) {
    if (isAtEdge(i, groupWidth, edgeDistance)) {
      for (int x = i - ghostWidth; x < i; ++x) {
        if (x >= 0) {
          comps.push_back(x);
        }
      }
      for (int x = i + 1; x < i + ghostWidth + 1; ++x) {
        if (x < static_cast<int>(segments.size())) {
          comps.push_back(x);
        }
      }

      for (int comp : comps) {
        ghost = isGhost(segments[i], segments[comp]);
        if (ghost) {
          canceledSegments[i].reset();
        }
      }
      comps.clear();
    }
  }

  return canceledSegments;
}

std::vector<ME0StubPrimitive> l1t::me0::processPartition(const std::vector<UInt192>& partitionData,
                                                         const std::vector<std::vector<int>>& partitionBxData,
                                                         int partition,
                                                         Config& config) {
  /*
    takes in partition data, a group size, and a ghost width to return a
    smaller data set, using ghost edge cancellation and segment quality
    filtering

    NOTE: ghost width denotes the width where we can likely see copies of the
    same segment in the data

    steps: process partition data with patMux, perform edge cancellations,
    divide partition into pieces, take best segment from each piece
    */
  std::vector<ME0StubPrimitive> tmp;
  std::vector<ME0StubPrimitive> out;
  std::vector<ME0StubPrimitive> maxSegs;
  const std::vector<ME0StubPrimitive> segments = patMux(partitionData, partitionBxData, partition, config);

  if (config.deghostPre) {
    tmp = cancelEdges(segments, config.groupWidth, config.ghostWidth, config.edgeDistance);
  } else {
    tmp = segments;
  }

  std::vector<std::vector<ME0StubPrimitive>> chunked = chunk(tmp, config.groupWidth);
  for (const std::vector<ME0StubPrimitive>& segV : chunked) {
    ME0StubPrimitive maxSeg = ME0StubPrimitive();
    for (const ME0StubPrimitive& seg : segV) {
      if (maxSeg.quality() < seg.quality()) {
        maxSeg = seg;
      }
    }
    maxSegs.push_back(maxSeg);
  }

  if (config.deghostPost) {
    out = cancelEdges(maxSegs, 0, 1, 1);
  } else {
    out = maxSegs;
  }

  return out;
}
