#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPartition.h"

using namespace l1t::me0;

bool l1t::me0::is_ghost(const ME0StubPrimitive& seg, const ME0StubPrimitive& comp, bool check_ids, bool check_strips) {
  /*
    takes in a segment and a list of segments to ensure that there aren't
    copies of the same data (ID value identical) or mirrors (ID value + 2 or - 2
    from each other)
    */

  bool ghost = (seg.Quality() < comp.Quality()) && (!check_strips || (std::abs(seg.Strip() - comp.Strip()) < 2)) &&
               (!check_ids || ((seg.PatternId() == comp.PatternId()) || (seg.PatternId() + 2 == comp.PatternId()) ||
                               (seg.PatternId() == comp.PatternId() + 2)));
  return ghost;
}

bool l1t::me0::is_at_edge(int x, int group_width, int edge_distance) {
  if (group_width > 0) {
    return ((x % group_width) < edge_distance) || ((x % group_width) >= (group_width - edge_distance));
  } else {
    return true;
  }
};

std::vector<ME0StubPrimitive> l1t::me0::cancel_edges(
    const std::vector<ME0StubPrimitive>& segments, int group_width, int ghost_width, int edge_distance, bool verbose) {
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

  std::vector<ME0StubPrimitive> canceled_segements = segments;
  std::vector<int> comps;

  bool ghost;
  for (int i = 0; i < static_cast<int>(segments.size()); ++i) {
    if (is_at_edge(i, group_width, edge_distance)) {
      for (int x = i - ghost_width; x < i; ++x) {
        if (x >= 0) {
          comps.push_back(x);
        }
      }
      for (int x = i + 1; x < i + ghost_width + 1; ++x) {
        if (x < static_cast<int>(segments.size())) {
          comps.push_back(x);
        }
      }

      for (int comp : comps) {
        ghost = is_ghost(segments[i], segments[comp]);
        if (ghost) {
          // canceled_segements[i] = ME0StubPrimitive();
          canceled_segements[i].reset();
        }
      }
      comps.clear();
    }
  }

  return canceled_segements;
}

std::vector<ME0StubPrimitive> l1t::me0::process_partition(const std::vector<UInt192>& partition_data,
                                                          const std::vector<std::vector<int>>& partition_bx_data,
                                                          int partition,
                                                          Config& config) {
  /*
    takes in partition data, a group size, and a ghost width to return a
    smaller data set, using ghost edge cancellation and segment quality
    filtering

    NOTE: ghost width denotes the width where we can likely see copies of the
    same segment in the data

    steps: process partition data with pat_mux, perfom edge cancellations,
    divide partition into pieces, take best segment from each piece
    */
  std::vector<ME0StubPrimitive> tmp;
  std::vector<ME0StubPrimitive> out;
  std::vector<ME0StubPrimitive> max_segs;
  const std::vector<ME0StubPrimitive> segments = pat_mux(partition_data, partition_bx_data, partition, config);

  if (config.deghost_pre) {
    tmp = cancel_edges(segments, config.group_width, config.ghost_width, config.edge_distance);
  } else {
    tmp = segments;
  }

  std::vector<std::vector<ME0StubPrimitive>> chunked = chunk(tmp, config.group_width);
  for (const std::vector<ME0StubPrimitive>& seg_v : chunked) {
    ME0StubPrimitive max_seg = ME0StubPrimitive();
    for (const ME0StubPrimitive& seg : seg_v) {
      if (max_seg.Quality() < seg.Quality()) {
        max_seg = seg;
      }
    }
    max_segs.push_back(max_seg);
  }

  if (config.deghost_post) {
    out = cancel_edges(max_segs, 0, 1, 1);
  } else {
    out = max_segs;
  }

  return out;
}
