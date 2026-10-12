#ifndef CondFormats_SiPixelObjects_SiPixelClusterShapeLimits_h
#define CondFormats_SiPixelObjects_SiPixelClusterShapeLimits_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <iosfwd>
#include <string>
#include <vector>

/*
 * Pixel cut windows of the cluster shape filter (RecoTracker/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h).
 *
 * For each measured pixel cluster size, the filter accepts a hit if the size predicted from the track direction
 * lies inside one of two windows.
 *
 * The payload holds any number of tables, and rules that assign each pixel module to one table.
 *
 * Table: one entry per measured cluster size (dx, dy); part = 0 for BPix entries, part = 1 for FPix entries.
 *   A module in BPix uses the part = 0 entries of its table, a module in FPix the part = 1 entries.
 *   Each entry consists of 8 numbers (limits), the allowed range of the predicted cluster size (in pixels)
 *   in two alternative windows; a hit passes if its prediction is inside window 1 OR window 2:
 *   {w1 x min, w1 x max, w1 y min, w1 y max, w2 x min, w2 x max, w2 y min, w2 y max}
 *
 * Rules: an ordered list; a module uses the table of the FIRST rule that matches it.
 *   A rule matches a module if the sub-detector is the same and layerOrDisk is either kAny or equal to the
 *   BPix layer or FPix disk of the module.
 *   Each sub-detector must have a catch-all rule (layerOrDisk = kAny), so that every module has a table.
 */
class SiPixelClusterShapeLimits {
public:
  // number of limits per entry: a hit passes if its predicted size is inside window 1 OR window 2
  //   0, 1  window 1: x min, x max
  //   2, 3  window 1: y min, y max
  //   4, 5  window 2: x min, x max
  //   6, 7  window 2: y min, y max
  static constexpr unsigned int kNLimits = 8;
  static constexpr int kAny = 0;
  // same values as PixelSubdetector::PixelBarrel and PixelSubdetector::PixelEndcap
  static constexpr int kBPix = 1;
  static constexpr int kFPix = 2;

  struct Entry {
    int part;
    int dx;
    int dy;
    std::vector<float> limits;  // kNLimits values

    COND_SERIALIZABLE;
  };

  struct Table {
    std::string name;  // for bookkeeping only, e.g. the name of the file it was made from
    std::vector<Entry> entries;

    COND_SERIALIZABLE;
  };

  struct Rule {
    int subdet;          // kBPix or kFPix
    int layerOrDisk;     // BPix layer or FPix disk, kAny for any
    unsigned int table;  // index in tables()

    COND_SERIALIZABLE;
  };

  SiPixelClusterShapeLimits() = default;
  ~SiPixelClusterShapeLimits() = default;

  // add a table and return its index; throws if an entry does not have kNLimits limits
  unsigned int addTable(const std::string& name, const std::vector<Entry>& entries);
  // add a rule at the end of the list; throws if it is not valid or if it can never match
  // (i.e. it comes after the catch-all rule of its sub-detector)
  void addRule(const Rule& rule);
  // throws if BPix or FPix has no catch-all rule (a rule for any layer or disk),
  // i.e. if some modules could be left without cluster shape limits
  void checkComplete() const;

  const std::vector<Table>& tables() const { return m_tables; }
  const std::vector<Rule>& rules() const { return m_rules; }

  // index of the table to use for a module, i.e. the table of the first matching rule; -1 if no rule matches
  int tableIndex(int subdet, int layerOrDisk) const;

  void printAll(std::ostream& out) const;

private:
  std::vector<Table> m_tables;
  std::vector<Rule> m_rules;

  COND_SERIALIZABLE;
};

#endif
