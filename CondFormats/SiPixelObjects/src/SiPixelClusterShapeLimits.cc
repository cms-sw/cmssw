#include "CondFormats/SiPixelObjects/interface/SiPixelClusterShapeLimits.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <ostream>

static_assert(SiPixelClusterShapeLimits::kBPix == PixelSubdetector::PixelBarrel);
static_assert(SiPixelClusterShapeLimits::kFPix == PixelSubdetector::PixelEndcap);

namespace {
  bool matches(int ruleValue, int value) { return ruleValue == SiPixelClusterShapeLimits::kAny || ruleValue == value; }

  bool isCatchAll(const SiPixelClusterShapeLimits::Rule& rule) {
    return rule.layerOrDisk == SiPixelClusterShapeLimits::kAny;
  }

  const char* subdetName(int subdet) { return subdet == SiPixelClusterShapeLimits::kBPix ? "BPix" : "FPix"; }
}  // namespace

unsigned int SiPixelClusterShapeLimits::addTable(const std::string& name, const std::vector<Entry>& entries) {
  for (const auto& entry : entries)
    if (entry.limits.size() != kNLimits)
      throw cms::Exception("SiPixelClusterShapeLimits") << "wrong number of limits for an entry of table '" << name
                                                        << "': " << entry.limits.size() << " instead of " << kNLimits;

  m_tables.push_back({name, entries});
  return m_tables.size() - 1;
}

void SiPixelClusterShapeLimits::addRule(const Rule& rule) {
  if (rule.subdet != kBPix && rule.subdet != kFPix)
    throw cms::Exception("SiPixelClusterShapeLimits") << "unknown sub-detector " << rule.subdet << " in a rule";
  if (rule.table >= m_tables.size())
    throw cms::Exception("SiPixelClusterShapeLimits")
        << "rule for " << subdetName(rule.subdet) << " points to table " << rule.table << ", but there are only "
        << m_tables.size() << " tables";
  for (const auto& previous : m_rules)
    if (previous.subdet == rule.subdet && isCatchAll(previous))
      throw cms::Exception("SiPixelClusterShapeLimits")
          << "rule for " << subdetName(rule.subdet) << " added after the catch-all rule of " << subdetName(rule.subdet)
          << ": it would never be used";

  m_rules.push_back(rule);
}

void SiPixelClusterShapeLimits::checkComplete() const {
  for (int subdet : {kBPix, kFPix}) {
    const bool found = std::any_of(m_rules.begin(), m_rules.end(), [subdet](const Rule& rule) {
      return rule.subdet == subdet && isCatchAll(rule);
    });
    if (!found)
      throw cms::Exception("SiPixelClusterShapeLimits") << "no catch-all rule for " << subdetName(subdet);
  }
}

int SiPixelClusterShapeLimits::tableIndex(int subdet, int layerOrDisk) const {
  for (const auto& rule : m_rules)
    if (rule.subdet == subdet && matches(rule.layerOrDisk, layerOrDisk))
      return rule.table;
  return -1;
}

void SiPixelClusterShapeLimits::printAll(std::ostream& out) const {
  out << "# rules (first match wins): subdet layerOrDisk -> table (0 = any)\n";
  for (const auto& rule : m_rules)
    out << "# " << subdetName(rule.subdet) << " " << rule.layerOrDisk << " -> " << rule.table << " ("
        << m_tables[rule.table].name << ")\n";

  for (unsigned int i = 0; i < m_tables.size(); ++i) {
    out << "# table " << i << " '" << m_tables[i].name << "' (" << m_tables[i].entries.size()
        << " entries): part dx dy limits[8]\n";
    for (const auto& entry : m_tables[i].entries) {
      out << entry.part << " " << entry.dx << " " << entry.dy;
      for (float limit : entry.limits)
        out << " " << limit;
      out << "\n";
    }
  }
}
