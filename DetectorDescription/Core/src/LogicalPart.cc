#include <algorithm>
#include <memory>

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DetectorDescription/Core/interface/LogicalPart.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using DDI::LogicalPart;

LogicalPart::LogicalPart(const DDMaterial& m, const DDSolid& s, DDEnums::Category c)
    : material_(m), solid_(s), cat_(c), specifics_(0), hasDDValue_(1, false) {}

const DDMaterial& LogicalPart::material() const { return material_; }

const DDSolid& LogicalPart::solid() const { return solid_; }

DDEnums::Category LogicalPart::category() const { return cat_; }

void LogicalPart::stream(std::ostream& os) {
  os << std::endl << " mat=" << material().ddname() << std::endl << " solid=" << solid();
}

void LogicalPart::addSpecifics(const std::pair<const DDPartSelection*, const DDsvalues_type*>& s) {
  if (!(s.first && s.second)) {
    // FIXME
    std::cerr << "LogicalPart::addSpecific error pointer 0 " << s.first << "," << s.second << std::endl;
    return;
  }
  specifics_.emplace_back(s);
  for (const auto& it : *s.second) {
    unsigned int id = it.first;
    if (id < hasDDValue_.size()) {
      hasDDValue_[id] = true;
    } else {
      hasDDValue_.resize(id + 1, false);
      hasDDValue_[id] = true;
    }
  }
}

bool LogicalPart::hasDDValue(const DDValue& v) const {
  bool result = false;
  unsigned int id = v.id();
  if (id < hasDDValue_.size()) {
    result = hasDDValue_[id];
  }
  return result;
}

void LogicalPart::removeSpecifics(const std::pair<const DDPartSelection*, const DDsvalues_type*>& s) {
  std::vector<std::pair<const DDPartSelection*, const DDsvalues_type*> >::iterator it =
      std::find(specifics_.begin(), specifics_.end(), s);
  specifics_.erase(it);
}

std::vector<const DDsvalues_type*> LogicalPart::specifics() const {
  std::vector<const DDsvalues_type*> result;
  specificsV(result);
  return result;
}

void LogicalPart::specificsV(std::vector<const DDsvalues_type*>& result) const {
  for (const auto& it : specifics_) {
    const DDPartSelection& ps = *(it.first);
    if (ps.size() == 1 && ps[0].selectionType_ == ddanylogp) {
      result.emplace_back(it.second);
    }
  }
}

DDsvalues_type LogicalPart::mergedSpecifics() const {
  DDsvalues_type merged;
  mergedSpecificsV(merged);
  return merged;
}

void LogicalPart::mergedSpecificsV(DDsvalues_type& merged) const {
  merged.clear();
  std::vector<const DDsvalues_type*> unmerged;
  specificsV(unmerged);
  if (unmerged.size() == 1) {
    merged = *(unmerged[0]);
  } else if (unmerged.size() > 1) {
    for (const auto& it : unmerged) {
      merge(merged, *it);
    }
  }
}
