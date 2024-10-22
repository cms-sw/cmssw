/**
 * \file TkModuleGroupSelector.cc
 *
 *  \author Joerg Behr
 *  \date May 2013
 *  $Revision: 1.5 $
 *  $Date: 2013/06/19 08:33:03 $
 *  (last update by $Author: jbehr $)
 */

#include "Alignment/CommonAlignmentAlgorithm/interface/TkModuleGroupSelector.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <map>
#include <set>

//============================================================================
TkModuleGroupSelector::TkModuleGroupSelector(AlignableTracker *aliTracker,
                                             const edm::ParameterSet &cfg,
                                             const std::vector<int> &sdets)
    : nparameters_(0), subdetids_(sdets) {
  //verify that all provided options are known
  std::vector<std::string> parameterNames = cfg.getParameterNames();
  for (std::vector<std::string>::const_iterator iParam = parameterNames.begin(); iParam != parameterNames.end();
       ++iParam) {
    const std::string &name = (*iParam);
    if (name != "RunRange" && name != "ReferenceRun" && name != "Granularity") {
      throw cms::Exception("BadConfig") << "@SUB=TkModuleGroupSelector::TkModuleGroupSelector:"
                                        << " Unknown parameter name '" << name << "' in PSet. Maybe a typo?";
    }
  }

  //extract the reference run range if defined
  const edm::RunNumber_t defaultReferenceRun =
      (cfg.exists("ReferenceRun") ? cfg.getParameter<edm::RunNumber_t>("ReferenceRun") : 0);

  //extract run range to be used for all module groups (if not locally overwritten)
  const std::vector<edm::RunNumber_t> defaultRunRange =
      (cfg.exists("RunRange") ? cfg.getParameter<std::vector<edm::RunNumber_t> >("RunRange")
                              : std::vector<edm::RunNumber_t>());

  // finally create everything from configuration
  this->createModuleGroups(
      aliTracker, cfg.getParameter<edm::VParameterSet>("Granularity"), defaultRunRange, defaultReferenceRun);
}

//============================================================================
void TkModuleGroupSelector::fillDetIdMap(const unsigned int detid, const unsigned int groupid) {
  //only add new entries
  if (mapDetIdGroupId_.find(detid) == mapDetIdGroupId_.end()) {
    mapDetIdGroupId_.insert(std::pair<unsigned int, unsigned int>(detid, groupid));
  } else {
    throw cms::Exception("BadConfig") << "@SUB=TkModuleGroupSelector:fillDetIdMap:"
                                      << " Module with det ID " << detid << " configured in group " << groupid
                                      << " but it was already selected"
                                      << " in group " << mapDetIdGroupId_[detid] << ".";
  }
}

//============================================================================
const bool TkModuleGroupSelector::testSplitOption(const edm::ParameterSet &pset) const {
  bool split = false;
  if (pset.exists("split")) {
    split = pset.getParameter<bool>("split");
  }
  return split;
}

//============================================================================
bool TkModuleGroupSelector::createGroup(unsigned int &Id,
                                        const std::vector<edm::RunNumber_t> &range,
                                        const std::list<Alignable *> &selected_alis,
                                        const edm::RunNumber_t refrun) {
  bool modules_selected = false;

  referenceRun_.push_back(refrun);
  firstId_.push_back(Id);
  runRange_.push_back(range);
  for (std::list<Alignable *>::const_iterator it = selected_alis.begin(); it != selected_alis.end(); ++it) {
    this->fillDetIdMap((*it)->id(), firstId_.size() - 1);
    modules_selected = true;
  }
  if (refrun > 0 && !range.empty()) {  //FIXME: last condition not really needed?
    Id += range.size() - 1;
    nparameters_ += range.size() - 1;
  } else {
    Id += range.size();
    nparameters_ += range.size();
  }

  if (refrun > 0 && range.front() > refrun) {  //range.size() > 0 checked before
    throw cms::Exception("BadConfig") << "@SUB=TkModuleGroupSelector::createGroup:\n"
                                      << "Invalid combination of reference run number and specified run dependence"
                                      << "\n in module group " << firstId_.size() << "."
                                      << "\n Reference run number (" << refrun << ") is smaller than starting run "
                                      << "\n number (" << range.front() << ") of first IOV.";
  }
  return modules_selected;
}

//============================================================================
void TkModuleGroupSelector::verifyParameterNames(const edm::ParameterSet &pset, unsigned int psetnr) const {
  std::vector<std::string> parameterNames = pset.getParameterNames();
  for (std::vector<std::string>::const_iterator iParam = parameterNames.begin(); iParam != parameterNames.end();
       ++iParam) {
    const std::string &name = (*iParam);
    if (name != "levels" && name != "RunRange" && name != "split" && name != "ReferenceRun") {
      throw cms::Exception("BadConfig") << "@SUB=TkModuleGroupSelector::verifyParameterNames:"
                                        << " Unknown parameter name '" << name << "' in PSet number " << psetnr
                                        << ". Maybe a typo?";
    }
  }
}

//============================================================================
void TkModuleGroupSelector::createModuleGroups(AlignableTracker *aliTracker,
                                               const edm::VParameterSet &granularityConfig,
                                               const std::vector<edm::RunNumber_t> &defaultRunRange,
                                               edm::RunNumber_t defaultReferenceRun) {
  std::set<edm::RunNumber_t> localRunRange;
  nparameters_ = 0;
  unsigned int Id = 0;
  unsigned int psetnr = 0;
  //loop over all LA groups
  for (edm::VParameterSet::const_iterator pset = granularityConfig.begin(); pset != granularityConfig.end(); ++pset) {
    //test for unknown parameters
    this->verifyParameterNames((*pset), psetnr);
    psetnr++;

    bool modules_selected = false;  //track whether at all a module has been selected in this group
    const std::vector<edm::RunNumber_t> range =
        ((*pset).exists("RunRange") ? pset->getParameter<std::vector<edm::RunNumber_t> >("RunRange") : defaultRunRange);
    if (range.empty()) {
      throw cms::Exception("BadConfig") << "@SUB=TkModuleGroupSelector::createModuleGroups:\n"
                                        << "Run range array empty!";
    }
    const bool split = this->testSplitOption((*pset));

    edm::RunNumber_t refrun = 0;
    if ((*pset).exists("ReferenceRun")) {
      refrun = (*pset).getParameter<edm::RunNumber_t>("ReferenceRun");
    } else {
      refrun = defaultReferenceRun;
    }

    AlignmentParameterSelector selector(aliTracker);
    selector.clear();
    selector.addSelections((*pset).getParameter<edm::ParameterSet>("levels"));

    const auto &alis = selector.selectedAlignables();
    std::list<Alignable *> selected_alis;
    for (const auto &it : alis) {
      const auto &aliDaughts = it->deepComponents();
      for (const auto &iD : aliDaughts) {
        if (iD->alignableObjectId() == align::AlignableDetUnit || iD->alignableObjectId() == align::AlignableDet) {
          if (split) {
            modules_selected = this->createGroup(Id, range, std::list<Alignable *>(1, iD), refrun);
          } else {
            selected_alis.push_back(iD);
          }
        }
      }
    }

    if (!split) {
      modules_selected = this->createGroup(Id, range, selected_alis, refrun);
    }

    edm::RunNumber_t firstRun = 0;
    for (std::vector<edm::RunNumber_t>::const_iterator iRun = range.begin(); iRun != range.end(); ++iRun) {
      localRunRange.insert((*iRun));
      if ((*iRun) > firstRun) {
        firstRun = (*iRun);
      } else {
        throw cms::Exception("BadConfig") << "@SUB=TkModuleGroupSelector::createModuleGroups:"
                                          << " Run range not sorted.";
      }
    }

    if (!modules_selected) {
      throw cms::Exception("BadConfig") << "@SUB=TkModuleGroupSelector:createModuleGroups:"
                                        << " No module was selected in the module group selector in group "
                                        << (firstId_.size() - 1) << ".";
    }
  }

  //copy local set into the global vector of run boundaries
  for (std::set<edm::RunNumber_t>::const_iterator itRun = localRunRange.begin(); itRun != localRunRange.end();
       ++itRun) {
    globalRunRange_.push_back((*itRun));
  }
}

//============================================================================
unsigned int TkModuleGroupSelector::getNumberOfParameters() const { return nparameters_; }

//============================================================================
unsigned int TkModuleGroupSelector::numIovs() const { return globalRunRange_.size(); }

//============================================================================
edm::RunNumber_t TkModuleGroupSelector::firstRunOfIOV(unsigned int iovNum) const {
  return iovNum < this->numIovs() ? globalRunRange_.at(iovNum) : 0;
}

//======================================================================
int TkModuleGroupSelector::getParameterIndexFromDetId(unsigned int detId, edm::RunNumber_t run) const {
  // Return the index of the parameter that is used for this DetId.
  // If this DetId is not treated, return values < 0.

  const DetId temp_id(detId);

  int index = -1;

  bool sel = false;
  for (std::vector<int>::const_iterator itSubDets = subdetids_.begin(); itSubDets != subdetids_.end(); ++itSubDets) {
    if (temp_id.det() == DetId::Tracker && temp_id.subdetId() == (*itSubDets)) {
      sel = true;
      break;
    }
  }

  if (temp_id.det() != DetId::Tracker || !sel)
    return -1;

  std::map<unsigned int, unsigned int>::const_iterator it = mapDetIdGroupId_.find(detId);
  if (it != mapDetIdGroupId_.end()) {
    const unsigned int iAlignableGroup = (*it).second;
    const std::vector<edm::RunNumber_t> &runs = runRange_.at(iAlignableGroup);
    const unsigned int id0 = firstId_.at(iAlignableGroup);
    const edm::RunNumber_t refrun = referenceRun_.at(iAlignableGroup);

    unsigned int iovNum = 0;
    for (; iovNum < runs.size(); ++iovNum) {
      if (runs[iovNum] > run)
        break;
    }
    if (iovNum == 0) {
      throw cms::Exception("BadConfig") << "@SUB=TkModuleGroupSelector::getParameterIndexFromDetId:\n"
                                        << "Run " << run << " not foreseen for detid '" << detId << "'"
                                        << " in module group " << iAlignableGroup << ".";
    } else {
      --iovNum;
    }

    //test whether the iov contains the reference run
    if (refrun > 0) {  //if > 0 a reference run number has been provided
      if (iovNum + 1 == runs.size()) {
        if (refrun >= runs[iovNum])
          return -1;
      } else if ((iovNum + 1) < runs.size()) {
        if (refrun >= runs[iovNum] && refrun < runs[iovNum + 1]) {
          return -1;
        }
      }
      if (run > refrun) {
        //iovNum > 0 due to checks in createGroup(...) and createModuleGroups(...)
        //remove IOV in which the reference run can be found
        iovNum -= 1;
      }
    }

    index = id0 + iovNum;
  }
  return index;
}
