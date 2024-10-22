#include "CondFormats/DQMObjects/interface/HDQMSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

HDQMSummary::HDQMSummary(std::vector<std::string>& userDBContent) {
  userDBContent_ = userDBContent;
  runNr_ = 0;
  timeValue_ = 0;
}

HDQMSummary::HDQMSummary(const HDQMSummary& input) {
  userDBContent_ = input.getUserDBContent();
  runNr_ = input.getTimeValue();
  timeValue_ = input.getRunNr();
  v_sum_.clear();
  indexes_.clear();
  v_sum_.insert(v_sum_.end(), input.v_sum_.begin(), input.v_sum_.end());
  indexes_.insert(indexes_.end(), input.indexes_.begin(), input.indexes_.end());
}

bool HDQMSummary::put(const uint32_t& DetId, InputVector& input, std::vector<std::string>& userContent) {
  Registry::iterator p = std::lower_bound(indexes_.begin(), indexes_.end(), DetId, HDQMSummary::StrictWeakOrdering());

  if (p == indexes_.end() || p->detid != DetId) {
    //First request for the given DetID
    //Create entries for all the declared userDBContent
    //and fill for the provided userContent

    DetRegistry detregistry;
    detregistry.detid = DetId;
    detregistry.ibegin = v_sum_.size();
    indexes_.insert(p, detregistry);
    InputVector tmp(userDBContent_.size(), -9999);

    for (size_t i = 0; i < userContent.size(); ++i)
      tmp[getPosition(userContent[i])] = input[i];

    v_sum_.insert(v_sum_.end(), tmp.begin(), tmp.end());
  } else {
    if (p->detid == DetId) {
      //I should already find the entries
      //fill for the provided userContent

      for (size_t i = 0; i < userContent.size(); ++i)
        v_sum_[p->ibegin + getPosition(userContent[i])] = input[i];
    }
  }

  return true;
}

const HDQMSummary::Range HDQMSummary::getRange(const uint32_t& DetId) const {
  RegistryIterator p = std::lower_bound(indexes_.begin(), indexes_.end(), DetId, HDQMSummary::StrictWeakOrdering());
  if (p == indexes_.end() || p->detid != DetId) {
    edm::LogWarning("HDQMSummary") << "not in range";
    return HDQMSummary::Range(v_sum_.end(), v_sum_.end());
  }
  return HDQMSummary::Range(v_sum_.begin() + p->ibegin, v_sum_.begin() + p->ibegin + userDBContent_.size());
}

std::vector<uint32_t> HDQMSummary::getDetIds() const {
  // returns vector of DetIds in map
  std::vector<uint32_t> DetIds_;
  HDQMSummary::RegistryIterator begin = indexes_.begin();
  HDQMSummary::RegistryIterator end = indexes_.end();
  for (HDQMSummary::RegistryIterator p = begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
  return DetIds_;
}

const short HDQMSummary::getPosition(std::string elementName) const {
  // returns position of elementName in UserDBContent_

  std::vector<std::string>::const_iterator it = find(userDBContent_.begin(), userDBContent_.end(), elementName);
  short pos = -1;
  if (it != userDBContent_.end())
    pos = it - userDBContent_.begin();
  else
    edm::LogError("HDQMSummary") << "attempting to retrieve non existing historic DB object : " << elementName
                                 << std::endl;
  return pos;
}

void HDQMSummary::setObj(const uint32_t& detID, std::string elementName, float value) {
  // modifies value of info "elementName" for the given detID
  // requires that an entry has be defined beforehand for detId in DB
  RegistryIterator p = std::lower_bound(indexes_.begin(), indexes_.end(), detID, HDQMSummary::StrictWeakOrdering());
  if (p == indexes_.end() || p->detid != detID) {
    throw cms::Exception("") << "not allowed to modify " << elementName
                             << " in historic DB - SummaryObj needs to be available first !";
  }

  const HDQMSummary::Range range = getRange(detID);

  std::vector<float>::const_iterator it = range.first + getPosition(elementName);
  std::vector<float>::difference_type pos = -1;
  if (it != v_sum_.end()) {
    pos = it - v_sum_.begin();
    v_sum_.at(pos) = value;
  }
}

std::vector<float> HDQMSummary::getSummaryObj(uint32_t& detID, const std::vector<std::string>& _list) const {
  std::vector<std::string> list = _list;
  std::vector<float> SummaryObj;
  const HDQMSummary::Range range = getRange(detID);
  if (range.first != range.second) {
    for (unsigned int i = 0; i < list.size(); i++) {
      const short pos = getPosition(list.at(i));

      if (pos != -1)
        SummaryObj.push_back(*((range.first) + pos));
      else
        SummaryObj.push_back(-999.);
    }
  } else
    for (unsigned int i = 0; i < list.size(); i++)
      SummaryObj.push_back(
          -99.);  // no summary obj has ever been inserted for this detid, most likely all related histos were not available in DQM

  return SummaryObj;
}

std::vector<float> HDQMSummary::getSummaryObj(uint32_t& detID) const {
  std::vector<float> SummaryObj;
  const HDQMSummary::Range range = getRange(detID);
  if (range.first != range.second) {
    for (unsigned int i = 0; i < userDBContent_.size(); i++)
      SummaryObj.push_back(*((range.first) + i));
  } else {
    for (unsigned int i = 0; i < userDBContent_.size(); i++)
      SummaryObj.push_back(-99.);
  }
  return SummaryObj;
}

std::vector<float> HDQMSummary::getSummaryObj() const { return v_sum_; }

std::vector<float> HDQMSummary::getSummaryObj(std::string elementName) const {
  std::vector<float> vSumElement;
  std::vector<uint32_t> DetIds_ = getDetIds();
  const short pos = getPosition(elementName);

  if (pos != -1) {
    for (unsigned int i = 0; i < DetIds_.size(); i++) {
      const HDQMSummary::Range range = getRange(DetIds_.at(i));
      if (range.first != range.second) {
        vSumElement.push_back(*((range.first) + pos));
      } else {
        vSumElement.push_back(-99.);
      }
    }
  }

  return vSumElement;
}

void HDQMSummary::print() {
  std::cout << "Nr. of detector elements in HDQMSummary object is " << indexes_.size() << " RunNr= " << runNr_
            << " timeValue= " << timeValue_ << std::endl;
}
