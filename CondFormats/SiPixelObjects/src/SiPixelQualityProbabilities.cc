#include "CondFormats/SiPixelObjects/interface/SiPixelQualityProbabilities.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>  // std::setw

//****************************************************************************//
void SiPixelQualityProbabilities::setProbabilities(const unsigned int puBin, const probabilityVec &theProbabilities) {
  if (m_probabilities.find(puBin) != m_probabilities.end()) {
    edm::LogWarning("SiPixelQualityProbabilities") << "PU bin: " << puBin << " is already in the map!" << std::endl;
    return;
  } else {
    m_probabilities.emplace(puBin, theProbabilities);
  }
}

//****************************************************************************//
SiPixelQualityProbabilities::probabilityVec SiPixelQualityProbabilities::getProbabilities(
    const unsigned int puBin) const {
  probabilityMap::const_iterator it = m_probabilities.find(puBin);

  if (it != m_probabilities.end()) {
    return it->second;
  } else {
    throw cms::Exception("SiPixelQualityProbabilities") << "No Probabilities are defined for PU bin " << puBin << "\n";
  }
}

//****************************************************************************//
const SiPixelQualityProbabilities::probabilityVec &SiPixelQualityProbabilities::getProbabilities(
    const unsigned int puBin) {
  probabilityMap::const_iterator it = m_probabilities.find(puBin);

  if (it != m_probabilities.end()) {
    return it->second;
  } else {
    throw cms::Exception("SiPixelQualityProbabilities") << "No Probabilities are defined for PU bin " << puBin << "\n";
  }
}

//****************************************************************************//
void SiPixelQualityProbabilities::printAll() const {
  edm::LogVerbatim("SiPixelQualityProbabilities") << "SiPixelQualityProbabilities::printAll()";
  edm::LogVerbatim("SiPixelQualityProbabilities") << " ================================================================"
                                                     "===================================================";
  for (auto it = m_probabilities.begin(); it != m_probabilities.end(); ++it) {
    edm::LogVerbatim("SiPixelQualityProbabilities") << "PU :" << it->first << "  \n ";
    for (const auto &entry : it->second) {
      edm::LogVerbatim("SiPixelQualityProbabilities")
          << "SiPixelQuality snapshot: " << entry.first << " |probability: " << entry.second << std::endl;
    }
  }
}

//****************************************************************************//
void SiPixelQualityProbabilities::print(std::ostream &os) const {
  for (auto it = m_probabilities.begin(); it != m_probabilities.end(); ++it) {
    os << "PU :" << it->first << "  \n ";
    for (const auto &entry : it->second) {
      os << "SiPixelQuality snapshot: " << entry.first << " |probability: " << entry.second << std::endl;
    }
  }
}

//****************************************************************************//
std::vector<unsigned int> SiPixelQualityProbabilities::getPileUpBins() const {
  std::vector<unsigned int> bins;
  bins.reserve(m_probabilities.size());

  for (auto it = m_probabilities.begin(); it != m_probabilities.end(); ++it) {
    bins.push_back(it->first);
  }
  return bins;
}

//****************************************************************************//
std::ostream &operator<<(std::ostream &os, SiPixelQualityProbabilities theProbabilities) {
  std::stringstream ss;
  theProbabilities.print(ss);
  os << ss.str();
  return os;
}
