#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cassert>
#include <iomanip>
#include <ostream>

std::vector<std::string> LumiDetails::m_algoNames;

LumiDetails::LumiDetails() :
  m_lumiVersion("-1"),
  m_algoToFirstIndex(kMaxNumAlgos + 1, 0),
  m_allValues(),
  m_allErrors(),
  m_allQualities(),
  m_allBeam1Intensities(),
  m_allBeam2Intensities()
{
}

LumiDetails::LumiDetails(std::string const& lumiVersion) :
  m_lumiVersion(lumiVersion),
  m_algoToFirstIndex(kMaxNumAlgos + 1, 0),
  m_allValues(),
  m_allErrors(),
  m_allQualities(),
  m_allBeam1Intensities(),
  m_allBeam2Intensities()

{
}

LumiDetails::~LumiDetails() {
}

void
LumiDetails::setLumiVersion(std::string const& lumiVersion){
  m_lumiVersion = lumiVersion;
}

std::string const&
LumiDetails::lumiVersion() const {
  return m_lumiVersion;
}

bool
LumiDetails::isValid() const {
  return (m_lumiVersion != "-1");
}

void
LumiDetails::fill(AlgoType algo,
                  std::vector<float> const& values,
                  std::vector<float> const& errors,
                  std::vector<short> const& qualities,
                  std::vector<short> const& beam1Intensities,
                  std::vector<short> const& beam2Intensities) {
  checkAlgo(algo);
  if (values.size() != errors.size() ||
      values.size() != qualities.size() ||
      values.size() != beam1Intensities.size() ||
      values.size() != beam2Intensities.size() ||
      m_algoToFirstIndex[algo] != m_algoToFirstIndex[algo + 1U]) {
    throw edm::Exception(edm::errors::LogicError)
      << "Illegal input values passed to LumiDetails::fill.\n"
      << "The current implementation of LumiDetails only allows filling\n"
      << "vectors for each algorithm once and the input vectors must\n"
      << "all be the same size.\n";
  }
  m_allValues.insert(m_allValues.begin() + m_algoToFirstIndex[algo], values.begin(), values.end());
  m_allErrors.insert(m_allErrors.begin() + m_algoToFirstIndex[algo], errors.begin(), errors.end());
  m_allQualities.insert(m_allQualities.begin() + m_algoToFirstIndex[algo], qualities.begin(), qualities.end());
  m_allBeam1Intensities.insert(m_allBeam1Intensities.begin() + m_algoToFirstIndex[algo], beam1Intensities.begin(), beam1Intensities.end());
  m_allBeam2Intensities.insert(m_allBeam2Intensities.begin() + m_algoToFirstIndex[algo], beam2Intensities.begin(), beam2Intensities.end());
  for (unsigned i = algo + 1U; i <= kMaxNumAlgos; ++i) {
    m_algoToFirstIndex[i] += values.size();
  }
}

float 
LumiDetails::lumiValue(AlgoType algo, unsigned int bx) const {
  checkAlgoAndBX(algo, bx);
  return m_allValues[m_algoToFirstIndex[algo] + bx];
}

float 
LumiDetails::lumiError(AlgoType algo, unsigned int bx) const {
  checkAlgoAndBX(algo, bx);
  return m_allErrors[m_algoToFirstIndex[algo] + bx];
}

short 
LumiDetails::lumiQuality(AlgoType algo, unsigned int bx) const {
  checkAlgoAndBX(algo, bx);
  return m_allQualities[m_algoToFirstIndex[algo] + bx];
}

short
LumiDetails::lumiBeam1Intensity(AlgoType algo, unsigned int bx) const {
  checkAlgoAndBX(algo, bx);
  return m_allBeam1Intensities[m_algoToFirstIndex[algo] + bx];
}

short
LumiDetails::lumiBeam2Intensity(AlgoType algo, unsigned int bx) const {
  checkAlgoAndBX(algo, bx);
  return m_allBeam2Intensities[m_algoToFirstIndex[algo] + bx];
}

LumiDetails::ValueRange
LumiDetails::lumiValuesForAlgo(AlgoType algo) const {
  checkAlgo(algo);
  return ValueRange(m_allValues.begin() + m_algoToFirstIndex[algo],
                    m_allValues.begin() + m_algoToFirstIndex[algo + 1U]);
}

LumiDetails::ErrorRange
LumiDetails::lumiErrorsForAlgo(AlgoType algo) const {
  checkAlgo(algo);
  return ErrorRange(m_allErrors.begin() + m_algoToFirstIndex[algo],
                    m_allErrors.begin() + m_algoToFirstIndex[algo + 1U]);
}

LumiDetails::QualityRange
LumiDetails::lumiQualitiesForAlgo(AlgoType algo) const {
  checkAlgo(algo);
  return QualityRange(m_allQualities.begin() + m_algoToFirstIndex[algo],
                      m_allQualities.begin() + m_algoToFirstIndex[algo + 1U]);
}

LumiDetails::Beam1IntensityRange
LumiDetails::lumiBeam1IntensitiesForAlgo(AlgoType algo) const {
  checkAlgo(algo);
  return Beam1IntensityRange(m_allBeam1Intensities.begin() + m_algoToFirstIndex[algo],
                             m_allBeam1Intensities.begin() + m_algoToFirstIndex[algo + 1U]);
}

LumiDetails::Beam2IntensityRange
LumiDetails::lumiBeam2IntensitiesForAlgo(AlgoType algo) const {
  checkAlgo(algo);
  return Beam2IntensityRange(m_allBeam2Intensities.begin() + m_algoToFirstIndex[algo],
                             m_allBeam2Intensities.begin() + m_algoToFirstIndex[algo + 1U]);
}

std::vector<std::string> const&
LumiDetails::algoNames() {
  if (m_algoNames.size() != kMaxNumAlgos) {
    assert(m_algoNames.size() == 0U);
    m_algoNames.push_back(std::string("OCC1"));
    m_algoNames.push_back(std::string("OCC2"));
    m_algoNames.push_back(std::string("ET"));
    m_algoNames.push_back(std::string("Algo3"));
    m_algoNames.push_back(std::string("PLT1"));
    m_algoNames.push_back(std::string("PLT2"));
    assert(m_algoNames.size() == kMaxNumAlgos);
  }
  return m_algoNames;
}

bool
LumiDetails::isProductEqual(LumiDetails const& lumiDetails) const {

  if (m_lumiVersion == lumiDetails.m_lumiVersion &&
      m_algoToFirstIndex == lumiDetails.m_algoToFirstIndex &&
      m_allValues == lumiDetails.m_allValues &&
      m_allErrors == lumiDetails.m_allErrors &&
      m_allQualities == lumiDetails.m_allQualities &&
      m_allBeam1Intensities == lumiDetails.m_allBeam1Intensities &&
      m_allBeam2Intensities == lumiDetails.m_allBeam2Intensities) {
    return true;
  }
  return false;
}

void
LumiDetails::checkAlgo(AlgoType algo) const {
  if (algo >= kMaxNumAlgos) {
    throw edm::Exception(edm::errors::LogicError)
      << "Algorithm type argument out of range in a call to a function in LumiDetails\n";
  }
}

void
LumiDetails::checkAlgoAndBX(AlgoType algo, unsigned int bx) const {
  checkAlgo(algo);
  if (bx >= (m_algoToFirstIndex[algo + 1U] - m_algoToFirstIndex[algo])) {
    throw edm::Exception(edm::errors::LogicError)
      << "Branch crossing argument out of range in call to a function in LumiDetails\n";
  }
}

std::ostream& operator<<(std::ostream& s, LumiDetails const& lumiDetails) {
  
  s << "\nDumping LumiDetails\n";
  s << std::setw(12) << "lumi version " << lumiDetails.lumiVersion() << "\n";

  std::vector<std::string>::const_iterator algo = lumiDetails.algoNames().begin();
  std::vector<std::string>::const_iterator algoEnd = lumiDetails.algoNames().end();

  LumiDetails::AlgoType i = 0;

  for( ; algo != algoEnd; ++algo, ++i) {

    std::vector<float>::const_iterator value = lumiDetails.lumiValuesForAlgo(i).first;
    std::vector<float>::const_iterator valueEnd = lumiDetails.lumiValuesForAlgo(i).second;
    std::vector<float>::const_iterator error = lumiDetails.lumiErrorsForAlgo(i).first;
    std::vector<short>::const_iterator quality = lumiDetails.lumiQualitiesForAlgo(i).first;
    std::vector<short>::const_iterator beam1 = lumiDetails.lumiBeam1IntensitiesForAlgo(i).first;
    std::vector<short>::const_iterator beam2 = lumiDetails.lumiBeam2IntensitiesForAlgo(i).first;

    s << "algorithm: " << *algo << "\n";
    s << std::setw(12) << "value"
      << std::setw(12) << "error"
      << std::setw(12) << "quality"
      << std::setw(16) << "beam1Intensity"
      << std::setw(16) << "beam2Intensity" << "\n";

    for( ; value != valueEnd; ++value, ++error, ++quality, ++beam1, ++beam2){
      s << std::setw(12) << *value
        << std::setw(12) << *error
        << std::setw(12) << *quality
        << std::setw(16) << *beam1
        << std::setw(16) << *beam2 << "\n";
    }
    s << "\n";
  }
  return s;
}
