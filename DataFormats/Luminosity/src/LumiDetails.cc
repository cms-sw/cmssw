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
  m_beam1Intensities(),
  m_beam2Intensities()
{
}

LumiDetails::LumiDetails(std::string const& lumiVersion) :
  m_lumiVersion(lumiVersion),
  m_algoToFirstIndex(kMaxNumAlgos + 1, 0),
  m_allValues(),
  m_allErrors(),
  m_allQualities(),
  m_beam1Intensities(),
  m_beam2Intensities()
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
  return m_allValues.size()!=0;
}

void
LumiDetails::fill(AlgoType algo,
                  std::vector<float> const& values,
                  std::vector<float> const& errors,
                  std::vector<short> const& qualities) {
  checkAlgo(algo);
  if (values.size() != errors.size() ||
      values.size() != qualities.size() ||
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
  for (unsigned i = algo + 1U; i <= kMaxNumAlgos; ++i) {
    m_algoToFirstIndex[i] += values.size();
  }
}

void
LumiDetails::fillBeamIntensities(std::vector<float> const& beam1Intensities,
                                 std::vector<float> const& beam2Intensities) {
  m_beam1Intensities = beam1Intensities;
  m_beam2Intensities = beam2Intensities;
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

float
LumiDetails::lumiBeam1Intensity(unsigned int bx) const {
  return m_beam1Intensities.at(bx);
}

float
LumiDetails::lumiBeam2Intensity(unsigned int bx) const {
  return m_beam2Intensities.at(bx);
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

std::vector<float> const&
LumiDetails::lumiBeam1Intensities() const {
  return m_beam1Intensities;
}

std::vector<float> const&
LumiDetails::lumiBeam2Intensities() const {
  return m_beam2Intensities;
}

std::vector<std::string> const&
LumiDetails::algoNames() {
  if (m_algoNames.size() != kMaxNumAlgos) {
    assert(m_algoNames.size() == 0U);
    // If in the future additional algorithm names are added,
    // it is important that they be added at the end of the list.
    // The Algos enum in LumiDetails.h also would need to be
    // updated to keep the list of names in sync.
    m_algoNames.push_back(std::string("OCC1"));
    m_algoNames.push_back(std::string("OCC2"));
    m_algoNames.push_back(std::string("ET"));
    m_algoNames.push_back(std::string("PLT"));
    assert(m_algoNames.size() == kMaxNumAlgos);
  }
  return m_algoNames;
}

std::vector<std::string> const&
LumiDetails::dipalgoNames() {
  m_algoNames.push_back(std::string("DIP"));
  return m_algoNames;
}
bool
LumiDetails::isProductEqual(LumiDetails const& lumiDetails) const {
  
  if (m_lumiVersion == lumiDetails.m_lumiVersion &&
      m_algoToFirstIndex == lumiDetails.m_algoToFirstIndex &&
      m_allValues == lumiDetails.m_allValues &&
      m_allErrors == lumiDetails.m_allErrors &&
      m_allQualities == lumiDetails.m_allQualities &&
      m_beam1Intensities == lumiDetails.m_beam1Intensities &&
      m_beam2Intensities == lumiDetails.m_beam2Intensities) {
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
  std::vector<std::string>::const_iterator algo;
  std::vector<std::string>::const_iterator algoEnd;
  if(lumiDetails.lumiVersion()!=std::string("DIP")){
    algo = lumiDetails.algoNames().begin();
    algoEnd = lumiDetails.algoNames().end();
  }else{
    algo = lumiDetails.dipalgoNames().begin();
    algoEnd = lumiDetails.dipalgoNames().end();
  }
  LumiDetails::AlgoType i = 0;

  for( ; algo != algoEnd; ++algo, ++i) {

    std::vector<float>::const_iterator value = lumiDetails.lumiValuesForAlgo(i).first;
    std::vector<float>::const_iterator valueEnd = lumiDetails.lumiValuesForAlgo(i).second;
    std::vector<float>::const_iterator error = lumiDetails.lumiErrorsForAlgo(i).first;
    std::vector<short>::const_iterator quality = lumiDetails.lumiQualitiesForAlgo(i).first;

    s << "algorithm: " << *algo << "\n";
    s << std::setw(12) << "value"
      << std::setw(12) << "error"
      << std::setw(12) << "quality" << "\n";

    for( ; value != valueEnd; ++value, ++error, ++quality){
      s << std::setw(12) << *value
        << std::setw(12) << *error
        << std::setw(12) << *quality << "\n";
    }
    s << "\n";
  }
  s << "beam 1 intensities:\n";
  std::vector<float> const& beam1Intensities = lumiDetails.lumiBeam1Intensities();
  for (std::vector<float>::const_iterator intensity = beam1Intensities.begin(),
	                                       iEnd = beam1Intensities.end();
       intensity != iEnd; ++intensity) {
    s << *intensity << "\n";
  }
  s << "\nbeam 2 intensities:\n";
  std::vector<float> const& beam2Intensities = lumiDetails.lumiBeam2Intensities();
  for (std::vector<float>::const_iterator intensity = beam2Intensities.begin(),
	                                       iEnd = beam2Intensities.end();
       intensity != iEnd; ++intensity) {
    s << *intensity << "\n";
  }
  s << "\n";
  return s;
}
