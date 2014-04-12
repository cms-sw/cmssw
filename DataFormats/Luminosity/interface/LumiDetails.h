#ifndef DataFormats_Luminosity_LumiDetails_h
#define DataFormats_Luminosity_LumiDetails_h

/** \class LumiDetails
 *
 *
 * LumiDetails holds Details information: the lumi value, the error on this value, 
 * its quality, and 2 beam intensities for each bunch crossing (BX) in a given
 * luminosity section (LS)   
 *
 * \author Valerie Halyo, David Dagenhart, created June 7, 2007>
 *
 ************************************************************/
 
#include <utility>
#include <vector>
#include <string>
#include <iosfwd>

class LumiDetails {
public:

  // If in the future additional algorithm names are added,
  // it is important that they be added at the end of the list.
  // The LumiDetails::algoNames function in LumiDetails.cc also
  // would need to be updated to keep the list of names in sync.
  enum Algos {
    kOCC1,
    kOCC2,
    kET,
    kPLT,
    kMaxNumAlgos
  };
  typedef unsigned int AlgoType;
  typedef std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator> ValueRange;
  typedef std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator> ErrorRange;
  typedef std::pair<std::vector<short>::const_iterator, std::vector<short>::const_iterator> QualityRange;

  LumiDetails();
  explicit LumiDetails(std::string const& lumiVersion);
  ~LumiDetails();

  void setLumiVersion(std::string const& lumiVersion);
  std::string const& lumiVersion() const;
  bool isValid() const;

  // This will perform more efficiently if the calls to this
  // are in the same order as the Algos enumeration.  It will
  // work properly even if they are not.
  void fill(AlgoType algo,
            std::vector<float> const& values,
            std::vector<float> const& errors,
            std::vector<short> const& qualities);

  void fillBeamIntensities(std::vector<float> const& beam1Intensities,
                           std::vector<float> const& beam2Intensities);

  float lumiValue(AlgoType algo, unsigned int bx) const;
  float lumiError(AlgoType algo, unsigned int bx) const;
  short lumiQuality(AlgoType algo, unsigned int bx) const;
  float lumiBeam1Intensity(unsigned int bx) const;
  float lumiBeam2Intensity(unsigned int bx) const;

  ValueRange lumiValuesForAlgo(AlgoType algo) const;
  ErrorRange lumiErrorsForAlgo(AlgoType algo) const;
  QualityRange lumiQualitiesForAlgo(AlgoType algo) const;
  std::vector<float> const& lumiBeam1Intensities() const;
  std::vector<float> const& lumiBeam2Intensities() const;

  bool isProductEqual(LumiDetails const& lumiDetails) const;

  static std::vector<std::string> const& algoNames();

  static std::vector<std::string> const& dipalgoNames();

private:

  void checkAlgo(AlgoType algo) const;
  void checkAlgoAndBX(AlgoType algo, unsigned int bx) const;

  static std::vector<std::string> m_algoNames;

  std::string m_lumiVersion;

  /* m_algoToFirstIndex is 'kMaxNumAlgos' long. Each algorithm's 
     numerical value from the enum Algos is used as the index into m_algoToFirstIndex
     to find the first entry into the m_all* vectors containing data for that
     algorithm.  The entry beyond the last entry is found by using the numerical value + 1.
     If the first and last index are the same then there is no information recorded for that
     algorithm.
  */
  std::vector<unsigned int> m_algoToFirstIndex;
  std::vector<float> m_allValues;
  std::vector<float> m_allErrors;
  std::vector<short> m_allQualities;
  std::vector<float> m_beam1Intensities;
  std::vector<float> m_beam2Intensities;
};

std::ostream& operator<<(std::ostream & s, LumiDetails const& lumiDetails);

#endif
