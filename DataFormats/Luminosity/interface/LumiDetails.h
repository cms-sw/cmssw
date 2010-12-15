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

  enum Algos {
    kOCC1,
    kOCC2,
    kET,
    kAlgo3,
    kPLT1,
    kPLT2,
    kMaxNumAlgos
  };
  typedef unsigned int AlgoType;
  typedef std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator> ValueRange;
  typedef std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator> ErrorRange;
  typedef std::pair<std::vector<short>::const_iterator, std::vector<short>::const_iterator> QualityRange;
  typedef std::pair<std::vector<short>::const_iterator, std::vector<short>::const_iterator> Beam1IntensityRange;
  typedef std::pair<std::vector<short>::const_iterator, std::vector<short>::const_iterator> Beam2IntensityRange;

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
            std::vector<short> const& qualities,
	    std::vector<short> const& beam1Intensities,
            std::vector<short> const& beam2Intensities);

  float lumiValue(AlgoType algo, unsigned int bx) const;
  float lumiError(AlgoType algo, unsigned int bx) const;
  short lumiQuality(AlgoType algo, unsigned int bx) const;
  short lumiBeam1Intensity(AlgoType algo, unsigned int bx) const;
  short lumiBeam2Intensity(AlgoType algo, unsigned int bx) const;

  ValueRange lumiValuesForAlgo(AlgoType algo) const;
  ErrorRange lumiErrorsForAlgo(AlgoType algo) const;
  QualityRange lumiQualitiesForAlgo(AlgoType algo) const;
  Beam1IntensityRange lumiBeam1IntensitiesForAlgo(AlgoType algo) const;
  Beam2IntensityRange lumiBeam2IntensitiesForAlgo(AlgoType algo) const;

  bool isProductEqual(LumiDetails const& lumiDetails) const;

  static std::vector<std::string> const& algoNames();

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
  std::vector<short> m_allBeam1Intensities;
  std::vector<short> m_allBeam2Intensities;
};

std::ostream& operator<<(std::ostream & s, LumiDetails const& lumiDetails);

#endif
