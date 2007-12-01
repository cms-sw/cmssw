#ifndef SurveyDataReader_h
#define SurveyDataReader_h

//
// Class to read in survey data from text file
// Where all numbers are floats, except DetId which is an unsigned integer
//
// The result is a map of DetIds and corresponding OpticalInfo
//
#include <string>
#include <vector>
#include <map>

class SurveyDataReader
{

  typedef unsigned int DetIdType;
  typedef std::map< DetIdType, std::vector<float> > MapType;
  typedef std::pair< DetIdType, std::vector<float> > PairType;
  typedef std::map< std::vector<int>, std::vector<float> > MapTypeOr;
  typedef std::pair< std::vector<int>, std::vector<float> > PairTypeOr;

public:
  /// Constructor
  SurveyDataReader() {};
  
  /// Destructor
  ~SurveyDataReader() {};
  
  /// Read given text file
  void readFile( const std::string& textFileName, const std::string& fileType );
  std::vector<float> convertToAlignableCoord( std::vector<float> align_params );

  // Returns the Map
  const MapType detIdMap() const { return theMap; }
  const MapTypeOr surveyMap() const { return theOriginalMap; }

private:

  MapType theMap;
  MapTypeOr theOriginalMap;

};

#endif
