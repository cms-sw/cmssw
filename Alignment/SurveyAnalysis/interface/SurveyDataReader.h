#ifndef SurveyDataReader_h
#define SurveyDataReader_h

//
// Class to read in survey data from text file
// Where all numbers are floats, except DetId which is an unsigned integer
//
// The result is a map of DetIds and corresponding OpticalInfo
//

#include "Alignment/CommonAlignment/interface/Utilities.h"

class TrackerTopology;

class SurveyDataReader
{

public:

  typedef std::map<align::ID, align::Scalars >  MapType;
  typedef std::pair<align::ID,align::Scalars > PairType;
  typedef std::map< std::vector<int>, align::Scalars > MapTypeOr;
  typedef std::pair< std::vector<int>, align::Scalars > PairTypeOr;
  
  /// Read given text file
  void readFile( const std::string& textFileName, const std::string& fileType, const TrackerTopology* tTopo);
  align::Scalars convertToAlignableCoord( const align::Scalars& align_params );

  // Returns the Map
  const MapType& detIdMap() const { return theMap; }
  const MapTypeOr& surveyMap() const { return theOriginalMap; }

private:

  MapType theMap;
  MapTypeOr theOriginalMap;

};

#endif
