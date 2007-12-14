#ifndef SurveyInputTextReader_h
#define SurveyInputTextReader_h
//
// Class to read in survey data from text file
//
// The format of the file is assumed to be:
//
// DetId AlignableObjectId dx sigma_x dy sigma_y dz sigma_z angle_x sigma(angle_x) ...
//   angle_y sigma(angle_y) angle_z sigma(angle_z)
// Where all numbers are floats, except DetId which is an unsigned integer
//
// The result is a map of UniqueIds and corresponding SurveyInfo
//
#include <string>
#include <vector>
#include <map>
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"


class SurveyInputTextReader
{
public:
	typedef unsigned int DetIdType;
	typedef std::map< TrackerAlignableId::UniqueId, std::vector<float> > MapType;
	typedef std::pair< TrackerAlignableId::UniqueId, std::vector<float> > PairType;
	
  /// Constructor
  SurveyInputTextReader() {};

  /// Destructor
  ~SurveyInputTextReader() {};

  /// Read given text file
  void readFile( const std::string& textFileName );

  // Returns the Map
  const MapType UniqueIdMap() const { return theMap; }

private:
	
  MapType theMap;
	static const int NINPUTS = 27; // Not including DetId

	
};

#endif
