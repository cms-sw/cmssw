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

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"

class SurveyInputTextReader {
public:
  typedef std::pair<align::ID, align::StructureType> UniqueId;

  typedef std::map<UniqueId, align::Scalars> MapType;
  typedef std::pair<UniqueId, align::Scalars> PairType;

  /// Read given text file
  void readFile(const std::string& textFileName);

  // Returns the Map
  const MapType& UniqueIdMap() const { return theMap; }

private:
  MapType theMap;

  static const int NINPUTS = 27;  // Not including DetId
};

#endif
