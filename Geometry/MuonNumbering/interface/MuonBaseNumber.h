#ifndef Geometry_MuonNumbering_MuonBaseNumber_h
#define Geometry_MuonNumbering_MuonBaseNumber_h

/** \class MuonBaseNumber
 *
 * the muon base number collects all significant copy
 * numbers to uniquely identify a detector unit;
 * the information is kept in a vector of all relevant 
 * LevelBaseNumber's needed to identify the detector unit;
 * a packed version of the MuonBaseNumber may replace 
 * the current numbering scheme in future
 *  
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 */

#include <vector>

#include "Geometry/MuonNumbering/src/LevelBaseNumber.h"

class MuonBaseNumber {
 public:

  MuonBaseNumber(){};
  ~MuonBaseNumber(){};

  void addBase(const int level,const int super,const int base);
  void addBase(const LevelBaseNumber&);
  
  int getLevels() const;
  int getSuperNo(int level) const;
  int getBaseNo(int level) const;


 protected:
  typedef std::vector<LevelBaseNumber> basenumber_type;
  basenumber_type sortedBaseNumber;  

};

#endif
