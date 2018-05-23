#ifndef MTDCommonData_MTDBaseNumber_h
#define MTDCommonData_MTDBaseNumber_h

/** \class MTDBaseNumber
 *   
 * Cloned from the EcalBaseNumber class
 */

#include <vector>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class MTDBaseNumber {
 public:

  MTDBaseNumber();
  MTDBaseNumber( const MTDBaseNumber & aBaseNumber );
  ~MTDBaseNumber(){}
  
  void setSize(const int & size); 
  void addLevel(const std::string& name, const int & copyNumber);
  
  int getLevels() const;
  int getCopyNumber(int level) const;
  int getCopyNumber(const std::string& levelName) const;
  std::string const & getLevelName(int level) const;
  int getCapacity();

  void reset();

 protected:
  static constexpr int MAXLEVEL=20;

  std::vector<std::string> _sortedName;
  std::vector<int> _sortedCopyNumber;
  int _theLevels;

};

#endif
