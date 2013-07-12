#ifndef EcalCommonData_EcalBaseNumber_h
#define EcalCommonData_EcalBaseNumber_h

/** \class EcalBaseNumber
 *   
 * \author Paolo Meridiani, INFN Roma1 
 *  $Id: EcalBaseNumber.h,v 1.4 2007/06/14 06:41:58 innocent Exp $
 */

#include <vector>
#include <string>

class EcalBaseNumber {
 public:

  EcalBaseNumber();
  EcalBaseNumber( const EcalBaseNumber & aBaseNumber );
  ~EcalBaseNumber(){}
  
  void setSize(const int & size); 
  void addLevel(const std::string& name, const int & copyNumber);
  
  int getLevels() const;
  int getCopyNumber(int level) const;
  int getCopyNumber(const std::string& levelName) const;
  std::string const & getLevelName(int level) const;
  int getCapacity();

  void reset();

 protected:
  std::vector<std::string> _sortedName;
  std::vector<int> _sortedCopyNumber;
  int _theLevels;

};

#endif
