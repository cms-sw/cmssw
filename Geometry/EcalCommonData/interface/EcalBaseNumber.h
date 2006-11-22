#ifndef EcalCommonData_EcalBaseNumber_h
#define EcalCommonData_EcalBaseNumber_h

/** \class EcalBaseNumber
 *   
 * \author Paolo Meridiani, INFN Roma1 
 *  $Id: EcalBaseNumber.h,v 1.1 2006/02/23 18:20:01 meridian Exp $
 */

#include <vector>
#include <string>

class EcalBaseNumber {
 public:

  EcalBaseNumber(){};
  ~EcalBaseNumber(){};
  
  void setSize(int size) { _sortedBaseNumber.reserve(size); }
  void addLevel(const std::string& name, int copyNumber);
  
  int getLevels() const;
  int getCopyNumber(int level) const;
  int getCopyNumber(const std::string& levelName) const;
  std::string getLevelName(int level) const;

 protected:
  typedef std::vector< std::pair< std::string,int > > basenumber_type;
  basenumber_type _sortedBaseNumber;  

};

#endif
