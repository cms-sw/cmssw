#ifndef EcalCommonData_EcalBaseNumber_h
#define EcalCommonData_EcalBaseNumber_h

/** \class EcalBaseNumber
 *   
 * \author Paolo Meridiani, INFN Roma1 
 *  $Id: $
 */

#include <vector>
#include <string>

class EcalBaseNumber {
 public:

  EcalBaseNumber(){};
  ~EcalBaseNumber(){};

  void addLevel(std::string name, int copyNumber);
  
  int getLevels() const;
  int getCopyNumber(int level) const;
  int getCopyNumber(std::string levelName) const;
  std::string getLevelName(int level) const;

 protected:
  typedef std::vector< std::pair< std::string,int > > basenumber_type;
  basenumber_type _sortedBaseNumber;  

};

#endif
