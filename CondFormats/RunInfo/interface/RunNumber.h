#ifndef RunNumber_h
#define RunNumber_h
*
 *  \class RunNumber
 *  
 *  hold runnumber info from Run Control
 *  
 *  \author Michele de Gruttola (degrutto) - INFN Naples / CERN (June-12-2008)
 *
*/

#include <iostream>
#include<vector>

class RunNumber {
public:
  struct Item {
    Item(){}
    ~Item(){}
    int m_index;
    std::string m_number;
    std::string m_name;
  };
  RunNumber();
  virtual ~RunNumber(){}
  typedef std::vector<Item>::const_iterator ItemIterator;
  std::vector<Item>  m_runnumber;
};
#endif
