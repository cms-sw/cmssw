#ifndef MuonSystemAging_H
#define MuonSystemAging_H
/*  example of polymorphic condition
 *   *  LUT, function, mixed....
 *    * this is just a prototype: classes do not need to be defined and declared in the same file
 *     * at the moment though all derived classes better sit in the same package together with the base one
 *      */

#include "CondFormats/Serialization/interface/Serializable.h"
#include<cmath>
#include<iostream>
#include <vector>
#include <regex>
#include <map>

enum CSCInefficiencyType 
  { 
    EFF_CHAMBER=0, 
    EFF_STRIPS=1,
    EFF_WIRES=2 
  };

class MuonSystemAging 
{

 public:

  MuonSystemAging();
  ~MuonSystemAging(){}

  std::map<unsigned int, float>  m_RPCChambEffs;
  std::map<unsigned int, float>  m_DTChambEffs;
  std::map<unsigned int, std::pair<unsigned int, float> >  m_CSCChambEffs;

  std::map<unsigned int, float>  m_GEMChambEffs;
  std::map<unsigned int, float>  m_ME0ChambEffs;
  
  COND_SERIALIZABLE;

};

#endif
