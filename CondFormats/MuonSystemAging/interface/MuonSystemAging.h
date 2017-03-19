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
class MuonSystemAging {
    public:
    MuonSystemAging();
    ~MuonSystemAging(){}
    std::vector<int>  m_RPCchambers;
    std::vector<std::string>  m_DTchambers;
    double m_CSCineff;
    std::vector<int>  m_GE11Pluschambers;
    std::vector<int>  m_GE11Minuschambers; 
    std::vector<int>  m_GE21Pluschambers;
    std::vector<int>  m_GE21Minuschambers;
    std::vector<int>  m_ME0Pluschambers;
    std::vector<int>  m_ME0Minuschambers;

   COND_SERIALIZABLE;
   };


#endif
