#include "skelsubsys/xxxEventHypothesis/interface/xxxEventHypothesis.h"

#include <iostream>

using namespace pat;

const char * xxxEventHypothesis::candidateRoles[] = {
PutMyListOfCandidateRolesHere
};

const bool xxxEventHypothesis::isVector[] = {
PutMyListOfVectorBoolsHere
};

bool xxxEventHypothesis::getIsVector( int i ) const
{
  if ( i >= 0 && i < N_ROLES ) {
    return isVector[i];
  } else {
    std::cerr << "xxxEventHypothesis: index out of bounds for roles: " << i << std::endl;
    return false; 
  }
}

const char *  xxxEventHypothesis::getCandidateRole( int i ) const
{
  if ( i >= 0 && i < N_ROLES ) {
    return candidateRoles[i];
  } else {
    std::cerr << "xxxEventHypothesis: index out of bounds for roles: " << i << std::endl;
    return 0; 
  }
}

Candidate & xxxEventHypothesis::getCandidate(std::string name, int index )
{

PutMyRoleSwitchHere;

 std::cerr << "xxxEventHypothesis: Unknown role " << name << ", returning first member" << std::endl;
 return PutMyDefaultReturnHere;
}

int xxxEventHypothesis::getSize(int i) const
{

  std::string name( candidateRoles[i] );

PutMySizesHere;

 std::cerr << "xxxEventHypothesis: Unknown role " << name << ", returning first member" << std::endl;
 return -1;
}
