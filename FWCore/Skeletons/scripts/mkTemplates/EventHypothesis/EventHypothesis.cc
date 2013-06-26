#include "__subsys__/__pkgname__/plugins/__class__.h"

#include <iostream>

using namespace pat;

const char * __class__::candidateRoles[] = {
PutMyListOfCandidateRolesHere
};

const bool __class__::isVector[] = {
PutMyListOfVectorBoolsHere
};

bool __class__::getIsVector( int i ) const
{
  if ( i >= 0 && i < N_ROLES ) {
    return isVector[i];
  } else {
    std::cerr << "__class__: index out of bounds for roles: " << i << std::endl;
    return false; 
  }
}

const char *  __class__::getCandidateRole( int i ) const
{
  if ( i >= 0 && i < N_ROLES ) {
    return candidateRoles[i];
  } else {
    std::cerr << "__class__: index out of bounds for roles: " << i << std::endl;
    return 0; 
  }
}

Candidate & __class__::getCandidate(std::string name, int index )
{

PutMyRoleSwitchHere;

 std::cerr << "__class__: Unknown role " << name << ", returning first member" << std::endl;
 return PutMyDefaultReturnHere;
}

int __class__::getSize(int i) const
{

  std::string name( candidateRoles[i] );

PutMySizesHere;

 std::cerr << "__class__: Unknown role " << name << ", returning first member" << std::endl;
 return -1;
}
