#ifndef GCTTESTSINGLEEVENT_H_
#define GCTTESTSINGLEEVENT_H_

/*!
 * \class gctTestSingleEvent
 * \brief Helper for hardware/emulator comparison tests using single events
 * 
 * Read in the array of region energies from a file
 *
 * \author Greg Heath
 * \date February 2010
 *
 */

#include <string>
#include <vector>
#include <stdint.h>

class L1CaloRegion;

class gctTestSingleEvent
{
public:

  // structs and typedefs

  // Constructor and destructor
  gctTestSingleEvent();
  ~gctTestSingleEvent();

  std::vector<L1CaloRegion> loadEvent(const std::string &fileName, const int16_t bx);

private:

};

#endif /*GCTTEST_H_*/
