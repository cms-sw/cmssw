#ifndef GCTTESTUSINGLHCDATA_H_
#define GCTTESTUSINGLHCDATA_H_

/*!
 * \class gctTestUsingLhcData
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

namespace edm {
  class Event;
}

class gctTestUsingLhcData
{
public:

  // structs and typedefs

  // Constructor and destructor
  gctTestUsingLhcData();
  ~gctTestUsingLhcData();

  std::vector<L1CaloRegion> loadEvent(const edm::Event& iEvent, const int16_t bx);

private:

};

#endif /*GCTTEST_H_*/
