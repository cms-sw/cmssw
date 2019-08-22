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
class L1GlobalCaloTrigger;

namespace edm {
  class Event;
  class InputTag;
}  // namespace edm

class gctTestUsingLhcData {
public:
  // structs and typedefs

  // Constructor and destructor
  gctTestUsingLhcData();
  ~gctTestUsingLhcData();

  std::vector<L1CaloRegion> loadEvent(const edm::Event& iEvent, const int16_t bx);

  void checkHwResults(const L1GlobalCaloTrigger* gct, const edm::Event& iEvent);
  void checkEmResults(const L1GlobalCaloTrigger* gct, const edm::Event& iEvent);

private:
  bool checkResults(const L1GlobalCaloTrigger* gct, const edm::Event& iEvent, const edm::InputTag tag);

  bool checkJets(const L1GlobalCaloTrigger* gct, const edm::Event& iEvent, const edm::InputTag tag);
  bool checkEtSums(const L1GlobalCaloTrigger* gct, const edm::Event& iEvent, const edm::InputTag tag);
  bool checkHtSums(const L1GlobalCaloTrigger* gct, const edm::Event& iEvent, const edm::InputTag tag);
};

#endif /*GCTTEST_H_*/
