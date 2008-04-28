#ifndef GCTTESTFIRMWARE_H_
#define GCTTESTFIRMWARE_H_

/*!
 * \class gctTestFirmware
 * \brief Test of the emulation of jetfinder firmware
 * 
 * Test functionality for comparison of emulator and firmware jetfinders,
 * migrated from standalone test programs
 *
 * \author Greg Heath
 * \date March 2007
 *
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

#include <vector>
#include <fstream> 

class L1GlobalCaloTrigger;

class gctTestFirmware
{
public:

  // structs and typedefs
  typedef std::vector<L1GctJet> JetsVector;

  // Constructor and destructor
  gctTestFirmware();
  ~gctTestFirmware();

  /// Read the firmware results from a file for the next event
  void fillJetsFromFirmware(const std::string &fileName, const int bxStart, const int numOfBx);

  /// Check the jet finder against the results from the firmware
  bool checkJetFinder(const L1GlobalCaloTrigger* gct) const;

  /// Analyse calculation of energy sums in firmware
  bool checkEnergySumsFromFirmware(const L1GlobalCaloTrigger* gct, const std::string &fileName, const int numOfBx);

private:

  // FUNCTION PROTOTYPES FOR JET FINDER CHECKING
  /// Read one event's worth of jets from the file
  std::vector<JetsVector> getJetsFromFile(const int bxStart, const int numOfBx);
  /// Read a single jet
  L1GctJet nextJetFromFile (const unsigned jf, const int bx);
  //=========================================================================

  std::vector<JetsVector> jetsFromFile;

  std::ifstream jetsFromFirmwareInputFile;
  std::ifstream esumsFromFirmwareInputFile;

};

#endif /*GCTTEST_H_*/
