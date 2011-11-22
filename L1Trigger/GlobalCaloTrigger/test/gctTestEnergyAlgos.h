#ifndef GCTTESTENERGYALGOS_H_
#define GCTTESTENERGYALGOS_H_

/*!
 * \class gctTestEnergyAlgos
 * \brief Test of the wheel card and final stage energy summing
 * 
 * Energy sum test functionality migrated from standalone test programs
 *
 * \author Greg Heath
 * \date March 2007
 *
 */
 
#include <vector>
#include <fstream>
#include <stdint.h>

class L1GlobalCaloTrigger;
class L1CaloRegion;

class gctTestEnergyAlgos
{
public:

  // structs and typedefs
  struct etmiss_vec { unsigned mag; unsigned phi; };

  // Constructor and destructor
  gctTestEnergyAlgos();
  ~gctTestEnergyAlgos();

  /// Load another event into the gct. Overloaded for the various ways of doing this.
  std::vector<L1CaloRegion> loadEvent(L1GlobalCaloTrigger* &gct, const bool simpleEvent, const int16_t bx);
  std::vector<L1CaloRegion> loadEvent(L1GlobalCaloTrigger* &gct, const std::string &fileName, bool &endOfFile, const int16_t bx);
  std::vector<L1CaloRegion> loadEvent(L1GlobalCaloTrigger* &gct, const std::vector<L1CaloRegion>& inputRegions, const int16_t bx);

  /// Set array sizes for the number of bunch crossings
  void setBxRange(const int bxStart, const int numOfBx);

  /// Check the energy sums algorithms
  bool checkEnergySums(const L1GlobalCaloTrigger* gct) const;

private:

  // FUNCTION PROTOTYPES FOR EVENT GENERATION
  /// Generates test data for missing Et as 2-vector (magnitude, direction)
  etmiss_vec randomMissingEtVector() const;
  /// Generates test data consisting of energies to be added together with their sum
  std::vector<unsigned> randomTestData(const int size, const unsigned max) const;
  /// Loads test input regions from a text file.
  L1CaloRegion nextRegionFromFile(const unsigned ieta, const unsigned iphi, const int16_t bx);

  //=========================================================================

  //
  // FUNCTION PROTOTYPES FOR ENERGY SUM CHECKING
  /// Integer calculation of Ex or Ey from magnitude for a given pair of phi bins
  int etComponent(const unsigned Emag0, const unsigned fact0,
                  const unsigned Emag1, const unsigned fact1) const;
  /// Calculate et vector from ex and ey, using floating arithmetic and conversion back to integer
  etmiss_vec trueMissingEt(const int ex, const int ey) const;
  //=========================================================================

  int m_bxStart;
  int m_numOfBx;

  std::vector<unsigned> etStripSums; 
  std::vector<bool> inMinusOvrFlow;
  std::vector<bool> inPlusOverFlow;

  std::ifstream regionEnergyMapInputFile;

};

#endif /*GCTTEST_H_*/
