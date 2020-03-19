#ifndef GCTTESTELECTRONS_H_
#define GCTTESTELECTRONS_H_

/*!
 * \class gctTestElectrons
 * \brief Test of the electron sorting
 * 
 * Electron sort test functionality migrated from standalone test programs
 *
 * \author Greg Heath
 * \date March 2007
 *
 */

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

#include <vector>
#include <string>

class L1GlobalCaloTrigger;
class L1GctElectronSorter;

class gctTestElectrons {
public:
  gctTestElectrons();
  ~gctTestElectrons();

  /// Load another event into the gct. Overloaded for the various ways of doing this.
  std::vector<L1CaloEmCand> loadEvent(L1GlobalCaloTrigger*& gct, const std::string fileName, const int16_t bx);

  /// Read the input electron data (after GCT processing).
  void fillElectronData(const L1GlobalCaloTrigger* gct);

  /// Repeat the sort locally and check the result
  bool checkElectrons(const L1GlobalCaloTrigger* gct, const int bxStart, const int numOfBx);

private:
  void LoadFileData(const std::string& inputFile, const int16_t bx);
  void print(const std::vector<L1GctEmCand> cands) const;

  L1GctElectronSorter* m_theIsoEmCandSorter;
  L1GctElectronSorter* m_nonIsoEmCandSorter;

  std::string m_fileNameUsed;

  std::vector<L1GctEmCand> m_theIsoEmCandsFromGct;
  std::vector<L1GctEmCand> m_nonIsoEmCandsFromGct;

  std::vector<L1CaloEmCand> m_theIsoEmCandsFromFileInput;
  std::vector<L1CaloEmCand> m_nonIsoEmCandsFromFileInput;

  std::vector<L1GctEmCand> m_theIsoEmCandsFromFileSorted;
  std::vector<L1GctEmCand> m_nonIsoEmCandsFromFileSorted;
};

#endif /*GCTTEST_H_*/
