#ifndef L1RCTElectronIsolationCard_h
#define L1RCTElectronIsolationCard_h

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTRegion.h"
#include <iostream>
#include <vector>

#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

// This class handles the electron isolation card.  Duh.
// Each card has a crate and a cardnumber to identify it when
// being called.
// The class does not actually have instances of L1RCTRegions but
// rather has pointers to regions that were created in L1RCTReceiverCard
// for efficiency purposes.

class L1RCTLookupTables;

class L1RCTElectronIsolationCard {
public:
  L1RCTElectronIsolationCard(int crateNumber, int cardNumber, const L1RCTLookupTables *rctLookupTables);
  L1RCTElectronIsolationCard() = delete;
  ~L1RCTElectronIsolationCard();

  int crateNumber() { return crtNo; }
  int cardNumber() { return cardNo; }

  void fillElectronCandidates();
  void setRegion(int i, const L1RCTRegion &region) { regions.at(i) = region; }
  // Valid arguments to the following two functions are 0 or 1,
  // corresponding to region0 or region1
  unsigned short getIsoElectrons(int i) { return isoElectrons.at(i); }

  unsigned short getNonIsoElectrons(int i) { return nonIsoElectrons.at(i); }
  void print();
  void printEdges() {
    regions.at(0).printEdges();
    regions.at(1).printEdges();
  }

private:
  std::vector<unsigned short> calcElectronCandidates(const L1RCTRegion &region, int regionNum);
  unsigned short calcMaxSum(unsigned short primaryEt,
                            unsigned short northEt,
                            unsigned short southEt,
                            unsigned short eastEt,
                            unsigned short westEt);

  unsigned short crtNo;   // changed from int
  unsigned short cardNo;  // changed from int

  const L1RCTLookupTables *rctLookupTables_;

  L1RCTRegion empty;

  std::vector<unsigned short> isoElectrons;
  std::vector<unsigned short> nonIsoElectrons;
  std::vector<L1RCTRegion> regions;
};

#endif
