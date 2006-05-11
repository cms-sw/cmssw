
#include "DataFormats/GctDigi/interface/L1GctCandidates.h"


L1GctCand::L1GctCand() : theCand(0) { }

L1GctCand::L1GctCand(uint16_t data) : theCand(data) { }

L1GctCand::L1GctCand(int rank, int phi, int eta) {
  theCand = (rank & 0x3f) + ((phi & 0x1f)<<6) + ((eta & 0xf)<<11);
}

L1GctCand::~L1GctCand() { } 


L1GctIsoEmCand::L1GctIsoEmCand() { }

L1GctIsoEmCand::L1GctIsoEmCand(uint16_t data) : L1GctCand(data) { }

L1GctIsoEmCand::L1GctIsoEmCand(int rank, int phi, int eta) : L1GctCand(rank, phi, eta) { }

L1GctIsoEmCand::~L1GctIsoEmCand() { } 


L1GctNonIsoEmCand::L1GctNonIsoEmCand() { }

L1GctNonIsoEmCand::L1GctNonIsoEmCand(uint16_t data) : L1GctCand(data) { }

L1GctNonIsoEmCand::L1GctNonIsoEmCand(int rank, int phi, int eta) : L1GctCand(rank, phi, eta) { }

L1GctNonIsoEmCand::~L1GctNonIsoEmCand() { } 


L1GctCenJetCand::L1GctCenJetCand() { }

L1GctCenJetCand::L1GctCenJetCand(uint16_t data) : L1GctCand(data) { }

L1GctCenJetCand::L1GctCenJetCand(int rank, int phi, int eta) : L1GctCand(rank, phi, eta) { }

L1GctCenJetCand::~L1GctCenJetCand() { } 


L1GctForJetCand::L1GctForJetCand() { }

L1GctForJetCand::L1GctForJetCand(uint16_t data) : L1GctCand(data) { }

L1GctForJetCand::L1GctForJetCand(int rank, int phi, int eta) : L1GctCand(rank, phi, eta) { }

L1GctForJetCand::~L1GctForJetCand() { } 


L1GctTauJetCand::L1GctTauJetCand() { }

L1GctTauJetCand::L1GctTauJetCand(uint16_t data) : L1GctCand(data) { }

L1GctTauJetCand::L1GctTauJetCand(int rank, int phi, int eta) : L1GctCand(rank, phi, eta) { }

L1GctTauJetCand::~L1GctTauJetCand() { } 
