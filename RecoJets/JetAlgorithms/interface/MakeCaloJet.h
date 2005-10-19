#ifndef JetAlgorithms_MakeCaloJet_h
#define JetAlgorithms_MakeCaloJet_h

// MakeCaloJet.h
// Initial Version form Fernando Varela Rodriguez
// History: R. Harris, Oct 19, 2005, modified to work with real CaloTowers from Jeremy Mans

#include <vector>
class ProtoJet;
#include "DataFormats/JetObjects/interface/CaloJetCollectionfwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

void MakeCaloJet(const CaloTowerCollection &ctc, const std::vector<ProtoJet>& protoJets, CaloJetCollection &caloJets);

#endif
