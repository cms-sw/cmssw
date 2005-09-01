#ifndef JetAlgorithms_MakeCaloJet_h
#define JetAlgorithms_MakeCaloJet_h

#include <vector>
class ProtoJet;
#include "DataFormats/JetObjects/interface/CaloJetCollectionfwd.h"
#include "DataFormats/CaloObjects/interface/CaloTowerCollectionfwd.h"

void MakeCaloJet(const CaloTowerCollection &ctc, const std::vector<ProtoJet>& protoJets, CaloJetCollection &caloJets);

#endif // JetAlgorithms_MakeCaloJet_h
