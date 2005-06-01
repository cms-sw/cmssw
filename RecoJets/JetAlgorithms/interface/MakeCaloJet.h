#ifndef JETALGORITHMS_MAKECALOJET_H
#define JETALGORITHMS_MAKECALOJET_H

#include <vector>
class ProtoJet;
#include "DataFormats/JetObjects/interface/CaloJetCollectionfwd.h"
#include "DataFormats/CaloObjects/interface/CaloTowerCollectionfwd.h"

void MakeCaloJet(const CaloTowerCollection &ctc, const std::vector<ProtoJet>& protoJets, CaloJetCollection &caloJets);

#endif // JETALGORITHMS_MAKECALOJET_H
