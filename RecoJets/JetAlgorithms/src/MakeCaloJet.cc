#include "RecoJets/JetAlgorithms/interface/MakeCaloJet.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/CaloObjects/interface/CaloTowerCollection.h"
#include "DataFormats/JetObjects/interface/CaloJet.h"
#include "DataFormats/JetObjects/interface/CaloJetCollection.h"

void MakeCaloJet(const CaloTowerCollection &ctc, const std::vector<ProtoJet>& protoJets, CaloJetCollection& caloJets){

   //Loop over the transient protoJets 
   for(std::vector<ProtoJet>::const_iterator i = protoJets.begin(); i != protoJets.end(); ++i){
      
     const std::vector<int> & indices = assignTowerIndices(ctc, i->getTowerList());
     //Make a CaloJet and add it to the JetCollection:
     const ProtoJet &p = *i;
     caloJets.push_back(CaloJet(p.px(), p.py(), p.pz(), p.e(), p.p(), p.pt(), p.et(), p.m(), p.phi(), p.eta(), p.y(), p.numberOfConstituents(), ctc, indices));
   }
};
