//  MakeCaloJet.cc
//  Revision History:  R. Harris 10/19/05  Modified to work with real CaloTowers from Jeremy Mans
//

#include "RecoJets/JetAlgorithms/interface/MakeCaloJet.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/JetObjects/interface/CaloJet.h"
#include "DataFormats/JetObjects/interface/CaloJetCollection.h"

std::vector<CaloTowerDetId> assignTowerIndices(const CaloTowerCollection &caloTowerColl, const std::vector<const CaloTower *> & towerList);
void MakeCaloJet(const CaloTowerCollection &ctc, const std::vector<ProtoJet>& protoJets, CaloJetCollection& caloJets){

   //Loop over the transient protoJets 
   for(std::vector<ProtoJet>::const_iterator i = protoJets.begin(); i != protoJets.end(); ++i){

     const std::vector<CaloTowerDetId> & indices = assignTowerIndices(ctc, i->getTowerList());

     //Make a CaloJet and add it to the JetCollection:
     const ProtoJet &p = *i;
     caloJets.push_back(CaloJet(p.px(), p.py(), p.pz(), p.e(), p.p(), p.pt(), p.et(), p.m(), p.phi(), p.eta(), p.y(), p.numberOfConstituents(), ctc, indices));
   }
};

std::vector<CaloTowerDetId> assignTowerIndices(const CaloTowerCollection &caloTowerColl, const std::vector<const CaloTower *> & towerList) {

  //Define the array of indices:
  std::vector<CaloTowerDetId> towerIndices;
  
  //Save list of CaloTowerDetIDs of the CaloTowers that make up the jet.
  for(std::vector<const CaloTower *>::const_iterator i = towerList.begin(); i != towerList.end(); ++i) {  
     towerIndices.push_back( (*i)->id() );
  }
  return towerIndices;
}//end of function  
