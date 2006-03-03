//  MakeCaloJet.cc
//  Revision History:  R. Harris 10/19/05  Modified to work with real CaloTowers from Jeremy Mans
//

#include "RecoJets/JetAlgorithms/interface/MakeCaloJet.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/JetObjects/interface/CaloJet.h"
#include "DataFormats/JetObjects/interface/CaloJetCollection.h"

#include <vector>

using namespace std;

namespace {
  bool makeSpecific (const ProtoJet& fProtojet, 
		     const CaloTowerCollection &fCaloTowerColl, 
		     CaloJet::Specific* fJetSpecific) {
    if (!fJetSpecific) return false;
    
    // 1.- Loop over the tower Ids, 
    // 2.- Get the corresponding CaloTower
    // 3.- Calculate the different CaloJet specific quantities
    double energy = fProtojet.e();
    
    vector<double> eECal_i;
    vector<double> eHCal_i;
    double eInHad = 0.;
    double eInEm = 0.;
    double eInHO = 0.;
    double eInHB = 0.;
    double eInHF = 0.;
    double eInHE = 0.;
    
    vector<CaloTowerDetId> ids = fProtojet.towerIds();
    for(vector<CaloTowerDetId>::const_iterator i = ids.begin(); i != ids.end(); ++i) {
      const CaloTower* aTower =  &*fCaloTowerColl.find(*i);
      //Array of energy in EM Towers:
      eECal_i.push_back(aTower->e_em());
      eInEm += aTower->e_em();
      //Array of energy in HCAL Towers:
      eHCal_i.push_back(aTower->e_had()); 
      eInHad += aTower->e_had();
      
      eInHO += aTower->e_outer();
      // have no data for eInHB eInHE eInHF
    }
    
    fJetSpecific->m_energyFractionInHO = eInHO / energy;
    fJetSpecific->m_energyFractionInHB = eInHB / energy;
    fJetSpecific->m_energyFractionInHE = eInHE / energy;
    fJetSpecific->m_energyFractionInHF = eInHF / energy;
    fJetSpecific->m_energyFractionInHCAL = eInHad / (eInHad + eInEm);
    fJetSpecific->m_energyFractionInECAL = eInEm / (eInHad + eInEm);
    
    //Sort the arrays
    sort(eECal_i.begin(), eECal_i.end(), greater<double>());
    sort(eHCal_i.begin(), eHCal_i.end(), greater<double>());
    
    //Highest value in the array is the first element of the array
    fJetSpecific->m_maxEInEmTowers = eECal_i.front(); 
    fJetSpecific->m_maxEInHadTowers = eHCal_i.front();
    
    //n90 using the sorted list
    fJetSpecific->m_n90 = 0;
    double ediff = (eInHad + eInEm) * 0.9;
    for (unsigned i = 0; i < fCaloTowerColl.size(); i++) {
      ediff = ediff - eECal_i[i] - eHCal_i[i];
      fJetSpecific->m_n90++;
      if (ediff <= 0) break; 
    }
    return true;
  }
}

void MakeCaloJet(const CaloTowerCollection &ctc, const std::vector<ProtoJet>& protoJets, CaloJetCollection& caloJets){
  
  //Loop over the transient protoJets 
  for(std::vector<ProtoJet>::const_iterator protojet = protoJets.begin(); protojet != protoJets.end(); ++protojet){
    //Make a CaloJet and add it to the JetCollection:
    CommonJetData common (protojet->px(), protojet->py(), protojet->pz(), 
			  protojet->e(), protojet->p(), protojet->pt(), protojet->et(), protojet->m(), 
			  protojet->phi(), protojet->eta(), protojet->y(), 
			  protojet->numberOfConstituents());
    CaloJet::Specific specific;
    makeSpecific (*protojet, ctc, &specific);
    caloJets.push_back (CaloJet (common, specific, protojet->towerIds()));
  }
}
