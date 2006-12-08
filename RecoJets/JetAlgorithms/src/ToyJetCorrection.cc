/* Template algorithm to correct jet
    F.Ratnikov (UMd)
    Mar 2, 2006
*/

#include "DataFormats/JetReco/interface/CaloJet.h"

#include "RecoJets/JetAlgorithms/interface/ToyJetCorrection.h"

using namespace std;
using namespace reco;

CaloJet ToyJetCorrection::applyCorrection (const CaloJet& fJet) {
  Jet::LorentzVector newP4 (fJet.px()*mScale, fJet.py()*mScale, 
			    fJet.pz()*mScale, fJet.energy()*mScale);
  CaloJet result (newP4, fJet.getSpecific (), fJet.getGonstituents());
  return result;
}
