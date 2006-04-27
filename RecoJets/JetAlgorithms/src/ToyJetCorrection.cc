/* Template algorithm to correct jet
    F.Ratnikov (UMd)
    Mar 2, 2006
*/

#include "DataFormats/JetReco/interface/CaloJet.h"

#include "RecoJets/JetAlgorithms/interface/ToyJetCorrection.h"

using namespace std;

CaloJet ToyJetCorrection::applyCorrection (const CaloJet& fJet) {
  CommonJetData common (fJet.px()*mScale, fJet.py()*mScale, fJet.pz()*mScale, 
			fJet.energy()*mScale, fJet.nConstituents());
  CaloJet result (common, fJet.getSpecific (), fJet.getTowerIndices());
  return result;
}
