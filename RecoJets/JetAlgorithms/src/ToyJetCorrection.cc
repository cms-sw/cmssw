/* Template algorithm to correct jet
    F.Ratnikov (UMd)
    Mar 2, 2006
*/

#include "DataFormats/JetReco/interface/CaloJet.h"

#include "RecoJets/JetAlgorithms/interface/ToyJetCorrection.h"

using namespace std;

CaloJet ToyJetCorrection::applyCorrection (const CaloJet& fJet) {
  CommonJetData common (fJet.getPx()*mScale, fJet.getPy()*mScale, fJet.getPz()*mScale, 
			fJet.getE()*mScale, fJet.getP()*mScale, fJet.getPt()*mScale, fJet.getEt()*mScale, fJet.getM()*mScale, 
			fJet.getPhi(), fJet.getEta(), fJet.getY(), 
			fJet.getNConstituents());
  CaloJet result (common, fJet.getSpecific (), fJet.getTowerIndices());
  return result;
}
