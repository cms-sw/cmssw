/** \class HFRecoEcalCandidateProducers
 *
 *  \author Kevin Klapoetke (Minnesota)
 *
 * $Id:
 *
 */

//#includes
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "CLHEP/Vector/LorentzVector.h"
#include <iostream>
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "RecoEgamma/EgammaHFProducers/interface/HFRecoEcalCandidateAlgo.h"
using namespace std;
using namespace reco;

HFRecoEcalCandidateAlgo::HFRecoEcalCandidateAlgo(bool correct,double e9e25Cut,double eCOREe9Cut,double eSeLCut) :
  m_correct(correct), 
  
  m_e9e25Cut(e9e25Cut),
  m_eCOREe9Cut(eCOREe9Cut),
  m_eSeLCut(eSeLCut){
}

RecoEcalCandidate HFRecoEcalCandidateAlgo::correctEPosition(const SuperCluster& original , const HFEMClusterShape& shape) {
  double energyCorrect=1.07;
  double etaCorrect=0.0184;
  double phiAmpCorrect=.00701641;
  double phiFreqCorrect=6.13245;

  double corEnergy= original.energy()/energyCorrect;
  double corEta=original.eta();
  corEta+=(original.eta()>0)?(etaCorrect):(-etaCorrect);
  double corPhi=original.phi()+phiAmpCorrect*sin(phiFreqCorrect*shape.CellPhi());
  double corPx=corEnergy*cos(corPhi)/cosh(corEta);
  double corPy=corEnergy*sin(corPhi)/cosh(corEta);
  double corPz=corEnergy*tanh(corEta);
    RecoEcalCandidate corCand(0,
			      math::XYZTLorentzVector(corPx,corPy,corPz,corEnergy),
			      math::XYZPoint(0,0,0));

 return corCand;
}

void HFRecoEcalCandidateAlgo::produce(const edm::Handle<SuperClusterCollection>& SuperClusters,
				      const HFEMClusterShapeAssociationCollection& AssocShapes,
				      RecoEcalCandidateCollection& RecoECand) {
  
  
  
  //get super's and cluster shapes and associations 
  for (unsigned int i=0; i < SuperClusters->size(); ++i) {
    const SuperCluster& supClus=(*SuperClusters)[i];    
    reco::SuperClusterRef theClusRef=edm::Ref<SuperClusterCollection>(SuperClusters,i);
  const HFEMClusterShapeRef clusShapeRef=AssocShapes.find(theClusRef)->val;
    const HFEMClusterShape& clusShape=*clusShapeRef;

    // basic candidate
    double px=supClus.energy()*cos(supClus.phi())/cosh(supClus.eta());
    double py=supClus.energy()*sin(supClus.phi())/cosh(supClus.eta());
    double pz=supClus.energy()*tanh(supClus.eta());
    RecoEcalCandidate theCand(0,
			      math::XYZTLorentzVector(px,py,pz,supClus.energy()),
			      math::XYZPoint(0,0,0));

    // correct it?
    if (m_correct)
      theCand=correctEPosition(supClus,clusShape);

    
    // cuts...  Cuts are still rough estimates and need further fine tuning
    if ((clusShape.e9e25()> m_e9e25Cut)&&(clusShape.eCOREe9()> m_eCOREe9Cut)&&(clusShape.eSeL()> m_eSeLCut)){
      theCand.setSuperCluster(theClusRef);
      RecoECand.push_back(theCand);
    }
  }
}
