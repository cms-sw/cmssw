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

HFRecoEcalCandidateAlgo::HFRecoEcalCandidateAlgo(bool correct,double e9e25Cut,double intercept2DCut,
						 const std::vector<double>& e1e9Cut,
						 const std::vector<double>& eCOREe9Cut,
						 const std::vector<double>& eSeLCut) :

  m_correct(correct), 
  m_e9e25Cut(e9e25Cut),
  m_intercept2DCut(intercept2DCut),
  m_e1e9Cuthi(e1e9Cut[1]),
  m_eCOREe9Cuthi(eCOREe9Cut[1]),
  m_eSeLCuthi(eSeLCut[1]),
  m_e1e9Cutlo(e1e9Cut[0]),
  m_eCOREe9Cutlo(eCOREe9Cut[0]),
  m_eSeLCutlo(eSeLCut[0]){

}

RecoEcalCandidate HFRecoEcalCandidateAlgo::correctEPosition(const SuperCluster& original , const HFEMClusterShape& shape) {
  double energyCorrect=0.7397;//.7515;
  double etaCorrect=.00938422+0.00682824*sin(6.28318531*shape.CellEta());//.0144225-.00484597*sin(6.17851*shape.CellEta());//0.01139;
  double phiAmpCorrect=0.00644091;//-0.006483;
  double phiFreqCorrect=6.28318531;//6.45377;

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

    double e9e25=clusShape.eLong3x3()/clusShape.eLong5x5();
    double e1e9=clusShape.eLong1x1()/clusShape.eLong3x3();
    // EMID cuts...  
    //if((clusShape.e9e25()> m_e9e25Cut)&&((clusShape.eCOREe9()-(clusShape.eSeL()*1.125)) > m_intercept2DCut)){
    //  if((e9e25> m_e9e25Cut)&&((clusShape.eCOREe9()-(clusShape.eSeL()*1.125)) > m_intercept2DCut)){
    double var2d=(clusShape.eCOREe9()-(clusShape.eSeL()*1.125));
 
    bool isAcceptable=true;
    isAcceptable=isAcceptable && (e9e25> m_e9e25Cut);
    isAcceptable=isAcceptable && (var2d > m_intercept2DCut);
    isAcceptable=isAcceptable && ((e1e9< m_e1e9Cuthi)&&(e1e9> m_e1e9Cutlo));
    isAcceptable=isAcceptable && ((clusShape.eCOREe9()< m_eCOREe9Cuthi)&&(clusShape.eCOREe9()>  m_eCOREe9Cutlo));
    isAcceptable=isAcceptable && ((clusShape.eSeL()<m_eSeLCuthi)&&(clusShape.eSeL()>  m_eSeLCutlo));
   
    
    if(isAcceptable){

      theCand.setSuperCluster(theClusRef);
      RecoECand.push_back(theCand);
    }
  }
}
 

//&&((clusShape.e1e9()< m_e1e9Cuthi)&&(clusShape.e1e9()> m_e1e9Cutlo))&&((clusShape.eCOREe9()< m_eCOREe9Cuthi)&&(clusShape.eCOREe9()>  m_eCOREe9Cutlo))&&((clusShape.eSeL()<m_eSeLCuthi)&&(clusShape.eSeL()>  m_eSeLCutlo))
