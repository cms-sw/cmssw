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
#include "RecoEgamma/EgammaHFProducers/interface/HFEGammaSLCorrector.h"

using namespace std;
using namespace reco;

HFRecoEcalCandidateAlgo::HFRecoEcalCandidateAlgo(bool correct,double e9e25Cut,double intercept2DCut,double intercept2DSlope,
						 const std::vector<double>& e1e9Cut,
						 const std::vector<double>& eCOREe9Cut,
						 const std::vector<double>& eSeLCut,
						 int era,
						 bool correctForPileup,
						 const std::vector<double>& PileupSlopes,
						 const std::vector<double>& PileupIntercepts
) :

  m_correct(correct), 
  m_e9e25Cut(e9e25Cut),
  m_intercept2DCut(intercept2DCut),
  m_intercept2DSlope(intercept2DSlope),
  m_e1e9Cuthi(e1e9Cut[1]),
  m_eCOREe9Cuthi(eCOREe9Cut[1]),
  m_eSeLCuthi(eSeLCut[1]),
  m_e1e9Cutlo(e1e9Cut[0]),
  m_eCOREe9Cutlo(eCOREe9Cut[0]),
  m_eSeLCutlo(eSeLCut[0]),
  m_era(era),
  m_correctForPileup(correctForPileup),
  m_PileupSlopes(PileupSlopes),
  m_PileupIntercepts(PileupIntercepts){

}

RecoEcalCandidate HFRecoEcalCandidateAlgo::correctEPosition(const SuperCluster& original , const HFEMClusterShape& shape,int nvtx) {
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
  if(m_correctForPileup){
    std::vector<double> m=m_PileupSlopes;

    std::vector<double> b=m_PileupIntercepts;

    double etabounds[30]={2.853,2.964,3.139,3.314,3.489,3.664,3.839,4.013,4.191,4.363,4.538,4.716,4.889,5.191};
   
    int ieta=99;
    for (int kk=0;kk<12;kk++){
     
      if((fabs(corEta) < etabounds[kk+1])&&(fabs(corEta) > etabounds[kk])){
	ieta = (corEta > 0)?(kk+29):(-kk-29);
      }
    }
    int neta=99;
    if(ieta<0)ieta=ieta+39;
    if(ieta>0)ieta=ieta-20;
    if((neta>=0)&&(neta<=19)){
      corEnergy=(m[neta]*1.0*(nvtx-1)+b[neta]*1.0)*corEnergy*1.0;
    }
  }//end vtx cor
  RecoEcalCandidate corCand(0,
			    math::XYZTLorentzVector(corPx,corPy,corPz,corEnergy),
			    math::XYZPoint(0,0,0));
  
  
  
  
  
  
  return corCand;
}

void HFRecoEcalCandidateAlgo::produce(const edm::Handle<SuperClusterCollection>& SuperClusters,
				      const HFEMClusterShapeAssociationCollection& AssocShapes,
				      RecoEcalCandidateCollection& RecoECand,
				      int nvtx) {
  
  
  
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
      theCand=correctEPosition(supClus,clusShape,nvtx);

    double e9e25=clusShape.eLong3x3()/clusShape.eLong5x5();
    double e1e9=clusShape.eLong1x1()/clusShape.eLong3x3();
    double eSeL=hf_egamma::eSeLCorrected(clusShape.eShort3x3(),clusShape.eLong3x3(),4);
    double var2d=(clusShape.eCOREe9()-(eSeL*m_intercept2DSlope));
 
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
 


