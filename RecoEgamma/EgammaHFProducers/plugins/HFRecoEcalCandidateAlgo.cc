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
#include "HFRecoEcalCandidateAlgo.h"
#include "HFEGammaSLCorrector.h"

using namespace std;
using namespace reco;

HFRecoEcalCandidateAlgo::HFRecoEcalCandidateAlgo(bool correct,double e9e25Cut,double intercept2DCut,double intercept2DSlope,
						 const std::vector<double>& e1e9Cut,
						 const std::vector<double>& eCOREe9Cut,
						 const std::vector<double>& eSeLCut,
						 const reco::HFValueStruct hfvv				
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
  m_era(4), 
  m_hfvv(hfvv)
{
  
}

RecoEcalCandidate HFRecoEcalCandidateAlgo::correctEPosition(const SuperCluster& original , const HFEMClusterShape& shape,int nvtx) {
 double corEta=original.eta();
  //piece-wise log energy correction to eta
  double logel=log(shape.eLong3x3()/100.0);
  double lowC= 0.00911;
  double hiC = 0.0193;
  double logSlope    = 0.0146;
  double logIntercept=-0.00988;
  double sgn=(original.eta()>0)?(1.0):(-1.0);
  if(logel<=1.25)corEta+=(lowC)*sgn;
  else if((logel>1.25)&&(logel<=2))corEta+=(logIntercept+logSlope*logel)*sgn;
  else if(logel>2)corEta+=hiC*sgn;
  //poly-3 cell eta correction to eta (de)
  double p0= 0.0125;
  double p1=-0.0475;
  double p2= 0.0732;
  double p3=-0.0506;
  double de= sgn*(p0+(p1*(shape.CellEta()))+(p2*(shape.CellEta()*shape.CellEta()))+(p3*(shape.CellEta()*shape.CellEta()*shape.CellEta())));
  corEta+=de;
 
  double corPhi=original.phi();
  //poly-3 cell phi correction to phi (dp)
  p0= 0.0115;
  p1=-0.0394;
  p2= 0.0486;
  p3=-0.0335;
  double dp =p0+(p1*(shape.CellPhi()))+(p2*(shape.CellPhi()*shape.CellPhi()))+(p3*(shape.CellPhi()*shape.CellPhi()*shape.CellPhi()));
  corPhi+=dp;
  
 
  double corEnergy= original.energy();
  //energy corrections
  //flat correction for using only long fibers
  double energyCorrect=0.7397;
  corEnergy= corEnergy/energyCorrect; 
  //corection based on ieta for pileup and general 
  //find ieta
  int ieta=0;
  double etabounds[30]={
    2.846,2.957,3.132,3.307,3.482,
    3.657,3.833,4.006,4.184,4.357,
    4.532,4.709,4.882,5.184};
  for (int kk=0;kk<=12;kk++){
    if((fabs(corEta) < etabounds[kk+1])&&(fabs(corEta) > etabounds[kk])){
      ieta = (corEta > 0)?(kk+29):(-kk-29);
    }
  }
  //check if ieta is in bounds, should be [-41, -29] U [29, 41]
  if (ieta != 0) corEnergy=(m_hfvv.PUSlope(ieta)*1.0*(nvtx-1)+m_hfvv.PUIntercept(ieta)*1.0)*corEnergy*1.0*m_hfvv.EnCor(ieta);

  //re-calculate the energy vector  
  double corPx=corEnergy*cos(corPhi)/cosh(corEta);
  double corPy=corEnergy*sin(corPhi)/cosh(corEta);
  double corPz=corEnergy*tanh(corEta);
  
  //make the candidate
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
 
