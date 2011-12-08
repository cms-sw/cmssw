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
						 bool correctForPileup) :

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
  m_correctForPileup(correctForPileup){

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
    double m[30]={-0.0036,
		  -0.0087,
		  -0.0049,
		  -0.0161,
		  -0.0072,
		  -0.0033,
		  -0.0066,
		  -0.0062,
		  -0.0045,
		  -0.0090,
		  -0.0056,
		  -0.0024,
		  -0.0064,
		  -0.0063,
		  -0.0078,
		  -0.0079,
		  -0.0075,
		  -0.0074,
		  0.0009,
		  -0.0180};
    double b[30]={1.0565,
		  1.0432,
		  1.0714,
		  1.1140,
		  1.0908,
		  1.0576,
		  1.0821,
		  1.0807,
		  1.0885,
		  1.1783,//end neg ieta
		  1.1570,
		  1.0631,
		  1.0401,
		  1.0803,
		  1.0506,
		  1.0491,
		  1.0235,
		  1.0643,
		  0.9910,
		  1.0489};
    double etabounds[30]={2.964,3.139,3.314,3.489,3.664,3.839,4.013,4.191,4.363,4.538,4.716};
   
    int ieta=32;
    for (int kk=0;kk<10;kk++){
      double sign=corEta*1.0/fabs(corEta);
      if((fabs(corEta) < etabounds[kk+1])&&(fabs(corEta) > etabounds[kk])){
	ieta = sign*(kk+30);
      }
    }
    if(ieta<0)ieta=ieta+39;
    if(ieta>0)ieta=ieta-20;

    corEnergy=(m[ieta]*(nvtx-1)+b[ieta])*corEnergy;
    //    corEnergy=(m[ieta]*(vtx-1)+1.0)*corEnergy;
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
 


