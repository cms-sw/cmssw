#include "DQMOffline/Trigger/interface/CutValues.h"





void CutValues::setEBHighNrgy(int inputCutMask)
{
  cutMask = inputCutMask;
  minEtCut = 30.; //from 20
  minEtaCut = 0.;
  maxEtaCut = 1.442; //now 1.442 rather than 1.4442 
  rejectCracks = true;
  //ep, not defined
  minEpInCut = 0.5;
  maxEpInCut = 3.0;
  epInReleaseEtCut=150.;
  //robust cuts, same as v11
  maxDEtaInCut = 0.005;
  maxDPhiInCut = 0.09;
  maxHademCut = 0.05;
  minSigmaEtaEtaCut = -999.;
  maxSigmaEtaEtaCut = 0.011;
  //misc other cuts, not defined
  minEpOutCut = 0.5;
  maxEpOutCut = 3.;
  maxDPhiOutCut = 0.2;
  minInvEInvPCut = -0.05;
  maxInvEInvPCut = 0.1;
  minBremFracCut = 0.;
  minE9E25Cut = 0.2;
  minSigmaPhiPhiCut = -999.;
  maxSigmaPhiPhiCut = 0.1;
  //isol Em Clus - not defined
  minIsolEmConstCut = 6.;
  isolEmGradCut = 0.01;
  //isol EmRec Hit - not defined
  minIsolEmRecHitConstCut = 8.;
  isolEmRecHitGradCut = 0.025;
  //isol Em+had, new for v12
  minIsolEmHadDepth1ConstCut =5;
  isolEmHadDepth1GradCut= 0.02;
  //isol had, no defined for v12
  minIsolHadConstCut = 4.;
  isolHadGradCut = 0.005;
  //isol had depth 2, only applicable for endcap
  minIsolHadDepth2ConstCut =1;
  isolHadDepth2GradCut= 0.005;
  //pt tracks -change for v12
  minIsolPtTrksConstCut = 7.5;//
  isolPtTrksGradCut =  0.; //
  //nr tracks, not defined for v12
  minIsolNrTrksConstCut = 4;
}



void CutValues::setEBPreSel(int inputCutMask)
{
  cutMask = inputCutMask;
  minEtCut = 30.;
  minEtaCut = 0.;
  maxEtaCut = 1.442;
  rejectCracks = true;
  minEpInCut = 0.35;
  maxEpInCut = 3.0;
  epInReleaseEtCut=150.;
  maxDEtaInCut = 0.02;
  maxDPhiInCut = 0.1;
  maxHademCut = 0.2;

}

void CutValues::setEEPreSel(int inputCutMask)
{
  cutMask = inputCutMask;
  minEtCut = 30.;
  minEtaCut = 1.560;
  maxEtaCut = 2.5;
  rejectCracks = true;
  minEpInCut = 0.35;
  maxEpInCut = 5.0;
  epInReleaseEtCut=150.;
  maxDEtaInCut = 0.02;
  maxDPhiInCut = 0.1;
  maxHademCut = 0.2;

}

void CutValues::setEEHighNrgy(int inputCutMask)
{
  cutMask = inputCutMask;
  minEtCut = 30.; //from 20
  minEtaCut = 1.560;
  maxEtaCut = 2.5;
  rejectCracks = true;
  minEpInCut = 0.5;
  maxEpInCut = 5.0;
  epInReleaseEtCut=150.;
  maxDEtaInCut = 0.007;
  maxDPhiInCut = 0.09; //from 0.92
  maxHademCut = 0.1; //from 0.1
  minEpOutCut = 0.5;
  maxEpOutCut = 3.;
  maxDPhiOutCut = 0.2;
  minInvEInvPCut = -0.05;
  maxInvEInvPCut = 0.1;
  minBremFracCut = 0.;
  minE9E25Cut = 0.2;
  minSigmaEtaEtaCut = -999.;
  maxSigmaEtaEtaCut = 0.0275;
  minSigmaPhiPhiCut = -999.;
  maxSigmaPhiPhiCut = 0.1;
  minIsolEmConstCut = 6.;
  isolEmGradCut = 0.01;
  minIsolEmRecHitConstCut = 10.;
  isolEmRecHitGradCut = 0.025;
  minIsolHadConstCut = 4.;
  isolHadGradCut = 0.005;

  minIsolHadDepth2ConstCut =1; //new
  isolHadDepth2GradCut= 0.005;//new
  minIsolEmHadDepth1ConstCut =4;//new
  isolEmHadDepth1GradCut= 0.04;//new

  minIsolPtTrksConstCut = 15.; //from zero
  isolPtTrksGradCut =  0.; //from 0.2
  minIsolNrTrksConstCut = 4;



}
