#define Conv4HitsReco_cxx
#include "Conv4HitsReco.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>
// 1 -> 4
// 2 -> 3
// 3 -> 2
// 4 -> 1

double Conv4HitsReco::GetDm(){

  return (Tq*O0-T0*Oq)/(Tq*Om-Tm*Oq); //dm
  
}

double Conv4HitsReco::GetDq(){

  return (Tm*O0-T0*Om)/(Tm*Oq-Tq*Om); //dq
  
}

void Conv4HitsReco::SetLinSystCoeff(double m, double q){
  
  // dq*Tq + dm*Tm = T0 // Tangent condition
  // dq*Oq + dm*Om = O0 // Orthogonality condition

  double sqrtEta2mm = sqrt(_eta2+m*m);
  double sqrtPi2qq = sqrt(_pi2+q*q);

  double signT = 1.;

  Tq = -2.*pPN+2.*m*pn+signT*2.*q*sqrtEta2mm/sqrtPi2qq;
  Tm =  2.*nPN+2.*q*pn+signT*2.*m*sqrtPi2qq/sqrtEta2mm;
  T0 = PN2-_eta2-_pi2-2.*q*m*pn+2.*q*pPN-2.*m*nPN-signT*2.*sqrtEta2mm*sqrtPi2qq;
  
  TVector3 vQminusM = q*unitVp - m*unitVn + vPminusN;
  double QM = vQminusM.Mag();
  double pQM = unitVp*vQminusM;
  double nQM = unitVn*vQminusM;
  double NVQM = vNminusV*vQminusM;
  
  double signO = 1.;

  Oq = sqrtEta2mm*pQM/QM+m*pn+pNV;
  Om = m*QM/sqrtEta2mm-signO*sqrtEta2mm*nQM/QM+nQM-nNV-m;
  O0 = -signO*sqrtEta2mm*QM-m*nQM-NVQM;

}

std::string Conv4HitsReco::qFromM_print(double m){
  std::stringstream ss;
  TVector3 vPminusM = vPminusN - m*unitVn;
  double m2 = m*m;
  double nPM = unitVn*vPminusM;
  double pPM = unitVp*vPminusM;
  double NVPM = vNminusV*vPminusM;

  double alpha = (m*pn+pNV)*(m*pn+pNV)-_eta2-m2;
  double beta = m2*pn*nPM+m*pn*NVPM+m*nPM*pNV-pPM*(_eta2+m2)+pNV*NVPM;
  double gamma = m2*nPM*nPM+NVPM*NVPM+2.*m*nPM*NVPM-vPminusM.Mag2()*(_eta2+m2);

  double disc = sqrt(beta*beta - alpha*gamma);

  double q01 = (-beta+disc)/alpha;
  double q02 = (-beta-disc)/alpha;


  ss << " m: " << m << " q01: " << std::setw(20) << q01 << " q02: "<< std::setw(20) << q02 << "/n";  
  return ss.str();
}

double Conv4HitsReco::qFromM(double m){

  LogDebug("Conv4HitsReco")<<qFromM_print(m);
  return 0.;

}

TVector3 Conv4HitsReco::GetPlusCenter(double &radius){
  radius = plusRadius;
  return plusCenter;
}

TVector3 Conv4HitsReco::GetMinusCenter(double &radius){
  radius = minusRadius;
  return minusCenter;
}

//
//Point of intersection between two lines (each identified by a vector and a point)
TVector3 Conv4HitsReco::GetIntersection(TVector3& V1, TVector3& p1, TVector3& V2, TVector3& p2){

  TVector3 v1 = V1.Unit();
  TVector3 v2 = V2.Unit();
  
  double v1v2=v1*v2;
  return v2*((p1 - p2)*(v1v2*v1-v2)/(v1v2*v1v2-1.))+p2;

}

double Conv4HitsReco::GetPtFromParamAndHitPair(double &m, double &eta){

  return 0.01*0.3*3.8*sqrt(m*m+eta*eta);

}

double Conv4HitsReco::GetPtMinusFromParam(double &m){

  return GetPtFromParamAndHitPair(m, _eta);

}

double Conv4HitsReco::GetPtPlusFromParam(double &q){

  return GetPtFromParamAndHitPair(q, _pi);

}

TVector3 Conv4HitsReco::GetConvVertexFromParams(double &m, double &q){

  TVector3 unitVQminusM = (plusCenter-minusCenter).Unit();
  TVector3 vtxViaPlus = vP+q*unitVp-plusRadius*unitVQminusM;
  TVector3 vtxViaMinus = vN+m*unitVn+minusRadius*unitVQminusM;

  //  return 0.5*(vN+m*unitVn+m*unitVQminusM+vP+q*unitVp-q*unitVQminusM);
  LogDebug("Conv4HitsReco")  << ">>>>>>>> Conversion vertex computed via Plus pair\n"
			     << vtxViaPlus.x() << "," << vtxViaPlus.y() << "," << vtxViaPlus.z()
			     << ">>>>>>>> Conversion vertex computed via Minus pair\n"
			     << vtxViaMinus.x() << "," << vtxViaMinus.y() << "," << vtxViaMinus.z();

  return 0.5*(vtxViaPlus+vtxViaMinus);

}

int Conv4HitsReco::ConversionCandidate(TVector3 &vtx, double &ptplus, double &ptminus){

  double m;
  double q;
  int nits = 0;

  int isNotValidBefore = ComputeMaxLimits()+ComputeMinLimits();
  int isNotValidAfter = 0;
  if ( ! isNotValidBefore ){

    GuessStartingValues(m, q);
    nits = mqFindByIteration(m, q);
    
    if ( q > qMaxLimit || q < qMinLimit ) {
      isNotValidAfter = 1;
      LogDebug("Conv4HitsReco")<< ">>>>>>>> quad result not valid for q: qMin= " << qMinLimit << " q= " << q << " qMax=  " << qMaxLimit << "\n";

    }
    if ( m > mMaxLimit || m < mMinLimit ) {
      isNotValidAfter = 1;
      LogDebug("Conv4HitsReco")<<">>>>>>>> quad result not valid for m: mMin= " << mMinLimit << " m= " << m << " mMax=  " << mMaxLimit << "\n";
    }
    
    ptminus = GetPtMinusFromParam(m);
    ptplus = GetPtPlusFromParam(q);
    minusCenter = vN+m*unitVn; 
    minusRadius = (hit4-minusCenter).Mag();
    plusCenter = vP+q*unitVp;
    plusRadius = (hit1-plusCenter).Mag();
    convVtx = GetConvVertexFromParams(m, q);
    vtx = convVtx;

  }

  if ( isNotValidBefore ) return 0;
  if ( IsNotValidForPtLimit(m, q, ptLegMinCut, ptLegMaxCut) ) return -1000;
  if ( IsNotValidForVtxPosition(maxVtxDistance) ) return -2000;
  if ( isNotValidAfter ) return -1*nits;
  return nits;

}

int Conv4HitsReco::mqFindByIteration(double &m, double &q){

  int maxIte = maxNumberOfIterations;
  double err = iterationStopRelThreshold;
  double edm = 1.;
  double edq = 1.;
  int i = 0;
  while ( ((edq > err) || (edm > err)) && (i < maxIte) ){
    SetLinSystCoeff(m,q);
    double dm = GetDm();
    double dq = GetDq();
    /*    
	  while( m+dm > mMaxLimit || m+dm < mMinLimit || q+dq > qMaxLimit || q+dq < qMinLimit ){

	  LogDebug("Conv4HitsReco")<<">>>>>>>> Going outside limits, reducing increments \n";
	  dm=dm/2.;
	  dq=dq/2.;
	  }
    */
    m+=dm;
    q+=dq;
    edm = fabs(dm/m);
    edq = fabs(dq/q);
    LogDebug("Conv4HitsReco")<< ">>>>>>>> Iteration " << i << " m: " << m << " q: " << q << " dm: " << dm << " dq: " << dq << " edm: " << edm << " edq: " << edq << "\n";
    i++;
  }

  return i;

}

int Conv4HitsReco::ComputeMaxLimits(){

  // Q max limit
  TVector3 vVminusHit2Orthogonal = (vV-hit2).Orthogonal();
  TVector3 medianPointHit2V = 0.5*(vV+hit2);
  vQMaxLimit = GetIntersection(vVminusHit2Orthogonal, medianPointHit2V, unitVp, vP);
  qMaxLimit = (vQMaxLimit-vP).Mag();

  // M max limit
  TVector3 vVminusHit3Orthogonal = (vV-hit3).Orthogonal();
  TVector3 medianPointHit3V = 0.5*(vV+hit3);
  vMMaxLimit = GetIntersection(vVminusHit3Orthogonal, medianPointHit3V, unitVn, vN);
  mMaxLimit = (vMMaxLimit-vN).Mag();

  LogDebug("Conv4HitsReco")<< " >>>>>> Limits: qMax= " << qMaxLimit << " ==>pt " << GetPtFromParamAndHitPair(qMaxLimit, _pi)  << " mMax= " << mMaxLimit << " ==>pt " <<GetPtFromParamAndHitPair(mMaxLimit, _eta) << "\n"; 

  return IsNotValidForPtLimit(mMaxLimit, qMaxLimit, ptLegMinCut, 100000.); //Max limit not applied here

}

int Conv4HitsReco::IsNotValidForPtLimit(double m, double q, double ptmin, double ptmax){

  if ( GetPtFromParamAndHitPair(q, _pi) < ptmin ) return 1;
  if ( GetPtFromParamAndHitPair(m, _eta) < ptmin ) return 1;
  if ( GetPtFromParamAndHitPair(q, _pi) > ptmax ) return 1;
  if ( GetPtFromParamAndHitPair(m, _eta) > ptmax ) return 1;
  return 0;

}

int Conv4HitsReco::IsNotValidForVtxPosition(double& maxDist){

  TVector3 hitAve = 0.25*(hit1+hit2+hit3+hit4); 
  if ( (convVtx-hitAve).Mag() > maxDist ) return 1;
  return 0;

}

int Conv4HitsReco::ComputeMinLimits(){

  //Evaluate if quad is valid and compute min limits 
  if ( ((vV-vQMaxLimit).Cross(vMMaxLimit-vQMaxLimit)).Z() > 0. ){
    //
    //Quad is invalid
    LogDebug("Conv4HitsReco")<< " >>>>>> Quad is invalid\n"; 
    return 1;
  } else {
    //
    // Compute q and m Min limits
    TVector3 vQMinLimit = GetIntersection(v1minus2, vMMaxLimit, unitVp, vP);
    qMinLimit = (vQMinLimit-vP)*unitVp;
    TVector3 vMMinLimit = GetIntersection(v3minus4, vQMaxLimit, unitVn, vN);
    mMinLimit = (vMMinLimit-vN)*unitVn;
    if ( mMinLimit > mMaxLimit || qMinLimit > qMaxLimit ){
      LogDebug("Conv4HitsReco")<< " >>>>>> Quad is invalid. qMin= " << qMinLimit << " mMin= " << mMinLimit << "\n"; 
      return 2;
    }
    if ( IsNotValidForPtLimit(mMinLimit, qMinLimit, -1000., ptLegMaxCut) ){ //Min limit not applied here
      return 2;
    }

    LogDebug("Conv4HitsReco")<< " >>>>>> Quad is valid. qMin= " << qMinLimit << " mMin= " << mMinLimit << "\n"; 
    return 0;
  }

}

int Conv4HitsReco::GuessStartingValues(double &m, double &q){

  /*
  m = 0.5*(mMinLimit+mMaxLimit);
  q = 0.5*(qMinLimit+qMaxLimit);
  */

  m = mMaxLimit;
  q = qMaxLimit;

  LogDebug("Conv4HitsReco")<< " >>>>>> Starting values: q= " << q << " m= " << m << "\n"; 

  return 0;

}


void Conv4HitsReco::Dump(){

  LogDebug("Conv4HitsReco")
    << " ======================================= " << "\n"
    << " Photon Vertex: "
    << vV.x() << "," << vV.y() << "," << vV.z()
    << " Hit1: "
    << hit1.x() << "," << hit1.y() << "," << hit1.z()
    << " Hit2: "
    << hit2.x() << "," << hit2.y() << "," << hit2.z()
    << " Hit3: "
    << hit3.x() << "," << hit3.y() << "," << hit3.z()
    << " Hit4: "
    << hit4.x() << "," << hit4.y() << "," << hit4.z()
    << " N: "
    << vN.x() << "," << vN.y() << "," << vN.z()
    << " P: "
    << vP.x() << "," << vP.y() << "," << vP.z()
    << " P-N: "
    << vPminusN.x() << "," << vP.y() << "," << vP.z()
    << " n: "
    << unitVn.x() << "," << unitVn.y() << "," << unitVn.z()
    << " p: "
    << unitVp.x() << "," << unitVp.y() << "," << unitVp.z()
    << " eta: " << _eta << " pi: " << _pi << "\n";

}
