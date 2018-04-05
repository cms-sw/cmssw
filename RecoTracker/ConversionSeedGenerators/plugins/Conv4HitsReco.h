#ifndef Conv4HitsReco_h
#define Conv4HitsReco_h

#include <iostream>
#include <iomanip>
#include <cmath>

#include <TVector3.h>

class Conv4HitsReco {
public :

  //Needed for construction
  TVector3 vV;
  TVector3 hit4;
  TVector3 hit3;
  TVector3 hit2;
  TVector3 hit1;

  //Important quantities
  TVector3 v3minus4;
  TVector3 v1minus2;
  TVector3 vN;
  TVector3 vP;
  TVector3 vPminusN;
  TVector3 vPminusV;
  TVector3 vNminusV;
  TVector3 unitVn;
  TVector3 unitVp;
  double pn;
  double pPN;
  double nPN;
  double PN;
  double PN2;
  double pNV;
  double nNV;
  double _eta;
  double _pi;
  double _eta2;
  double _pi2;
  TVector3 vMMaxLimit;
  TVector3 vQMaxLimit;
  double qMaxLimit;
  double mMaxLimit;
  double qMinLimit;
  double mMinLimit;
  TVector3 plusCenter;
  TVector3 minusCenter;
  TVector3 convVtx;
  double plusRadius;
  double minusRadius;

  double iterationStopRelThreshold;
  int maxNumberOfIterations;
  double maxVtxDistance;
  double ptLegMinCut;
  double ptLegMaxCut;
  double ptPhotMaxCut;

  std::string qFromM_print(double m);
  double qFromM(double); //For debugging purposes

  //Elements of the linearized system matrix
  double Tq, Tm, T0;
  double Oq, Om, O0;
  int ComputeMaxLimits();
  int ComputeMinLimits();
  int GuessStartingValues(double&, double&);
  int mqFindByIteration(double&, double&);
  TVector3 GetIntersection(TVector3&, TVector3&, TVector3&, TVector3&);
  TVector3 GetPlusCenter(double &);
  TVector3 GetMinusCenter(double &);
  void SetPtLegMinCut(double val){ptLegMinCut = val;};
  void SetPtLegMaxCut(double val){ptLegMaxCut = val;};
  void SetPtPhotMaxCut(double val){ptPhotMaxCut = val;};
  void SetLinSystCoeff(double, double);
  void Set(double);
  void SetIterationStopRelThreshold(double val){iterationStopRelThreshold = val;};
  void SetMaxNumberOfIterations(int val){maxNumberOfIterations = val;};
  void SetMaxVtxDistance(int val){maxVtxDistance = val;};
  double GetDq();
  double GetDm();
  double GetPtFromParamAndHitPair(double &, double &);
  double GetPtPlusFromParam(double &);
  double GetPtMinusFromParam(double &);
  TVector3 GetConvVertexFromParams(double &, double &);
  int IsNotValidForPtLimit(double, double, double, double);
  int IsNotValidForVtxPosition(double&);

  int ConversionCandidate(TVector3&, double&, double&);
  void Dump();

  Conv4HitsReco(TVector3 &, TVector3 &, TVector3 &, TVector3 &, TVector3 &);
  ~Conv4HitsReco();

};

#endif

#ifdef Conv4HitsReco_cxx
Conv4HitsReco::Conv4HitsReco(TVector3 & vPhotVertex, TVector3 & h1, TVector3 & h2, TVector3 & h3, TVector3 & h4)
{

  vV = vPhotVertex;
  hit4 = h4;
  hit3 = h3;
  hit2 = h2;
  hit1 = h1;

  //Let's stay in the transverse plane...
  vV.SetZ(0.);
  hit4.SetZ(0.);
  hit3.SetZ(0.);
  hit2.SetZ(0.);
  hit1.SetZ(0.);

  //Filling important quantities
  vN.SetXYZ(0.5*hit4.X()+0.5*hit3.X(),0.5*hit4.Y()+0.5*hit3.Y(),0.5*hit4.Z()+0.5*hit3.Z());
  vP.SetXYZ(0.5*hit2.X()+0.5*hit1.X(),0.5*hit2.Y()+0.5*hit1.Y(),0.5*hit2.Z()+0.5*hit1.Z());
  vPminusN=vP-vN;
  vPminusV=vP-vV;
  vNminusV=vN-vV;
  v3minus4=hit3-hit4;
  v1minus2=hit1-hit2;
  unitVn=v3minus4.Orthogonal().Unit();
  unitVp=v1minus2.Orthogonal().Unit();
  pPN = unitVp*vPminusN;
  nPN = unitVn*vPminusN;
  pNV = unitVp*vNminusV;
  nNV = unitVn*vNminusV;
  pn = unitVp*unitVn;
  PN = vPminusN.Mag();
  PN2 = vPminusN.Mag2();

  _eta=0.5*(hit3-hit4).Mag();
  _pi=0.5*(hit1-hit2).Mag();
  _eta2=_eta*_eta;
  _pi2=_pi*_pi;

  //Default values for parameters
  SetIterationStopRelThreshold(0.0005);
  SetMaxNumberOfIterations(10);
  SetPtLegMinCut(0.2);
  SetPtLegMaxCut(20.);
  SetPtPhotMaxCut(30.);
  SetMaxVtxDistance(20.);

}

Conv4HitsReco::~Conv4HitsReco(){
  
  //  std::cout << " Bye..." << std::endl;

}

#endif // #ifdef Conv4HitsReco_h
