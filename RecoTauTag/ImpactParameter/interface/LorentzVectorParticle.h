/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#ifndef LorentzVectorParticle_h
#define LorentzVectorParticle_h

#include "RecoTauTag/ImpactParameter/interface/Particle.h"
#include "TString.h"
#include "TMath.h"
#include "TLorentzVector.h"

class LorentzVectorParticle : public Particle {
 public:
  enum LorentzandVectorPar{vx=0,vy,vz,px,py,pz,m,NLorentzandVertexPar,E=-1,p=-2,pt=-3};// Lorentez+vertex parameters
  enum VertexInfo{NVertex=3};
  LorentzVectorParticle();
  LorentzVectorParticle(TMatrixT<double> par_, TMatrixTSym<double> cov_, int pdgid_, double charge_, double b_);
  virtual ~LorentzVectorParticle(){};

  static TString Name(int i);
  virtual int NParameters(){return NLorentzandVertexPar;}
  virtual double Parameter(int i){
    if(i==E)  return sqrt(pow(Particle::Parameter(m),2.0)+pow(Particle::Parameter(px),2.0)+pow(Particle::Parameter(py),2.0)+pow(Particle::Parameter(pz),2.0)); 
    if(i==p)  return sqrt(pow(Particle::Parameter(px),2.0)+pow(Particle::Parameter(py),2.0)+pow(Particle::Parameter(pz),2.0));
    if(i==pt) return sqrt(pow(Particle::Parameter(px),2.0)+pow(Particle::Parameter(py),2.0));
    return Particle::Parameter(i);
  }
  virtual double Mass(){return Parameter(m);}
  TLorentzVector LV(){return TLorentzVector(Parameter(px),Parameter(py),Parameter(pz),Parameter(E));}
  TVector3 Vertex(){return TVector3(Parameter(vx),Parameter(vy),Parameter(vz));}
  TMatrixTSym<double> VertexCov(){
    TMatrixTSym<double> vcov(NVertex);
    for(int i=0;i<NVertex;i++){
      for(int j=0;j<NVertex;j++){vcov(i,j)=Covariance(i,j);}
    }
    return vcov; 
  }
};
#endif


