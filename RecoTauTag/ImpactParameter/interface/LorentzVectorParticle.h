#ifndef RecoTauTag_ImpactParameter_LorentzVectorParticle_h
#define RecoTauTag_ImpactParameter_LorentzVectorParticle_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

#include "RecoTauTag/ImpactParameter/interface/Particle.h"
#include "TString.h"
#include "TMath.h"
#include "TLorentzVector.h"

namespace tauImpactParameter {

class LorentzVectorParticle : public Particle {
 public:
  enum LorentzandVectorPar{vx=0,vy,vz,px,py,pz,m,NLorentzandVertexPar,E=-1,p=-2,pt=-3};// Lorentez+vertex parameters
  enum VertexInfo{NVertex=3};
  LorentzVectorParticle();
  LorentzVectorParticle(const TVectorT<double>& par, const TMatrixTSym<double>& cov, int pdgid, double charge, double b);
  virtual ~LorentzVectorParticle(){};

  static TString name(int i);
  virtual int nParameters() const { return NLorentzandVertexPar; }
  virtual double parameter(int i) const {
    double particle_px = Particle::parameter(px);
    double particle_py = Particle::parameter(py);
    double particle_pz = Particle::parameter(pz);    
    double particle_m  = Particle::parameter(m);
    if(i==E)  return sqrt(particle_px*particle_px + particle_py*particle_py + particle_pz*particle_pz + particle_m*particle_m);
    if(i==p)  return sqrt(particle_px*particle_px + particle_py*particle_py + particle_pz*particle_pz);
    if(i==pt) return sqrt(particle_px*particle_px + particle_py*particle_py);
    return Particle::parameter(i);
  }
  virtual double mass() const { return parameter(m); }
  TLorentzVector p4() const { return TLorentzVector(parameter(px),parameter(py),parameter(pz),parameter(E)); }
  TVector3 vertex() const { return TVector3(parameter(vx),parameter(vy),parameter(vz)); }
  TMatrixTSym<double> vertexCov() const {
    TMatrixTSym<double> vcov(NVertex);
    for(int i=0;i<NVertex;i++){
      for(int j=0;j<NVertex;j++){vcov(i,j)=covariance(i,j);}
    }
    return vcov; 
  }
};

}
#endif


