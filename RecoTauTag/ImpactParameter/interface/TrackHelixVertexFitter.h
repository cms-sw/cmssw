#ifndef RecoTauTag_ImpactParameter_TrackHelixVertexFitter_h
#define RecoTauTag_ImpactParameter_TrackHelixVertexFitter_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

// system include files
#include <TMatrixT.h>
#include <TVectorT.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVectorD.h>

#include "RecoTauTag/ImpactParameter/interface/Particle.h"
#include "RecoTauTag/ImpactParameter/interface/TrackParticle.h"
#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"
#include "RecoTauTag/ImpactParameter/interface/ErrorMatrixPropagator.h"

namespace tauImpactParameter {

class TrackHelixVertexFitter{
 public:
  enum FreeVertexPar{x0=0,y0,z0,NFreeVertexPar};
  enum FreeTrackPar{kappa0=3,lambda0,phi0,NFreeTrackPar};
  enum ExtraPar{BField0=0,MassOffSet=1,NExtraPar=1};

  TrackHelixVertexFitter(const std::vector<TrackParticle>& particles, const TVector3& vguess);
  virtual ~TrackHelixVertexFitter();

   virtual bool fit()=0;
   virtual double updateChisquare(const TVectorT<double>& inpar);
   virtual double chiSquare(){return chi2_;}
   virtual double ndf(){return ndf_;}
   LorentzVectorParticle getMother(int pdgid);
   virtual std::vector<TrackParticle> getRefitTracks();
   std::vector<LorentzVectorParticle> getRefitLorentzVectorParticles();
   virtual TVector3 getVertex();
   virtual TMatrixTSym<double> getVertexError();

   static void computedxydz(const TVectorT<double>& inpar,int particle,double& kappa,double& lam,double& phi,double& x,double& y,double& z,double& s,double& dxy,double& dz);
   static TVectorT<double> computeLorentzVectorPar(const TVectorT<double>& inpar);
 protected:
   bool isFit_,isConfigured_;
   TVectorT<double> par_;
   TMatrixTSym<double> parcov_;
   virtual TString freeParName(int Par);
   double chi2_, ndf_;
 private:
   static TVectorT<double> computePar(const TVectorT<double>& inpar);
   static TVectorT<double> computeTrackPar(const TVectorT<double>& inpar,int p=0);
   static TVectorT<double> computeMotherLorentzVectorPar(const TVectorT<double>& inpar);

   static int measuredValueIndex(int TrackPar,int Particle){
     return TrackPar+Particle*TrackParticle::NHelixPar;
   }
   static int freeParIndex(int Par,int Particle){
     if(Par==x0 || Par==y0 || Par==z0) return Par;
     return Par+Particle*(NFreeTrackPar-NFreeVertexPar);
   }
   static void parSizeInfo(const TVectorT<double>& inpar, int& np, int& parsize,bool hasextras=0);

   std::vector<TrackParticle> particles_;
   TVectorT<double> val_;
   TMatrixTSym<double> cov_;
   TMatrixTSym<double> cov_inv_;
   int nParticles_,nPar_,nVal_;
};

}
#endif


