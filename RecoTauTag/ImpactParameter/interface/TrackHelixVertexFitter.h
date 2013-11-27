/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#ifndef TrackHelixVertexFitter_h
#define TrackHelixVertexFitter_h

// system include files
#include <TMatrixT.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVectorD.h>

#include "RecoTauTag/ImpactParameter/interface/Particle.h"
#include "RecoTauTag/ImpactParameter/interface/TrackParticle.h"
#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"
#include "RecoTauTag/ImpactParameter/interface/ErrorMatrixPropagator.h"

class TrackHelixVertexFitter{
 public:
  enum FreeVertexPar{x0=0,y0,z0,NFreeVertexPar};
  enum FreeTrackPar{kappa0=3,lambda0,phi0,NFreeTrackPar};
  enum ExtraPar{BField0=0,MassOffSet=1,NExtraPar=1};

  TrackHelixVertexFitter(std::vector<TrackParticle> &particles_,TVector3 vguess);
  virtual ~TrackHelixVertexFitter();

   virtual bool Fit()=0;
   virtual double UpdateChisquare(TMatrixT<double> inpar);
   virtual double ChiSquare(){return chi2;}
   virtual double NDF(){return ndf;}
   LorentzVectorParticle GetMother(int pdgid);
   virtual std::vector<TrackParticle> GetReFitTracks();
   std::vector<LorentzVectorParticle> GetReFitLorentzVectorParticles();
   virtual TVector3 GetVertex();
   virtual TMatrixTSym<double> GetVertexError();

   static void Computedxydz(TMatrixT<double> &inpar,int particle,double &kappa,double &lam,double &phi,double &x,double &y,double &z,double &s,double &dxy,double &dz);
   static TMatrixT<double> ComputeLorentzVectorPar(TMatrixT<double> &inpar);
 protected:
   bool isFit,isConfigure;
   TMatrixT<double> par;
   TMatrixTSym<double> parcov;
   virtual TString FreeParName(int Par);
   double chi2, ndf;
 private:
   static TMatrixT<double> ComputePar(TMatrixT<double> &inpar);
   static TMatrixT<double> ComputeTrackPar(TMatrixT<double> &inpar,int p=0);
   static TMatrixT<double> ComputeMotherLorentzVectorPar(TMatrixT<double> &inpar);

   static int MeasuredValueIndex(int TrackPar,int Particle){
     return TrackPar+Particle*TrackParticle::NHelixPar;
   }
   static int FreeParIndex(int Par,int Particle){
     if(Par==x0 || Par==y0 || Par==z0) return Par;
     return Par+Particle*(NFreeTrackPar-NFreeVertexPar);
   }
   static void ParSizeInfo(TMatrixT<double> &inpar, int &np, int &parsize,bool hasextras=0);

   std::vector<TrackParticle> particles;
   TMatrixT<double> val;
   TMatrixTSym<double> cov;
   TMatrixTSym<double> cov_inv;
   int nParticles,nPar,nVal;

};
#endif


