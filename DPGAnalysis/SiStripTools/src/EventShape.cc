#include "DPGAnalysis/SiStripTools/interface/EventShape.h"
#include <DataFormats/TrackReco/interface/Track.h>
#include <TVector3.h>
#include <vector>
#include <TMatrixDSym.h>
#include <TMatrixDSymEigen.h>
#include <TVectorD.h>

using namespace edm;
using namespace reco;
using namespace std;
using reco::TrackCollection;

EventShape::EventShape(reco::TrackCollection& tracks):eigenvalues(3)
{
  for(reco::TrackCollection::const_iterator itTrack = tracks.begin(); itTrack<tracks.end(); ++itTrack) {
    p.push_back(TVector3(itTrack->px(),itTrack->py(),itTrack->pz()));
  }
  
  // first fill the momentum tensor
  TMatrixDSym MomentumTensor(3);
  for(std::vector<TVector3>::const_iterator momentum = p.begin();momentum<p.end();++momentum) {
    for(unsigned int i=0;i<3;i++)
      for(unsigned int j=0;j<=i;j++) {
        MomentumTensor[i][j] += momentum[i]*momentum[j];
      }
  }
  MomentumTensor*=1/(MomentumTensor[0][0]+MomentumTensor[1][1]+MomentumTensor[2][2]);
  // find the eigen values
  TMatrixDSymEigen eigen(MomentumTensor);
  TVectorD eigenvals = eigen.GetEigenValues();
  eigenvalues[0] = eigenvals[0];
  eigenvalues[1] = eigenvals[1];
  eigenvalues[2] = eigenvals[2];
  sort(eigenvalues.begin(),eigenvalues.end());
}

math::XYZTLorentzVectorF EventShape::thrust() const
{
  math::XYZTLorentzVectorF output = math::XYZTLorentzVectorF(0,0,0,0);
  TVector3 qtbo;
  TVector3 zero(0.,0.,0.);
  float vnew = 0.;
  uint32_t Np = p.size();
  
  // for more than 2 tracks
  if (Np > 2) { 
      float vmax  = 0.;
      TVector3 vn, vm, vc, vl;
      for(unsigned int i=0; i< Np-1; i++) 
	for(unsigned int j=i+1; j < Np; j++) {
	    vc = p[i].Cross(p[j]);
	    vl = zero; 
	    for(unsigned int k=0; k<Np; k++)
	      if ((k != i) && (k != j)) {
		if (p[k].Dot(vc) >= 0.) vl = vl + p[k];
		else vl = vl - p[k];
	      }
	    // make all four sign-combinations for i,j
	    vn = vl + p[j] + p[i];
	    vnew = vn.Mag2();
	    if (vnew >  vmax) {  
		vmax = vnew;
		vm = vn;
	    }
	    vn = vl + p[j] - p[i];
	    vnew = vn.Mag2();
	    if (vnew >  vmax) {  
		vmax = vnew;
		vm = vn;
	    }
	    vn = vl - p[j] + p[i];
	    vnew = vn.Mag2();
	    if (vnew >  vmax) {  
		vmax = vnew;
		vm = vn;
	    }
	    vn = vl - p[j] - p[i];
	    vnew = vn.Mag2();
	    if (vnew >  vmax) {  
		vmax = vnew;
		vm = vn;
	    }    
	  }
      // sum momenta of all particles and iterate
      for(int iter=1; iter<=4; iter++) {  
	  qtbo = zero;
	  for(unsigned int i=0; i< Np; i++)
	    if (vm.Dot(p[i]) >= 0.) 
	      qtbo = qtbo + p[i];
	    else 
	      qtbo = qtbo - p[i];
	  vnew = qtbo.Mag2();
	  if (vnew  == vmax) break;
	  vmax = vnew;
	  vm = qtbo;
	}
    }  // of if Np > 2
  else
    if (Np == 2)
      if (p[0].Dot(p[1]) >= 0.) 
	qtbo = p[0] + p[1];
      else
	qtbo = p[0] - p[1];
    else if (Np == 1)
      qtbo = p[0];
    else {
	qtbo = zero;
	return output;
    }
  // normalize thrust -division by total momentum-
  float vsum = 0.;
  for(unsigned int i=0; i < Np; i++) vsum = vsum + p[i].Mag();
  vnew  = qtbo.Mag();
  float v = vnew/vsum; 
  float x = qtbo.X()/vnew;
  float y = qtbo.Y()/vnew;
  float z = qtbo.Z()/vnew;
  output.SetPxPyPzE(x, y, z, v);
  return output;
}

math::XYZTLorentzVectorF EventShape::thrust(const reco::TrackCollection& tracks)
{
  std::vector<TVector3> pp;   
  uint32_t Np = tracks.size();
  math::XYZTLorentzVectorF output = math::XYZTLorentzVectorF(0,0,0,0);
  for(reco::TrackCollection::const_iterator itTrack = tracks.begin(); itTrack<tracks.end(); ++itTrack) {
    pp.push_back(TVector3(itTrack->px(),itTrack->py(),itTrack->pz()));
  }
  TVector3 qtbo;
  TVector3 zero(0.,0.,0.);
  float vnew = 0.;
  
  // for more than 2 tracks
  if (Np > 2) { 
      float vmax  = 0.;
      TVector3 vn, vm, vc, vl;
      for(unsigned int i=0; i< Np-1; i++) 
	for(unsigned int j=i+1; j < Np; j++) {
	    vc = pp[i].Cross(pp[j]);
	    vl = zero; 
	    for(unsigned int k=0; k<Np; k++)
	      if ((k != i) && (k != j)) {
		if (pp[k].Dot(vc) >= 0.) vl = vl + pp[k];
		else vl = vl - pp[k];
	      }
	    // make all four sign-combinations for i,j
	    vn = vl + pp[j] + pp[i];
	    vnew = vn.Mag2();
	    if (vnew >  vmax) {  
		vmax = vnew;
		vm = vn;
	    }
	    vn = vl + pp[j] - pp[i];
	    vnew = vn.Mag2();
	    if (vnew >  vmax) {  
		vmax = vnew;
		vm = vn;
	    }
	    vn = vl - pp[j] + pp[i];
	    vnew = vn.Mag2();
	    if (vnew >  vmax) {  
		vmax = vnew;
		vm = vn;
	    }
	    vn = vl - pp[j] - pp[i];
	    vnew = vn.Mag2();
	    if (vnew >  vmax) {  
		vmax = vnew;
		vm = vn;
	    }    
	  }
      // sum momenta of all particles and iterate
      for(int iter=1; iter<=4; iter++) {  
	  qtbo = zero;
	  for(unsigned int i=0; i< Np; i++)
	    if (vm.Dot(pp[i]) >= 0.) 
	      qtbo = qtbo + pp[i];
	    else 
	      qtbo = qtbo - pp[i];
	  vnew = qtbo.Mag2();
	  if (vnew  == vmax) break;
	  vmax = vnew;
	  vm = qtbo;
	}
    }  // of if Np > 2
  else
    if (Np == 2)
      if (pp[0].Dot(pp[1]) >= 0.) 
	qtbo = pp[0] + pp[1];
      else
	qtbo = pp[0] - pp[1];
    else if (Np == 1)
      qtbo = pp[0];
    else {
	qtbo = zero;
	return output;
    }
  // normalize thrust -division by total momentum-
  float vsum = 0.;
  for(unsigned int i=0; i < Np; i++) vsum = vsum + pp[i].Mag();
  vnew  = qtbo.Mag();
  float v = vnew/vsum; 
  float x = qtbo.X()/vnew;
  float y = qtbo.Y()/vnew;
  float z = qtbo.Z()/vnew;
  output.SetPxPyPzE(x, y, z, v);
  return output;
}

float EventShape::sphericity(const reco::TrackCollection& tracks)
{
  // a critical check
  if(tracks.empty()) return 0;
  
  // first fill the momentum tensor
  TMatrixDSym MomentumTensor(3);
  for(reco::TrackCollection::const_iterator itTrack = tracks.begin(); itTrack<tracks.end(); ++itTrack) {
  std::vector<double> momentum(3);
  momentum[0] = itTrack->px();
  momentum[1] = itTrack->py();
  momentum[2] = itTrack->pz();
    for(unsigned int i=0;i<3;i++)
      for(unsigned int j=0;j<=i;j++) {
        MomentumTensor[i][j] += momentum[i]*momentum[j];
      }
  }
  MomentumTensor*=1/(MomentumTensor[0][0]+MomentumTensor[1][1]+MomentumTensor[2][2]);
  // find the eigen values
  TMatrixDSymEigen eigen(MomentumTensor);
  TVectorD eigenvals = eigen.GetEigenValues();
  vector<float> eigenvaluess(3);
  eigenvaluess[0] = eigenvals[0];
  eigenvaluess[1] = eigenvals[1];
  eigenvaluess[2] = eigenvals[2];
  sort(eigenvaluess.begin(),eigenvaluess.end());
  // compute spericity
  float sph = ( 1.5*(1-eigenvaluess[2]));
  return sph;
}

float EventShape::aplanarity(const reco::TrackCollection& tracks)
{
  // a critical check
  if (tracks.empty()) return 0;
  // first fill the momentum tensor
  TMatrixDSym MomentumTensor(3);
  for(reco::TrackCollection::const_iterator itTrack = tracks.begin(); itTrack<tracks.end(); ++itTrack) {
  std::vector<double> momentum(3);
  momentum[0] = itTrack->px();
  momentum[1] = itTrack->py();
  momentum[2] = itTrack->pz();
    for(unsigned int i=0;i<3;i++)
      for(unsigned int j=0;j<=i;j++) {
        MomentumTensor[i][j] += momentum[i]*momentum[j];
      }
  }
  MomentumTensor*=1/(MomentumTensor[0][0]+MomentumTensor[1][1]+MomentumTensor[2][2]);
  // find the eigen values
  TMatrixDSymEigen eigen(MomentumTensor);
  TVectorD eigenvals = eigen.GetEigenValues();
  vector<float> eigenvaluess(3);
  eigenvaluess[0] = eigenvals[0];
  eigenvaluess[1] = eigenvals[1];
  eigenvaluess[2] = eigenvals[2];
  sort(eigenvaluess.begin(),eigenvaluess.end());
  // compute aplanarity
  return ( 1.5*eigenvaluess[0]);
}

float EventShape::planarity(const reco::TrackCollection& tracks)
{
  // First a critical check
  if (tracks.empty()) return 0;
  // first fill the momentum tensor
  TMatrixDSym MomentumTensor(3);
  for(reco::TrackCollection::const_iterator itTrack = tracks.begin(); itTrack<tracks.end(); ++itTrack) {
  std::vector<double> momentum(3);
  momentum[0] = itTrack->px();
  momentum[1] = itTrack->py();
  momentum[2] = itTrack->pz();
    for(unsigned int i=0;i<3;i++)
      for(unsigned int j=0;j<=i;j++) {
        MomentumTensor[i][j] += momentum[i]*momentum[j];
      }
  }
  MomentumTensor*=1/(MomentumTensor[0][0]+MomentumTensor[1][1]+MomentumTensor[2][2]);
  // find the eigen values
  TMatrixDSymEigen eigen(MomentumTensor);
  TVectorD eigenvals = eigen.GetEigenValues();
  vector<float> eigenvaluess(3);
  eigenvaluess[0] = eigenvals[0];
  eigenvaluess[1] = eigenvals[1];
  eigenvaluess[2] = eigenvals[2];
  sort(eigenvaluess.begin(),eigenvaluess.end());
  // compute planarity
  return (eigenvaluess[0]/eigenvaluess[1]);
}

float EventShape::sphericity() const
{
  // compute sphericity
  return ( 1.5*(1-eigenvalues[2]));
}

float EventShape::aplanarity() const
{
  // compute aplanarity
  return ( 1.5*eigenvalues[0]);
}

float EventShape::planarity() const
{
  // compute planarity
  return (eigenvalues[0]/eigenvalues[1]);
}
