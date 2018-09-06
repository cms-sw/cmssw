/*
Kalman Track class for 
Kalman Muon Track Finder
Michalis Bachtis (UCLA)
Sep. 2017
*/

#ifndef L1MuKBMTrack_H
#define L1MuKBMTrack_H

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

class L1MuKBMTrack;
typedef std::vector<L1MuKBMTrack> L1MuKBMTrackCollection;
typedef BXVector<L1MuKBMTrack> L1MuKBMTrackBxCollection;

class L1MuKBMTrack : public reco::LeafCandidate
{

public:
  L1MuKBMTrack();
  ~L1MuKBMTrack() override;
  L1MuKBMTrack(const L1MuKBMTCombinedStubRef&,int,int);

  //UnConstrained curvature at station 1
  int curvatureAtMuon() const; 
  //unconstrained phi at station 1
  int phiAtMuon() const;
  //unconstrained phiB at station 1
  int phiBAtMuon() const;
  //Constrained curvature at vertex
  int curvatureAtVertex() const; 
  //constrained phi at the vertex
  int phiAtVertex() const;
  //Impact parameter as calculated from the muon track 
  int dxy() const;
  //Unconstrained curvature at the Muon systen 
  int curvature() const;
  //Unconstrained phi at the Muon systen 
  int positionAngle() const;
  //Unconstrained bending angle at the Muon systen 
  int bendingAngle() const;
  //Coarse eta caluclated only using phi segments 
  int coarseEta() const;
  //Approximate Chi2 metric
  int approxChi2() const;
  //Approximate Chi2 metric
  int hitPattern() const;
  //step;
  int step() const;
  //sector;
  int sector() const;
  //wheel
  int wheel() const;
  //quality
  int quality() const;

  //unconstrained pt
  float ptUnconstrained() const;

  //fine eta
  int fineEta() const;
  bool hasFineEta() const;


  //BX
  int bx() const;

  //rank
  int rank() const;

  //Associated stubs
  const L1MuKBMTCombinedStubRefVector& stubs() const;

  //get Kalman gain
  const std::vector<float>& kalmanGain(unsigned int) const;

  //get covariance
  const std::vector<double>& covariance() const;


  //get residual
  int residual(uint) const;

  //check ogverlap
  bool overlapTrack(const L1MuKBMTrack&) const; 

  bool operator==(const L1MuKBMTrack& t2) const{   
    if (this->stubs().size()!=t2.stubs().size())
      return false;
    for (unsigned int i=0;i<this->stubs().size();++i)  {
      const L1MuKBMTCombinedStubRef& s1 = this->stubs()[i];
      const L1MuKBMTCombinedStubRef& s2 = t2.stubs()[i];
      if (s1->scNum()!= s2->scNum() ||
	  s1->whNum()!=s2->whNum() ||
	  s1->stNum()!=s2->stNum() ||
	  s1->tag()!=s2->tag())
	return false;
    }
    return true;
  }



  //Set coordinates general
  void setCoordinates(int,int,int,int );

  //Set coordinates at vertex
  void setCoordinatesAtVertex(int,int,int );

  //Set coordinates at muon
  void setCoordinatesAtMuon(int,int,int );

  //Set eta coarse and pattern
  void setCoarseEta(int);

  //Set phi hit pattern
  void setHitPattern(int);

  //Set chi2 like metric
  void setApproxChi2(int);

  //Set floating point coordinates for studies
  void setPtEtaPhi(double,double,double);
  void setPtUnconstrained(float);

  //Add a stub
  void addStub(const L1MuKBMTCombinedStubRef&); 

  //kalman gain management
  void setKalmanGain(unsigned int step, unsigned int K,float a1 ,float a2,float a3,float a4=0 ,float a5=0,float a6=0);

  //set covariance
  void setCovariance(const CovarianceMatrix&);

  //set fine eta
  void setFineEta(int);

  //set rank
  void setRank(int);

  //set residual
  void setResidual(uint,int);


 private:

  //Covariance matrix for studies
  std::vector<double> covariance_;
  
  L1MuKBMTCombinedStubRefVector stubs_;

  //vertex coordinates
  int curvVertex_;
  int phiVertex_;
  int dxy_;

  //muon coordinates
  int curvMuon_;
  int phiMuon_;
  int phiBMuon_;


  //generic coordinates
  int curv_;
  int phi_;
  int phiB_; 
  //common coordinates
  int coarseEta_;

  //Approximate Chi2 metric
  int approxChi2_;

  //phi bitmask
  int hitPattern_;

  //propagation step
  int step_;

  //sector
  int sector_;
  //wheel
  int wheel_;

  //quality
  int quality_;

  //Fine eta
  int fineEta_;

  //has fine eta?
  bool hasFineEta_;

  //BX
  int bx_;

  //rank
  int rank_;


  //Unconstrained floating point pt
  float ptUnconstrained_;

  //Kalman Gain for making LUTs
  std::vector<float> kalmanGain0_;
  std::vector<float> kalmanGain1_;
  std::vector<float> kalmanGain2_;
  std::vector<float> kalmanGain3_;


  std::vector<int> residuals_;
  
  
}; 

#endif
