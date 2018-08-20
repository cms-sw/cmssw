/*
Kalman Filter L1 Muon algorithm
Michalis Bachtis (UCLA)
Sep. 2017

*/

#ifndef L1TMuonBarrelKalmanAlgo_H
#define L1TMuonBarrelKalmanAlgo_H

#include "DataFormats/L1TMuon/interface/L1MuKBMTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanLUTs.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

class L1TMuonBarrelKalmanAlgo {
 public:
  typedef ROOT::Math::SVector<double,2> Vector2;
  typedef ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > CovarianceMatrix2;
  typedef ROOT::Math::SMatrix<double,3,2> Matrix32;
  typedef ROOT::Math::SMatrix<double,2,3> Matrix23;
  typedef ROOT::Math::SMatrix<double,1,3> Matrix13;
  typedef ROOT::Math::SMatrix<double,3,1> Matrix31;
  typedef ROOT::Math::SMatrix<double,3,3> Matrix33;

  L1TMuonBarrelKalmanAlgo (const edm::ParameterSet& settings);
  std::pair<bool,L1MuKBMTrack> chain(const L1MuKBMTCombinedStubRef&, const L1MuKBMTCombinedStubRefVector&);

  L1MuKBMTrackCollection clean(const L1MuKBMTrackCollection&,uint);


  L1MuKBMTrackCollection cleanAndSort(const L1MuKBMTrackCollection&,uint);
  void addBMTFMuon(int,const L1MuKBMTrack&,std::unique_ptr<l1t::RegionalMuonCandBxCollection>&);
  l1t::RegionalMuonCand  convertToBMTF(const L1MuKBMTrack& track); 






 
 private:
  bool verbose_;
  std::pair<bool,uint> match(const L1MuKBMTCombinedStubRef&, const L1MuKBMTCombinedStubRefVector&,int );
  int correctedPhi(const L1MuKBMTCombinedStubRef&,int);
  int correctedPhiB(const L1MuKBMTCombinedStubRef&);
  void propagate(L1MuKBMTrack&);
  void updateEta(L1MuKBMTrack&,const L1MuKBMTCombinedStubRef&);
  bool update(L1MuKBMTrack&,const L1MuKBMTCombinedStubRef&,int);
  bool updateOffline(L1MuKBMTrack&,const L1MuKBMTCombinedStubRef&);
  bool updateOffline1D(L1MuKBMTrack&,const L1MuKBMTCombinedStubRef&);
  bool updateLUT(L1MuKBMTrack&,const L1MuKBMTCombinedStubRef&,int);
  void vertexConstraint(L1MuKBMTrack&);
  void vertexConstraintOffline(L1MuKBMTrack&);
  void vertexConstraintLUT(L1MuKBMTrack&);
  int hitPattern(const L1MuKBMTrack&);
  int customBitmask(unsigned int,unsigned int,unsigned int,unsigned int);
  bool getBit(int,int);
  void setFloatingPointValues(L1MuKBMTrack&,bool);
  int phiAt2(const L1MuKBMTrack& track);
  void estimateChiSquare(L1MuKBMTrack&);
  int rank(const L1MuKBMTrack&);
  int wrapAround(int,int);
  std::pair<bool,uint> getByCode(const L1MuKBMTrackCollection& tracks,int mask);
  std::map<int,int> trackAddress(const L1MuKBMTrack&,int&);
  int encode(bool ownwheel,int sector,bool tag); 
  uint twosCompToBits(int);
  int fp_product(float,int, uint);

  uint etaStubRank(const L1MuKBMTCombinedStubRef&);

  void calculateEta(L1MuKBMTrack& track);


  //LUT service
  L1TMuonBarrelKalmanLUTs* lutService_;
  bool punchThroughVeto(const L1MuKBMTrack& track);
  int ptLUT(int K);


  //Initial Curvature
  std::vector<double> initK_;
  std::vector<double> initK2_;

  //propagation coefficients
  std::vector<double> eLoss_;
  std::vector<double> aPhi_;
  std::vector<double> aPhiB_;
  std::vector<double> aPhiBNLO_;
  std::vector<double> bPhi_;
  std::vector<double> bPhiB_;
  std::vector<double> phiAt2_;
  std::vector<double> etaLUT0_;
  std::vector<double> etaLUT1_;

  //Chi Square estimator input
  uint globalChi2Cut_;
  std::vector<double> chiSquare_;
  std::vector<int> chiSquareCutPattern_;
  std::vector<int> chiSquareCutCurv_;
  std::vector<int> chiSquareCut_;


  //bitmasks to run== diferent combinations for a given seed in a given station
  std::vector<int> combos4_;
  std::vector<int> combos3_;
  std::vector<int> combos2_;
  std::vector<int> combos1_;


  //STUFF NOT USED IN THE FIRMWARE BUT ONLY FOR DEBUGGING
  ///////////////////////////////////////////////////////

  bool useOfflineAlgo_;
  std::vector<double> mScatteringPhi_;
  std::vector<double> mScatteringPhiB_;
  //point resolution for phi
  double pointResolutionPhi_;
  //point resolution for phiB
  double pointResolutionPhiB_;
  //point resolution for vertex
  double pointResolutionVertex_;

  

  //Sorter
  class StubSorter {
  public:
    StubSorter(uint sector) {
      sec_ = sector;
    }

    bool operator() (const L1MuKBMTCombinedStubRef& a ,const L1MuKBMTCombinedStubRef& b) {
      if (correctedPhi(a)<correctedPhi(b))
	return true;
      return false;
    }


  private:
    int sec_;
    int correctedPhi(const L1MuKBMTCombinedStubRef& stub) {
      if (stub->scNum()==sec_)
	return stub->phi();
      else if ((stub->scNum()==sec_-1) || (stub->scNum()==11 && sec_==0))
	return stub->phi()-2144;
      else if ((stub->scNum()==sec_+1) || (stub->scNum()==0 && sec_==11))
	return stub->phi()+2144;
      return 0;
    } 

  };

  class TrackSorter {
  public:
    TrackSorter() {
    }

    bool operator() (const L1MuKBMTrack& a ,const L1MuKBMTrack& b) {
      if (abs(a.curvatureAtVertex())<=abs(b.curvatureAtVertex()))
	return true;
      return false;
    }
  };
  



 


};
#endif



