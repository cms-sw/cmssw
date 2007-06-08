/*
 *  See header file for a description of this class.
 *
 *
 *  $Date: 2006/10/05 13:11:55 $
 *  $Revision: 1.16 $
 *  \author D. Pagano - University of Pavia & INFN Pavia
 *
 */

#include "RecoMuon/MuonSeedGenerator/src/RPCSeedHits.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "gsl/gsl_statistics.h"
#include "TH1F.h"
#include "math.h"

using namespace std;

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

template <class T> T sqr(const T& t) {return t*t;}

TrajectorySeed RPCSeedHits::seed(const edm::EventSetup& eSetup) const {

  double pt[4] = {0,0,0,0};
  double spt[4] = {0,0,0,0}; 
  
  computePtWithoutVtx(pt, spt);
  
  float ptmean=0.;
  float sptmean=0.;

  computeBestPt(pt, spt, ptmean, sptmean);
  
  cout << "III--> Seed Pt : " << ptmean << endl;
  ConstMuonRecHitPointer last = best_cand();
  return createSeed(ptmean, sptmean,last,eSetup);
}


ConstMuonRecHitPointer RPCSeedHits::best_cand() const {

  MuonRecHitPointer best = 0;

  for (MuonRecHitContainer::const_iterator iter=theRhits.begin();
       iter!=theRhits.end(); iter++) {
       best = (*iter);
  } 

  return best;
}


void RPCSeedHits::computePtWithoutVtx(double* pt, double* spt) const {

  int i = 0;
  float y1,y2,y3,y4,ys,yt,yq,yd;
  float x1,x2,x3,x4,xs,xt,xq,xd;
  x1=x2=x3=x4=xs=xt=xq=xd=y1=y2=y3=y4=ys=yd=yt=yq=0;
  for (MuonRecHitContainer::const_iterator iter=theRhits.begin(); 
       iter!=theRhits.end(); iter++ ) {
    i++;

    cout << "X= " << (*iter)->globalPosition().x() << " Y= " << (*iter)->globalPosition().y() << endl;

    if (i == 1) {y1= (*iter)->globalPosition().y(); x1= (*iter)->globalPosition().x();}
    if (i == 2) {y2= (*iter)->globalPosition().y(); x2= (*iter)->globalPosition().x();}
    if (i == 3) {y3= (*iter)->globalPosition().y(); x3= (*iter)->globalPosition().x();}
    if (i == 4) {y4= (*iter)->globalPosition().y(); x4= (*iter)->globalPosition().x();}
  }

  string ord = "1-2-3";
  for (int comb = 1; comb < 5; comb++) {
    
    if (comb == 2) {
      ord = "1-2-4";
      yt=y3; y3=y4; y4=yq; xt=x3; x3=x4; x4=xq; 
    }
    
    if (comb == 3) {
      ord = "1-3-4";
      yd=y2; y2=yt; xd=x2; x2=xt;  
    }
    
    if (comb == 4) {
      ord = "2-3-4";
      ys=y1; y1=yd; xs=x1; x1=xd; 
    }
    
    float A = (y3-y2)/(x3-x2) - (y2-y1)/(x2-x1);
    float TYO = (x3-x1)/A + (y3*y3-y2*y2)/((x3-x2)*A) - (y2*y2-y1*y1)/((x2-x1)*A);
    float TXO = (x3+x2) + (y3*y3-y2*y2)/(x3-x2) - TYO*(y3-y2)/(x3-x2);
    float XO = 0.5 * TXO;
    float YO = 0.5 * TYO;
    float R2 = (x1-XO)*(x1-XO) + (y1-YO)*(y1-YO); 
    pt[comb-1] = 0.01 * sqrt(R2) * 2 * 0.3;
    }
     
}


void RPCSeedHits::computeBestPt(double* pt,
                                double* spt,
                                float& ptmean,
                                float& sptmean) const {

  cout << "[RPCSeedHits] --> computeBestPt class called." << endl;

  cout << "---< best pt computing >---" << endl;
  cout << "1-2-3 pt = " << pt[0] << endl;
  cout << "1-2-4 pt = " << pt[1] << endl;
  cout << "1-3-4 pt = " << pt[2] << endl;
  cout << "2-3-4 pt = " << pt[3] << endl;
  cout << "---------------------------" << endl;

  ptmean = (pt[0]+pt[1]+pt[2]+pt[3])/4;

  sptmean = spt[0];
  
}


TrajectorySeed RPCSeedHits::createSeed(float ptmean,
			               float sptmean,
				       ConstMuonRecHitPointer last,
				       const edm::EventSetup& eSetup) const{
  
  MuonPatternRecoDumper debug;
  
  edm::ESHandle<MagneticField> field;
  eSetup.get<IdealMagneticFieldRecord>().get(field);

  double theMinMomentum = 3.0;
 
  if ( fabs(ptmean) < theMinMomentum ) ptmean = theMinMomentum * ptmean/fabs(ptmean) ;

  AlgebraicVector t(4);
  AlgebraicSymMatrix mat(5,0) ;

  LocalPoint segPos=last->localPosition();
  GlobalVector mom=last->globalPosition()-GlobalPoint();
  GlobalVector polar(GlobalVector::Spherical(mom.theta(), last->globalDirection().phi(), 1.));
  polar *= fabs(ptmean)/polar.perp();
  LocalVector segDirFromPos=last->det()->toLocal(polar);
  int charge=(int)(ptmean/fabs(ptmean));

  LocalTrajectoryParameters param(segPos,segDirFromPos, charge);

  mat = last->parametersError().similarityT( last->projectionMatrix() );
  
  float p_err = sqr(sptmean/(ptmean*ptmean));
  mat[0][0]= p_err;
 
  LocalTrajectoryError error(mat);
  
  TrajectoryStateOnSurface tsos(param, error, last->det()->surface(),&*field);

  cout << "Trajectory State on Surface before the extrapolation"<<endl;
  cout << debug.dumpTSOS(tsos);
  
  DetId id = last->geographicalId();

  cout << "The RecSegment relies on: "<<endl;
  cout << debug.dumpMuonId(id);
  cout << debug.dumpTSOS(tsos);

  TrajectoryStateTransform tsTransform;
  
  PTrajectoryStateOnDet *seedTSOS = tsTransform.persistentState( tsos ,id.rawId());
  
  edm::OwnVector<TrackingRecHit> container;
  TrajectorySeed theSeed(*seedTSOS,container,oppositeToMomentum);
    
  return theSeed;
}
