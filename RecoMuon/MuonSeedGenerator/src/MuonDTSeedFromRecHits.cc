/**
 *  See header file for a description of this class.
 *
 *
 *  $Date: 2007/09/14 18:57:06 $
 *  $Revision: 1.10 $
 *  \author A. Vitelli - INFN Torino, V.Palichik
 *  \author porting  R. Bellan
 *
 */
#include "RecoMuon/MuonSeedGenerator/src/MuonDTSeedFromRecHits.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedPtExtractor.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

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

using namespace std;
double ptmin = 3.;



template <class T> T sqr(const T& t) {return t*t;}

MuonDTSeedFromRecHits::MuonDTSeedFromRecHits()
: MuonSeedFromRecHits()
{
}


TrajectorySeed MuonDTSeedFromRecHits::seed() const {
  double pt[8] = { 0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 };
  // these weights are supposed to be 1/sigma^2(pt), but they're so small.
  // Instead of the 0.2-0.5 GeV here, we'll add something extra in quadrature later
  double spt[8] = { 1/0.048 , 1/0.075 , 1/0.226 , 1/0.132 , 1/0.106 , 1/0.175 , 1/0.125 , 1/0.185 }; 

  const std::string metname = "Muon|RecoMuon|MuonDTSeedFromRecHits";

  /// compute pts with vertex constraint
  computePtWithVtx(pt, spt);

  /// now w/o vertex constrain
  computePtWithoutVtx(pt, spt);

  // some dump...
  LogTrace(metname) << " Pt MB1 @vtx: " << pt[0] << " w: " << spt[0] << "\n" 
		    << " Pt MB2 @vtx: " << pt[1] << " w: " << spt[1]<< endl ;
  
  LogTrace(metname) << " Pt MB2-MB1 " << pt[2] << " w: " << spt[2]<< "\n" 
		    << " Pt MB3-MB1 " << pt[3] << " w: " << spt[3]<< "\n" 
		    << " Pt MB3-MB2 " << pt[4] << " w: " << spt[4]<< "\n" 
		    << " Pt MB4-MB1 " << pt[5] << " w: " << spt[5]<< "\n" 
		    << " Pt MB4-MB2 " << pt[6] << " w: " << spt[6]<< "\n" 
		    << " Pt MB4-MB3 " << pt[7] << " w: " << spt[7]<< endl  ;
  
  /// now combine all pt estimate
  float ptmean=0.;
  float sptmean=0.;
  computeBestPt(pt, spt, ptmean, sptmean);
  
  // add an extra term to the error in quadrature, 30% of pT per point
  int npoints = 0;
  for(int i = 0; i < 8; ++i)
    if(pt[i] != 0) ++npoints;
  if(npoints != 0) {
    sptmean = sqrt(sptmean*sptmean + 0.09*ptmean*ptmean/npoints);
  }

  LogTrace(metname) << " Seed Pt: " << ptmean << " +/- " << sptmean << endl;
     
  // take the best candidate
  ConstMuonRecHitPointer last = best_cand();
  return createSeed(ptmean, sptmean,last);
}


MuonDTSeedFromRecHits::ConstMuonRecHitPointer 
MuonDTSeedFromRecHits::best_cand() const {

  int alt_npt = 0;
  int best_npt = 0;
  int cur_npt = 0;
  MuonRecHitPointer best = 0;
  MuonRecHitPointer alter=0;

  for (MuonRecHitContainer::const_iterator iter=theRhits.begin();
       iter!=theRhits.end(); iter++ ) {

    bool hasZed = ((*iter)->projectionMatrix()[1][1]==1);

    cur_npt = 0;
    vector<TrackingRecHit*> slrhs=(*iter)->recHits();
    for (vector<TrackingRecHit*>::const_iterator slrh=slrhs.begin(); slrh!=slrhs.end(); ++slrh ) {
      cur_npt+=(*slrh)->recHits().size();
    }
    float radius1 = (*iter)->det()->position().perp();

    if (hasZed) {
      if ( cur_npt > best_npt ) {
        best = (*iter);
        best_npt = cur_npt;
      }
      else if ( best && cur_npt == best_npt ) {
        float radius2 = best->det()->position().perp();
       if (radius1 < radius2) {
          best = (*iter);
          best_npt = cur_npt;
       }
      }
    }

    if ( cur_npt > alt_npt ) {
      alter = (*iter);
      alt_npt = cur_npt;
    }
    else if ( alter && cur_npt == alt_npt ) {
      float radius2 = alter->det()->position().perp();
      if (radius1 < radius2) {
        alter = (*iter);
        alt_npt = cur_npt;
      }
    }
  }

  if ( !best ) best = alter;

  return best;
}


float MuonDTSeedFromRecHits::bestEta() const {

  int Maxseg = 0;
  float Msdeta = 100.;
  float result = (*theRhits.begin())->globalPosition().eta();

  for (MuonRecHitContainer::const_iterator iter=theRhits.begin(); iter!=theRhits.end(); iter++ ) {

    float eta1= (*iter)->globalPosition().eta(); 

    int Nseg = 0;
    float sdeta = .0;

    for (MuonRecHitContainer::const_iterator iter2=theRhits.begin();  iter2!=theRhits.end(); iter2++ ) {

      if ( iter2 == iter )  continue;

      float eta2 = (*iter2)->globalPosition().eta(); 

      if ( fabs (eta1-eta2) > .2 ) continue;

      Nseg++;
      sdeta += fabs (eta1-eta2); 

      if ( Nseg > Maxseg ||  Nseg == Maxseg && sdeta < Msdeta ) {
	Maxseg = Nseg;
	Msdeta = sdeta;
	result = eta1;
      }

    }
  }   //  +v.
  return result;
}


void MuonDTSeedFromRecHits::computePtWithVtx(double* pt, double* spt) const {

  float eta0 = bestEta();

  for (MuonRecHitContainer::const_iterator iter=theRhits.begin(); iter!=theRhits.end(); iter++ ) {

 //+vvp !:

    float eta1= (*iter)->globalPosition().eta(); 

    if ( fabs (eta1-eta0) > .2 ) continue;  //   !!! +vvp

    // assign Pt from MB1 & vtx   
    float radius = (*iter)->det()->position().perp();
    unsigned int stat = 0;
    if ( radius>450 && radius<550 ) stat=2;
    if ( radius<450 ) stat=1;

    if(stat == 0) continue;

    std::vector<double> pts = thePtExtractor->pT_extract(*iter, *iter);
    pt[stat-1] = pts[0];
    spt[stat-1] = pts[1];
    if(pt[stat-1]>0 && pt[stat-1] < ptmin) pt[stat-1] = ptmin;
    if(pt[stat-1]<0 && pt[stat-1] > ptmin) pt[stat-1] = -ptmin;

    float ptmax = 2000.;
    if(pt[0] > ptmax) pt[0] = ptmax;
    if(pt[0] < -ptmax) pt[0] = -ptmax;
    if(pt[1] > ptmax) pt[1] = ptmax;
    if(pt[1] < -ptmax) pt[1] = -ptmax;

  }
}


void MuonDTSeedFromRecHits::computePtWithoutVtx(double* pt, double* spt) const {
  float eta0 = bestEta();

  for (MuonRecHitContainer::const_iterator iter=theRhits.begin(); 
        iter!=theRhits.end(); iter++ ) {
    //+vvp !:
    float eta1= (*iter)->globalPosition().eta(); 
    if ( fabs (eta1-eta0) > .2 ) continue;  //   !!! +vvp

    float radius1= (*iter)->det()->position().perp(); 

    for (MuonRecHitContainer::const_iterator iter2=theRhits.begin(); 
          iter2!=iter; iter2++ ) {

      //+vvp !:
      float eta2= (*iter2)->globalPosition().eta(); 
      if ( fabs (eta2-eta0) > .2 ) continue;  //   !!! +vvp

      float radius2= (*iter2)->det()->position().perp();

      unsigned int stat1(0), stat2(0);

      if ( radius1>450 && radius1<550 ) stat1=2;
      if ( radius1>550 && radius1<650 ) stat1=3;
      if ( radius1>650 ) stat1=4;
      if ( radius1<450 ) stat1=1;

      if ( radius2>450 && radius2<550 ) stat2=2;
      if ( radius2>550 && radius2<650 ) stat2=3;
      if ( radius2<450 ) stat2=1;
      if ( radius2>650 ) stat2=4;


      if ( stat1>stat2) {
        int tmp = stat1;
        stat1 = stat2;
        stat2 = tmp;
      }
      unsigned int st = stat1*10+stat2;
      int index = 0;
      switch (st) {
        case  12 : {//MB1-MB2
          index = 2;
          break;
	}
	case  13 : {//MB1-MB3
          index = 3;
          break;
	}
	case  14 : {//MB1-MB4
          index = 5;
          break;
	}
	case  23 : {//MB2-MB3
          index = 4;
          break;
	}
	case  24 : {//MB2-MB4
          index = 6;
          break;
	}
	case  34 : {//MB3-MB4
          index = 7;
          break;
	}
	default: break;
      }  
      std::vector<double> pts = thePtExtractor->pT_extract(*iter, *iter2);
      if(pts[0]>0 && pts[0] < ptmin) pts[0] = ptmin;
      if(pts[0]<0 && pts[0] > ptmin) pts[0] = -ptmin;

      pt[index] = pts[0];
      spt[index] = pts[1];
    }
  }
}

void MuonDTSeedFromRecHits::computeBestPt(double* pt,
                                           double* spt,
                                           float& ptmean,
                                           float& sptmean) const {

  const std::string metname = "Muon|RecoMuon|MuonDTSeedFromRecHits";

  int nTotal=8;
  int igood=-1;
  for (int i=0;i<=7;i++) {
    if(pt[i]==0) {
      spt[i]=0.;
      nTotal--;
    } else {
      igood=i;
    }
  }

  if (nTotal==1) {
    /// just one pt estimate:  use it.
    ptmean=pt[igood];
    sptmean=1/sqrt(spt[igood]);
  } else if (nTotal==0) {
    // No estimate (e.g. only one rechit!)
    ptmean=50;
    sptmean=30;
  } else {
    /// more than a pt estimate, do all the job.
    // calculate pt with vertex
    float ptvtx=0.0;
    float sptvtx=0.0;
    computeMean(pt, spt, 2, false, ptvtx, sptvtx);
    LogTrace(metname) << " GSL: Pt w vtx :" << ptvtx << "+/-" <<
      sptvtx << endl;
      
    // FIXME: temp hack
    if(ptvtx != 0.) {
      ptmean = ptvtx;
      sptmean = sptvtx;
      return;
      //
    }
    // calculate pt w/o vertex
    float ptMB=0.0;
    float sptMB=0.0;
    computeMean(pt+2, spt+2, 6, false, ptMB, sptMB);
    LogTrace(metname) << " GSL: Pt w/o vtx :" << ptMB << "+/-" <<
        sptMB << endl;

    // decide wheter the muon comes or not from vertex 
    float ptpool=0.0;
    if((ptvtx+ptMB)!=0.0) ptpool=(ptvtx-ptMB)/(ptvtx+ptMB);
    bool fromvtx=true;
    if(fabs(ptpool)>0.2) fromvtx=false; 
    LogTrace(metname) << "From vtx? " <<fromvtx << " ptpool "<< ptpool << endl;

    // now choose the "right" pt => weighted mean
    computeMean(pt, spt, 8, true, ptmean, sptmean);
    LogTrace(metname) << " GSL Pt :" << ptmean << "+/-" << sptmean << endl;

  }
}


void MuonDTSeedFromRecHits::computeMean(const double* pt, const double * weights, int sz,
                                        bool tossOutlyers, float& ptmean, float & sptmean) const
{
  int n=0;
  double ptTmp[8];
  double wtTmp[8];
  assert(sz<=8);

  for (int i=0; i<sz; ++i) {
    ptTmp[i] = 0.;
    wtTmp[i] = 0;
    if (pt[i]!=0) {
      ptTmp[n]=pt[i];
      wtTmp[n]=weights[i];
      ++n;
    }
  }

  if(n != 0) 
  {
    if (n==1) {
      ptmean=ptTmp[0];
      sptmean=1/sqrt(wtTmp[0]);
    } else {
      ptmean = gsl_stats_wmean(wtTmp, 1, ptTmp, 1, n);
      sptmean = sqrt( gsl_stats_wvariance_m (wtTmp, 1, ptTmp, 1, n, ptmean) );
    }

    if(tossOutlyers)
    {
      // Recompute mean with a cut at 3 sigma
      for ( int nm =0; nm<8; nm++ ){
        if ( ptTmp[nm]!=0 && fabs(ptTmp[nm]-ptmean)>3*(sptmean) ) {
          wtTmp[nm]=0.;
        }
      }
      ptmean = gsl_stats_wmean(wtTmp, 1, ptTmp, 1, n);
      sptmean = sqrt( gsl_stats_wvariance_m (wtTmp, 1, ptTmp, 1, n, ptmean) );
    }
  }

}

