/**
 *  See header file for a description of this class.
 *
 *
 *  $Date: 2007/09/11 03:11:32 $
 *  $Revision: 1.3 $
 *  \author A. Vitelli - INFN Torino, V.Palichik
 *  \author porting  R. Bellan
 *
 */
#include "RecoMuon/MuonSeedGenerator/src/MuonDTSeedFromRecHits.h"

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


template <class T> T sqr(const T& t) {return t*t;}

MuonDTSeedFromRecHits::MuonDTSeedFromRecHits(const edm::EventSetup& eSetup)
: MuonSeedFromRecHits(eSetup)
{
}


TrajectorySeed MuonDTSeedFromRecHits::seed() const {
  double pt[8] = { 0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 };
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

void MuonDTSeedFromRecHits::computePtWithVtx(double* pt, double* spt) const {


//+vvp ! Try to search group of nearest segm-s:

  int Maxseg = 0;
  float Msdeta = 100.;
  float eta0 = (*theRhits.begin())->globalPosition().eta();

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
	eta0 = eta1;
      }

    }
  }   //  +v.


  for (MuonRecHitContainer::const_iterator iter=theRhits.begin(); iter!=theRhits.end(); iter++ ) {

 //+vvp !:
      float eta1= (*iter)->globalPosition().eta(); 

      if ( fabs (eta1-eta0) > .2 ) continue;  //   !!! +vvp

    // assign Pt from MB1 & vtx   
    float radius = (*iter)->det()->position().perp();
    unsigned int stat = 0;
    if ( radius>450 && radius<550 ) stat=2;
    if ( radius<450 ) stat=1;
    if( stat==1 ) {
      GlobalPoint pos = (*iter)->globalPosition();

      GlobalVector dir = (*iter)->globalDirection();
      
      float dphi = -pos.phi()+dir.phi();
      if(dphi>M_PI) dphi=2*M_PI-dphi;
      pt[0]=1.0-1.46/dphi; 
      if ( abs(pos.z()) > 500 ) {
        // overlap 
        float a1 = dir.y()/dir.x(); float a2 = pos.y()/pos.x();
        dphi = fabs((a1-a2)/(1+a1*a2));

        float new_pt = fabs(-3.3104+(1.2373/dphi));
        pt[0] = new_pt*pt[0]/fabs(pt[0]);
      }
    }
    // assign Pt from MB2 & vtx
    if( stat==2 ) {
      GlobalPoint pos = (*iter)->globalPosition();

      GlobalVector dir = (*iter)->globalDirection();
      
      float dphi = -pos.phi()+dir.phi();
      if(dphi>M_PI) dphi=2*M_PI-dphi;
      pt[1]=1.0-0.9598/dphi;
      if ( abs(pos.z()) > 600 ) {
        // overlap 
        float a1 = dir.y()/dir.x(); float a2 = pos.y()/pos.x();
        dphi = fabs((a1-a2)/(1+a1*a2));

        float new_pt = fabs(10.236+(0.5766/dphi));
        pt[1] = new_pt*pt[1]/fabs(pt[1]);
      }
    }
    float ptmax = 2000.;
    if(pt[0] > ptmax) pt[0] = ptmax;
    if(pt[0] < -ptmax) pt[0] = -ptmax;
    if(pt[1] > ptmax) pt[1] = ptmax;
    if(pt[1] < -ptmax) pt[1] = -ptmax;

  }
}

void MuonDTSeedFromRecHits::computePtWithoutVtx(double* pt, double* spt) const {



 //+vvp ! Try to search group of nearest segm-s:

  int Maxseg = 0;
  float Msdeta = 100.;
  float eta0 = (*theRhits.begin())->globalPosition().eta();

  for (MuonRecHitContainer::const_iterator iter=theRhits.begin(); 
        iter!=theRhits.end(); iter++ ) {

    float eta1= (*iter)->globalPosition().eta(); 

    int Nseg = 0;
    float sdeta = .0;

    for (MuonRecHitContainer::const_iterator iter2=theRhits.begin(); 
          iter2!=theRhits.end(); iter2++ ) {

      if ( iter2 == iter )  continue;

      float eta2 = (*iter2)->globalPosition().eta(); 

      if ( fabs (eta1-eta2) > .2 ) continue;

      Nseg++;
      sdeta += fabs (eta1-eta2); 

      if ( Nseg > Maxseg ||  Nseg == Maxseg && sdeta < Msdeta ) {
	Maxseg = Nseg;
	Msdeta = sdeta;
	eta0 = eta1;
      }

    }
  }   //  +v.


  int ch=0;

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

      GlobalVector globalDir1 = (*iter)->globalDirection();
      GlobalVector globalDir2 = (*iter2)->globalDirection();
      float dphi = -globalDir1.phi()+globalDir2.phi();
      // Maybe these aren't necessary with Geom::Phi
      if(dphi>M_PI) dphi -= 2*M_PI;
      if(dphi<-M_PI) dphi += 2*M_PI;

      ch = (dphi > 0) ? 1 : -1;

      if ( stat1>stat2) {
        ch = -ch
        int tmp = stat1;
        stat1 = stat2;
        stat2 = tmp;
      }
      unsigned int st = stat1*10+stat2;

      if ( dphi ) {
        dphi = fabs(dphi);
        switch (st) {
	case  12 : {//MB1-MB2
	  pt[2]=(12.802+0.38647/dphi)*ch ; 
	  GlobalPoint pos_iter = (*iter)->globalPosition();
	  if (  (*iter)->det()->position().perp() <450 ) {
	    if ( fabs(pos_iter.z())>500. ) {
	      pt[2]=(12.802+0.16647/dphi)*ch ; 
	    }
	  } else {
	    if ( fabs(pos_iter.z())>600. ) {
	      pt[2]=(12.802+0.16647/dphi)*ch ; 
	    }
	  }
	  ;break;
	}
	case  13 : {//MB1-MB3
	  pt[3]=(.0307+0.99111/dphi)*ch ; ;break;
	}
	case  14 : {//MB1-MB4
	  pt[5]=(2.7947+1.1991/dphi)*ch ; ;break;
	}
	case  23 : {//MB2-MB3
	  pt[4]=(2.4583+0.69044/dphi)*ch ;;break; 
	}
	case  24 : {//MB2-MB4
	  pt[6]=(2.5267+1.1375/dphi)*ch ; ;break; 
	}
	case  34 : {//MB3-MB4
	  pt[7]=(4.06444+0.59189/dphi)*ch ; ;break;
	}
	default: break;
        }  
      }
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
    if(pt[0]!=0. || pt[1]!=0.) {
      float ptTmp[2]={0.,0.};
      float sptTmp[2]={0.,0.};
      int n=0;
      for (int i=0; i<2; ++i) {
        if (pt[i]!=0) {
          ptTmp[n]=pt[i];
          sptTmp[n]=spt[i];
          n++;
        }
      }
      if (n==1) {
        ptvtx=ptTmp[0];
        sptvtx=1/sqrt(sptTmp[0]);
      } else {
        ptvtx = gsl_stats_wmean(spt, 1, pt, 1, n);
        sptvtx = gsl_stats_wvariance_m (spt, 1, pt, 1, n, ptvtx);
        sptvtx = sqrt(sptvtx);
      }
      LogTrace(metname) << " GSL: Pt w vtx :" << ptvtx << "+/-" <<
        sptvtx << endl;
      
      // FIXME: temp hack
      ptmean = ptvtx;
      sptmean = sptvtx;
      return;
      //
    }
    // calculate pt w/o vertex
    float ptMB=0.0;
    float sptMB=0.0;
    if(pt[2]!=0. || pt[3]!=0. || pt[4]!=0. || pt[5]!=0. || pt[6]!=0. || pt[7]!=0. ) {
      int n=0;
      float ptTmp[6]={ 0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0};
      float sptTmp[6]={ 0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0};
      for (int i=2; i<8; ++i) {
        if (pt[i]!=0) {
          ptTmp[n]=pt[i];
          sptTmp[n]=spt[i];
          n++;
        }
      }
      if (n==1) {
        ptMB=ptTmp[0];
        sptMB=1/sqrt(sptTmp[0]);
      } else {
        ptMB = gsl_stats_wmean(&spt[2], 1, &pt[2], 1, n);
        sptMB = gsl_stats_wvariance_m (&spt[2], 1, &pt[2], 1, n, ptMB);
        sptMB = sqrt(sptMB);
      }
      LogTrace(metname) << " GSL: Pt w/o vtx :" << ptMB << "+/-" <<
        sptMB << endl;
    }

    // decide wheter the muon comes or not from vertex 
    float ptpool=0.0;
    if((ptvtx+ptMB)!=0.0) ptpool=(ptvtx-ptMB)/(ptvtx+ptMB);
    bool fromvtx=true;
    if(fabs(ptpool)>0.2) fromvtx=false; 
    LogTrace(metname) << "From vtx? " <<fromvtx << " ptpool "<< ptpool << endl;

    // now choose the "right" pt => weighted mean
    int n=0;
    double ptTmp[8]={ 0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0, 0.0};
    double sptTmp[8]={ 0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0, 0.0};
    for (int i=0; i<8; ++i) {
      if (pt[i]!=0) {
        ptTmp[n]=pt[i];
        sptTmp[n]=spt[i];
        n++;
      }
    }
    if (n==1) {
      ptmean=ptTmp[0];
      sptmean=1/sqrt(sptTmp[0]);
    } else {
      ptmean = gsl_stats_wmean(sptTmp, 1, ptTmp, 1, n);
      sptmean = gsl_stats_wvariance_m (sptTmp, 1, ptTmp, 1, n, ptmean);
      sptmean = sqrt(sptmean);
    }
    LogTrace(metname) << " GSL Pt :" << ptmean << "+/-" << sptmean << endl;

    // Recompute mean with a cut at 3 sigma
    for ( int nm =0; nm<8; nm++ ){
      if ( ptTmp[nm]!=0 && fabs(ptTmp[nm]-ptmean)>3*(sptmean) ) {
        sptTmp[nm]=0.;
      }
    }  
    ptmean = gsl_stats_wmean(sptTmp, 1, ptTmp, 1, n);
    sptmean = gsl_stats_wvariance_m (sptTmp, 1, ptTmp, 1, n, ptmean);
    sptmean = sqrt(sptmean);
    LogTrace(metname) << " GSL recomp Pt :" << ptmean << "+/-" << sptmean << endl;
  }
}

