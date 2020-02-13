//This class implementes the track fit for the hybrid project
#ifndef HYBRIDFIT_H
#define HYBRIDFIT_H

#ifdef USEHYBRID
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"
#include "L1Trigger/TrackFindingTracklet/interface/HybridFit.h"
#ifdef USE_HLS
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFParamsCombCallHLS.h"
#endif
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFTrackletTrack.h"
#endif

using namespace std;

class HybridFit{

  public:

    HybridFit(unsigned int iSector, bool extended, unsigned int nHelixPar){
      iSector_ = iSector;
      extended_ = extended;
      nHelixPar_ = nHelixPar;
    }

    void Fit(Tracklet* tracklet, std::vector<std::pair<Stub*,L1TStub*>> &trackstublist){

      std::vector<const TMTT::Stub*> TMTTstubs;
      std::map<unsigned int, L1TStub*> L1StubIndices;
      unsigned int L1stubID = 0;

      static const TMTT::Settings settings;

      int kf_phi_sec=iSector_;

      for (unsigned int k=0;k<trackstublist.size();k++) {
        L1TStub* L1stubptr=trackstublist[k].second;

        double kfphi=L1stubptr->phi();
        double kfr=L1stubptr->r();
        double kfz=L1stubptr->z();
        double kfbend=L1stubptr->bend();
        bool psmodule = L1stubptr->isPSmodule();
        unsigned int iphi = L1stubptr->iphi();
        double alpha = L1stubptr->alpha();

        bool isBarrel = trackstublist[k].first->isBarrel();
        int kflayer;

        // Barrel-specific
        if (isBarrel) {
          kflayer=L1stubptr->layer()+1;
          if (printDebugKF) cout << "Will create layer stub with : ";

        // Disk-specific
        } else {
          kflayer=abs(L1stubptr->disk());
          if (kfz>0) {
            kflayer+=10;
          } else {
            kflayer+=20;
          }
          if (printDebugKF) cout << "Will create disk stub with : ";
        }

        if (printDebugKF) cout <<kfphi<<" "<<kfr<<" "<<kfz<<" "<<kfbend<<" "<<kflayer<<" "<<isBarrel<<" "<<psmodule<<" "<<endl;
        TMTT::Stub* TMTTstubptr = new TMTT::Stub(kfphi, kfr, kfz, kfbend, kflayer, psmodule, isBarrel, iphi, -alpha, &settings, nullptr, L1stubID, kf_phi_sec);
        TMTTstubs.push_back(TMTTstubptr);
        L1StubIndices[L1stubID++] = L1stubptr;
      }

      if (printDebugKF) cout << "Made TMTTstubs: trackstublist.size() = " << trackstublist.size()<< endl;


      double kfrinv=tracklet->rinvapprox();
      double kfphi0=tracklet->phi0approx();
      double kfz0=tracklet->z0approx();
      double kft=tracklet->tapprox();
      double kfd0=tracklet->d0approx();

      if (printDebugKF) {
       std::cout << "tracklet phi0 = "<< kfphi0 << std::endl;
       std::cout << "iSector = " << iSector_ << std::endl;
       std::cout << "dphisectorHG = " << dphisectorHG << std::endl;
      }

      // KF wants global phi0, not phi0 measured with respect to lower edge of sector (Tracklet convention).
      kfphi0 = kfphi0 + iSector_*2*M_PI/NSector - 0.5*dphisectorHG;

      if (kfphi0>M_PI) kfphi0-=2*M_PI;
      if (kfphi0<-M_PI) kfphi0+=2*M_PI;

      std::pair<float,float> helixrphi(kfrinv*1.0e11/(2.9979e8*settings.getBfield()),kfphi0);
      std::pair<float,float> helixrz(kfz0,kft);

      // KF HLS uses HT mbin (which is binned q/Pt) to allow for scattering. So estimate it from tracklet.
      double chargeOverPt = helixrphi.first;
      int mBin = std::floor(settings.houghNbinsPt()/2) + std::floor((settings.houghNbinsPt()/2) * chargeOverPt/(1./settings.houghMinPt()));
      mBin = max(min(mBin, int(settings.houghNbinsPt()-1)), 0); // protect precision issues.
      std::pair<unsigned int, unsigned int> celllocation(mBin,1);

      // Get range in z of tracks covered by this sector at chosen radius from beam-line

      const vector<double> etaRegions = settings.etaRegions();
      const float chosenRofZ = settings.chosenRofZ();

      float  kfzRef = kfz0 + chosenRofZ*kft;

      unsigned int kf_eta_reg = 0;
      for (unsigned int iEtaSec = 1; iEtaSec < etaRegions.size() - 1; iEtaSec++) { // Doesn't apply eta < 2.4 cut.
        const float etaMax = etaRegions[iEtaSec];
	const float zRefMax = chosenRofZ / tan( 2. * atan(exp(-etaMax)) );
	if (kfzRef > zRefMax) kf_eta_reg = iEtaSec;
      }

      TMTT::L1track3D l1track3d(&settings,TMTTstubs,celllocation,helixrphi,helixrz,kfd0,kf_phi_sec,kf_eta_reg,1,false);
      unsigned int seedType = tracklet->getISeed();
      unsigned int numPS = tracklet->PSseed(); // Function PSseed() is out of date!
      l1track3d.setSeedLayerType(seedType);
      l1track3d.setSeedPS(numPS);

      // Create Kalman track fitter.
      static bool firstPrint = true;
#ifdef USE_HLS
      if (firstPrint) cout << "Will make KFParamsCombHLS for " << nHelixPar_ << " param fit" << endl;
      static thread_local TMTT::KFParamsCombCallHLS fitterKF(&settings, nHelixPar_, "KFfitterHLS");
#else
      if (firstPrint) cout << "Will make KFParamsComb for " << nHelixPar_ << " param fit"<< endl;
      static thread_local TMTT::KFParamsComb fitterKF(&settings, nHelixPar_, "KFfitter");
#endif
      firstPrint = false;

      //  cout << "Will call fit" << endl;

      TMTT::L1fittedTrack fittedTrk = fitterKF.fit(l1track3d); 
     
      TMTT::KFTrackletTrack trk = fittedTrk.returnKFTrackletTrack();

      if (printDebugKF) cout << "Done with Kalman fit. Pars: pt = " << trk.pt() << ", 1/2R = " << 3.8*3*trk.qOverPt()/2000 << ", phi0 = " << trk.phi0() << ", eta = " << trk.eta() << ", z0 = " << trk.z0() << ", chi2 = "<<trk.chi2()  << ", accepted = "<< trk.accepted() << endl;

      // Tracklet wants phi0 with respect to lower edge of sector, not global phi0.
      double phi0fit=trk.phi0()-iSector_*2*M_PI/NSector+0.5*dphisectorHG;

      if (phi0fit>M_PI) phi0fit-=2*M_PI;
      if (phi0fit<-M_PI) phi0fit+=2*M_PI;

      double rinvfit=0.01*0.3*settings.getBfield()*trk.qOverPt();

      int irinvfit = rinvfit / krinvpars;
      int iphi0fit = phi0fit / kphi0pars;
      int itanlfit = trk.tanLambda() / ktpars;
      int iz0fit   = trk.z0() / kz0pars;
      int id0fit   = trk.d0() / kd0pars;
      int ichi2fit = trk.chi2() / 16;  // CHECK THIS (but not used to make TTTrack)

      if (trk.accepted()) {

        const vector<const TMTT::Stub*>& stubsFromFit = trk.getStubs();
        vector<L1TStub*> l1stubsFromFit;
        for (const TMTT::Stub* s : stubsFromFit) {
            unsigned int IDf = s->index();
            L1TStub* l1s = L1StubIndices.at(IDf);
            l1stubsFromFit.push_back(l1s);
        }

        if (printDebugKF) cout<<"#stubs before/after KF fit = "<<TMTTstubs.size()<<"/"<<l1stubsFromFit.size()<<endl;

	// TO DO: update setFitPars() args, adding trk.getHitPattern() 
	// & replacing trk.chi2() by trk.chi2rphi() & trk.chi2rz().

	tracklet->setFitPars(rinvfit,phi0fit,trk.d0(),trk.tanLambda(),trk.z0(),trk.chi2(),
			     rinvfit,phi0fit,trk.d0(),trk.tanLambda(),trk.z0(),trk.chi2(),
			     irinvfit,iphi0fit,id0fit,itanlfit,iz0fit,ichi2fit,l1stubsFromFit);
      }
      else {
	if (printDebugKF) cout << "FitTrack:KF rejected track"<<endl;
      }

      for (const TMTT::Stub* s : TMTTstubs) {
	delete s;
      }
      
    }

  private:
    unsigned int iSector_;
    bool extended_;
    unsigned int nHelixPar_;
    
};

#endif
