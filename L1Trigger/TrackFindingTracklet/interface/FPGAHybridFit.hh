//This class implementes the track fit for the hybrid project
#ifndef FPGAHYBRIDFIT_H
#define FPGAHYBRIDFIT_H

using namespace std;

class FPGAHybridFit{

  public:

    FPGAHybridFit(unsigned int iSector){
      iSector_ = iSector;
    }

    void Fit(FPGATracklet* tracklet, std::vector<std::pair<FPGAStub*,L1TStub*>> &trackstublist){

      std::vector<const TMTT::Stub*> TMTTstubs;
      std::map<unsigned int, L1TStub*> L1StubIndices;
      unsigned int L1stubID = 0;

      static TMTT::Settings* settings = new TMTT::Settings();

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

       /* edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
          iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeometryHandle );

          const TrackerGeometry*  trackerGeometry = trackerGeometryHandle.product();
  */
  /*        edm::ESHandle<TrackerTopology> trackerTopologyHandle;
          iSetup.get<TrackerTopologyRcd>().get(trackerTopologyHandle);/
          const TrackerTopology*  trackerTopology = trackerTopologyHandle.product();
  */        


        if (printDebugKF) cout <<kfphi<<" "<<kfr<<" "<<kfz<<" "<<kfbend<<" "<<kflayer<<" "<<isBarrel<<" "<<psmodule<<" "<<endl;
        TMTT::Stub* TMTTstubptr = new TMTT::Stub(kfphi, kfr, kfz, kfbend, kflayer, psmodule, isBarrel, iphi, -alpha, settings, nullptr, L1stubID);
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

      // IRT bug fix
      //kfphi0 = kfphi0 + iSector_*2*M_PI/NSector - 0.5*dphisectorHG - M_PI;
      kfphi0 = kfphi0 + iSector_*2*M_PI/NSector - 0.5*dphisectorHG;

      if (kfphi0>M_PI) kfphi0-=2*M_PI;
      if (kfphi0<-M_PI) kfphi0+=2*M_PI;

      std::pair<unsigned int, unsigned int> celllocation(1,1);
      //    std::pair<float,float> helixrphi(300*kfrinv/settings->getBfield(),kfphi0);
      std::pair<float,float> helixrphi(kfrinv*1.0e11/(2.9979e8*settings->getBfield()),kfphi0);
      std::pair<float,float> helixrz(kfz0,kft);

      //  TMTT phi sector definition: phiCentre_ = 2.*M_PI * (0.5 + float(iPhiSec)) / float(settings->numPhiSectors()) - M_PI; // Centre of sector in phi

      unsigned int kf_eta_reg;

      /*   
  72:  float   tanLambda()  const  {return helixRz_.second;}
  73:  float   theta()      const  {return atan2(1., this->tanLambda());} // Use atan2 to ensure 0 < theta < pi.
  78:  float   zAtChosenR()   const  {return (this->z0() + (settings_->chosenRofZ()) * this->tanLambda());} // neglects transverse impact parameter & track curvature.
  */

  // Get range in z of tracks covered by this sector at chosen radius from beam-line

      const vector<double> etaRegions = settings->etaRegions();
      const float chosenRofZ = settings->chosenRofZ();

      float  kfzRef = kfz0 + chosenRofZ*kft;

      kf_eta_reg = 0;
      for (unsigned int iEtaSec = 1; iEtaSec < etaRegions.size() - 1; iEtaSec++) { // Doesn't apply eta < 2.4 cut.
        const float etaMax = etaRegions[iEtaSec];
	const float zRefMax = chosenRofZ / tan( 2. * atan(exp(-etaMax)) );
	if (kfzRef > zRefMax) kf_eta_reg = iEtaSec;
      }

      int kf_phi_sec=tracklet->homeSector() -3;
      if(kf_phi_sec < 0){kf_phi_sec+=9;}      


      TMTT::L1track3D l1track3d(settings,TMTTstubs,celllocation,helixrphi,helixrz,-kfd0,kf_phi_sec,kf_eta_reg,1,false); //remember to change after fixing the d0 sign convention.
      unsigned int seedType = tracklet->getISeed();
      unsigned int numPS = tracklet->PSseed(); // Function PSseed() is out of date!
      l1track3d.setSeedLayerType(seedType);
      l1track3d.setSeedPS(numPS);

      // Create Kalman track fitter.
      static bool firstPrint = true;
  #ifdef USE_HLS
      if (firstPrint) cout << "Will make KFParamsCombHLS for " << nHelixPar << " param fit" << endl;
      static TMTT::TrackFitGeneric* fitterKF = new TMTT::KFParamsCombCallHLS(settings, nHelixPar, "KFfitterHLS");
  #else
      if (firstPrint) cout << "Will make KFParamsComb for " << nHelixPar << " param fit"<< endl;
      static TMTT::TrackFitGeneric* fitterKF = new TMTT::KFParamsComb(settings, nHelixPar, "KFfitter");
  #endif
      firstPrint = false;

      //  cout << "Will call fit" << endl;
      //fitterKF->fit(l1track3d,1,kf_eta_reg);

      TMTT::L1fittedTrack fittedTrk = fitterKF->fit(l1track3d); 
     
      TMTT::KFTrackletTrack trk = fittedTrk.returnKFTrackletTrack();

      if (printDebugKF) cout << "Done with Kalman fit. Pars: pt = " << trk.pt() << ", 1/2R = " << 3.8*3*trk.qOverPt()/2000 << ", phi0 = " << trk.phi0() << ", eta = " << trk.eta() << ", z0 = " << trk.z0() << ", chi2 = "<<trk.chi2()  << ", accepted = "<< trk.accepted() << endl;

      // IRT bug fix
      //double tracklet_phi0=M_PI+trk.phi0()-iSector_*2*M_PI/NSector+0.5*dphisectorHG;
      double tracklet_phi0=trk.phi0()-iSector_*2*M_PI/NSector+0.5*dphisectorHG;

      if (tracklet_phi0>M_PI) tracklet_phi0-=2*M_PI;
      if (tracklet_phi0<-M_PI) tracklet_phi0+=2*M_PI;

      double rinvfit=0.01*0.3*settings->getBfield()*trk.qOverPt();

      int id0fit   = trk.d0()  / kd0;

      if(trk.accepted()){

        const vector<const TMTT::Stub*>& stubsFromFit = trk.getStubs();
        vector<L1TStub*> l1stubsFromFit;
        for (const TMTT::Stub* s : stubsFromFit) {
            unsigned int IDf = s->index();
            L1TStub* l1s = L1StubIndices.at(IDf);
            l1stubsFromFit.push_back(l1s);
        }

        if (printDebugKF) cout<<"#stubs before/after KF fit = "<<TMTTstubs.size()<<"/"<<l1stubsFromFit.size()<<endl;

       tracklet->setFitPars(rinvfit,tracklet_phi0,trk.d0(),sinh(trk.eta()),trk.z0(),
         trk.chi2(),rinvfit,tracklet_phi0, trk.d0(), sinh(trk.eta()),
         trk.z0(),trk.chi2(),rinvfit/krinvpars,
         tracklet_phi0/kphi0pars,id0fit,
         sinh(trk.eta())/ktpars,trk.z0()/kz0pars,trk.chi2(),l1stubsFromFit);
       //cout<<" KF fit d0 is "<<trk.d0()<<"\n";
      } else {
       if (printDebugKF) cout << "FPGAFitTrack:KF rejected track"<<endl;
      }
    }

  private:
    unsigned int iSector_;

};

#endif
