#ifndef ROOT_FPGAEvent
#define ROOT_FPGAEvent

#include <vector>
#include <map>
#include <TROOT.h>
#include <TMath.h>

class FPGAEventL1Stub{

  public:

    int eventid_;
    int simtrackid_;
    unsigned int iphi_;
    unsigned int iz_;
    unsigned int layer_;
    unsigned int ladder_;
    unsigned int module_;
    unsigned int strip_;
    double x_;
    double y_;
    double z_;
    double sigmax_;
    double sigmaz_;
    double pt_;
    double bend_;

    unsigned int isPSmodule_;
    unsigned int isFlipped_;

    int allStubIndex_;

    FPGAEventL1Stub() { }

    FPGAEventL1Stub(int eventid, int simtrackid, int iphi, int iz, int layer, int ladder, int module, int strip, double x, double y, double z, double sigmax, double sigmaz, double pt, double bend, int isPSmodule, int isFlipped, int allStubIndex) {
      eventid_=eventid; // eventid=0: stub from hard scatter, eventid>0: stub from PU interaction, eventid=-1: stub not matched to MC truth
      simtrackid_=simtrackid;
      iphi_=iphi;
      iz_=iz;
      layer_=layer;
      ladder_=ladder;
      module_=module;
      strip_=strip;
      x_=x;
      y_=y;
      z_=z;
      sigmax_=sigmax;
      sigmaz_=sigmaz;
      pt_=pt;
      bend_ = bend;
      isPSmodule_ = isPSmodule;
      isFlipped_ = isFlipped;
      allStubIndex_ = allStubIndex;

      /*
      if (layer_>999&&z_<0.0) {
        //cout <<"Flipping pt sign"<<endl;
        pt_=-pt_;
        bend_ = -bend_;
      }
      */
    }

       
    virtual ~FPGAEventL1Stub() { }
    
    ClassDef(FPGAEventL1Stub,1)

};

class FPGAEventStub : public TObject {

  public:
    int nSector_;
    int layer_;
    int disk_;
    int iphi_;
    int ir_;
    int iz_;
    int stubID_;
//    FPGAEventL1Stub* L1TStub_;
    int stubpt_;
    double r_;
    double phi_;
    double z_;
    double rr_;
    double rphi_;
    double rz_;
    double rpt_; 
    
    FPGAEventStub() { }
	    
//    FPGAEventStub(int nSector, int layer, int disk, int stubID, FPGAEventL1Stub* L1TStub, int stubpt, int ir, int iphi, int iz, double r, double phi, double z, double rr, double rphi, double rz, double rpt) {
    FPGAEventStub(int nSector, int layer, int disk, int stubID, int stubpt, int ir, int iphi, int iz, double r, double phi, double z, double rr, double rphi, double rz, double rpt) {
      nSector_ =nSector;
      layer_   =layer;
      disk_    =disk;
      stubID_  =stubID;
//      L1TStub_ =L1TStub;
      stubpt_  =stubpt;
      ir_      =ir;  
      iphi_    =iphi;
      iz_      =iz;
      r_       =r;
      phi_     =phi;
      z_       =z;
      rr_      =rr;
      rphi_    =rphi;
      rz_      =rz;
      rpt_     =rpt;
    }
       
    virtual ~FPGAEventStub() { }
    
    ClassDef(FPGAEventStub,1)
    
};


class FPGAEventTrack : public TObject {

  public:

    int irinv_;
    int iphi0_;
    int iz0_;
    int it_;
    int ichisq_;

    std::vector<std::pair<int, int>> stubIDpremerge_;
    std::vector<std::pair<int, int>> stubIDprefit_;
    std::map<int, int> stubID_;

    int seed_;
    bool duplicate_;
    int sector_;

    double pt_;
    double phi0_;
    double eta_;
    double z0_;
    double rinv_;
    double chisq_;

    FPGAEventTrack() { }

    FPGAEventTrack(int irinv, int iphi0, int iz0, int it, int ichisq,
                   std::vector<std::pair<int, int>> stubIDpremerge,
                   std::vector<std::pair<int, int>> stubIDprefit,
                   std::map<int, int> stubID,
                   int seed, bool duplicate, int sector, 
                   double pt, double phi0, double eta, double z0, double rinv, double chisq ) {
        irinv_=irinv;
        iphi0_=iphi0;
        iz0_=iz0;
        it_=it;
        ichisq_=ichisq;

        stubIDpremerge_=stubIDpremerge;
        stubIDprefit_=stubIDprefit;
        stubID_=stubID;

        seed_=seed;
        duplicate_=duplicate;
        sector_=sector;

        pt_=pt;
        phi0_=phi0;
        eta_=eta;
        z0_=z0;
        rinv_=rinv;
        chisq_=chisq;
    }

    virtual ~FPGAEventTrack() { }

    ClassDef(FPGAEventTrack,1)

};


class FPGAEventMCTrack : public TObject {
  
  public:

    int type_;
    double pt_;
    double eta_;
    double phi_;
    double vx_;
    double vy_;
    double vz_;
    float  charge_;
    
    FPGAEventMCTrack() { }
	     
    FPGAEventMCTrack(int type, double pt, double eta, double phi, double vx, double vy, double vz) {
      type_=type;
      pt_  = pt;
      eta_ = eta;
      phi_ = phi;
      vx_  = vx;
      vy_  = vy;
      vz_  = vz;
      charge_ = getCharge(type);
    }
    
    float getCharge(int type) {
      // this is stupid...should perhaps put it in the input file
      // all comes down to particle vs antiparticle choice
      if(fabs(type)==11)   return  (float) -type/abs(type); //electron
      if(fabs(type)==13)   return  (float) -type/abs(type); //muon
      if(fabs(type)==211)  return  (float) +type/abs(type); //charged pions
      if(fabs(type)==321)  return  (float) +type/abs(type); //charged kaons 
      if(fabs(type)==2212) return  (float) +type/abs(type); //protons
      
      //neutrals
      if(fabs(type)==12 || fabs(type)==14 || fabs(type)==16) return 0.; //neutrinos   
      if(fabs(type)==111) return 0.; //pi zero 
      if(fabs(type)==2112) return 0.; //neutrons
      
      // send unknown 
      return 99.;
    }	  
    
    virtual ~FPGAEventMCTrack() { }

    ClassDef(FPGAEventMCTrack,1)

};


class FPGAEventTrackMatch : public TObject {
  
  public:
    bool   goodMatch_;
    double deltaPt_;
    double deltaEta_;
    double deltaPhi_;
    double deltaR_;
    double deltaZ0_;
    
    FPGAEventTrackMatch() { }
    
    FPGAEventTrackMatch(FPGAEventMCTrack mcTrack, FPGAEventTrack fpgaTrack) {

      goodMatch_   = false;
      deltaPt_     = fpgaTrack.pt_  - mcTrack.charge_*mcTrack.pt_; //mc pt value is positive regardless of charge
      deltaEta_    = fpgaTrack.eta_ - mcTrack.eta_;
      deltaPhi_    = fpgaTrack.phi0_ - mcTrack.phi_;
      if(deltaPhi_ > TMath::Pi())  deltaPhi_ -= TMath::TwoPi();
      if(deltaPhi_ < -TMath::Pi()) deltaPhi_ += TMath::TwoPi();
      deltaR_      = sqrt(deltaEta_*deltaEta_ + deltaPhi_*deltaPhi_);
      deltaZ0_     = fpgaTrack.z0_ - mcTrack.vz_;

    }
    void setGoodMatch(bool flag) {goodMatch_ = flag;}
	    
    virtual ~FPGAEventTrackMatch() { }

    ClassDef(FPGAEventTrackMatch,1)
    
};


class FPGAEvent : public TObject {
  
  public:

    FPGAEvent() {	}

    virtual ~FPGAEvent() { }

    void reset() {
      nevt = -1;
      tracks.clear();
      mcTracks.clear();
      stubs.clear();
    }
      
    int nevt;
    int dummy;

    std::vector<FPGAEventTrack> tracks;
    std::vector<FPGAEventMCTrack> mcTracks;
    std::vector<FPGAEventStub> stubs;
    std::vector<FPGAEventL1Stub> l1stubs;

    ClassDef(FPGAEvent,1)
};

#endif
