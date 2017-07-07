
#ifndef jhugon_TFTrack_h
#define jhugon_TFTrack_h
// system include files
#include <vector>
#include <string>

#include <FWCore/Framework/interface/EventSetup.h>

#include <L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h>

#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"
#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h>
#include <DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h>

#include <L1Trigger/CSCTrackFinder/test/src/Track.h>

namespace csctf_analysis
{
  class TFTrack : public Track
  {
    public:
	TFTrack();
	TFTrack(const L1MuRegionalCand& track);
	TFTrack(const L1CSCTrack& track, const edm::EventSetup& iSetup );
	TFTrack(L1MuGMTExtendedCand track);
//	double distanceTo(RefTrack* reftrack);
        void print() override;
	double getPt() const override {return Pt;};
	double getPhi() const override {return Phi;};
	double getEta() const override {return Eta;};
	double getTFPt() const override {return Pt;};
	double getRank() const override {return Rank;};
	int getMode() const override {return Mode;};  
	int getPtPacked() const override {return PtPacked;};
	int getEtaPacked() const override {return EtaPacked;};
	int getPhiPacked() const override {return PhiPacked;};
	int getChargePacked() const override {return ChargePacked;};
	int getFR() const override {return FR;};
	int getBX() const override {return Bx;};
	int getLUTAddress() const override {return LUTAddress;}
	int getEndcap() const {if(isEndcap1==true){return 1;} else{return 2;}}
	//added by josh and nathaniel
    private:
	float Pt;
	double Phi;
	double Eta;
	int PtPacked;
	int EtaPacked;
	int PhiPacked;
	int ChargePacked;
	int Bx;
	int Charge;
	int Halo;
	int Mode;
	int Rank;
	int FR;
	int LUTAddress;
	bool isEndcap1;
  };
}
#endif
