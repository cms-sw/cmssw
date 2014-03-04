
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "L1Trigger/CSCTrackFinder/test/src/TFTrack.h"
#include "iostream"
namespace csctf_analysis
{
  TFTrack::TFTrack():Track() {}
  TFTrack::TFTrack(const L1MuRegionalCand& track):Track()
  {
	
  const float ptscale[33] = { 
  	-1.,   0.0,   1.5,   2.0,   2.5,   3.0,   3.5,   4.0,
    4.5,   5.0,   6.0,   7.0,   8.0,  10.0,  12.0,  14.0,  
    16.0,  18.0,  20.0,  25.0,  30.0,  35.0,  40.0,  45.0, 
    50.0,  60.0,  70.0,  80.0,  90.0, 100.0, 120.0, 140.0, 1.E6 };

	Phi = (2.5*( track.phi_packed() ))*(M_PI)/180 + 0.0218;
	Eta = 0.9 + 0.05*( track.eta_packed() ) +0.025;
	PhiPacked = track.phi_packed();
	EtaPacked = track.eta_packed();
	PtPacked = track.pt_packed();
	Pt = ptscale[PtPacked];
	Quality = track.quality_packed();
	Bx = track.bx();
	Halo = track.finehalo_packed();
	ChargePacked = track.charge_packed();
	if (ChargePacked==1)
		Charge=-1;
	else 
		Charge=1;
	Rank = -1;
	Mode = -1;
	FR = -1;
	LUTAddress = -1;
   
  }
  TFTrack::TFTrack(const L1CSCTrack& track, 
		const edm::EventSetup& iSetup ):Track()
  {

  const float ptscale[33] = { 
  	-1.,   0.0,   1.5,   2.0,   2.5,   3.0,   3.5,   4.0,
    4.5,   5.0,   6.0,   7.0,   8.0,  10.0,  12.0,  14.0,  
    16.0,  18.0,  20.0,  25.0,  30.0,  35.0,  40.0,  45.0, 
    50.0,  60.0,  70.0,  80.0,  90.0, 100.0, 120.0, 140.0, 1.E6 };

    //unsigned int endcap = track.first.endcap();//get the encap
    unsigned int sector = track.first.sector();// get sector
    Rank = track.first.rank();// get rank

    unsigned int quality_packed;
    unsigned int rank=Rank;
    unsigned int pt_packed;
	
    track.first.decodeRank(rank,pt_packed,quality_packed); //get the pt and gaulity packed
    Quality=quality_packed;
    PtPacked=pt_packed;
    Pt = ptscale[PtPacked]; 

    edm::ESHandle< L1MuTriggerScales > scales;//get structures for scales (phi and eta
    iSetup.get< L1MuTriggerScalesRcd >().get(scales); // get scales from EventSetup

    const L1MuTriggerScales  *ts;// the trigger scales 
    ts = scales.product();
    
    unsigned gbl_phi = track.first.localPhi() + ((sector - 1)*24) + 6;
    if(gbl_phi > 143) gbl_phi -= 143;
    float phi = ts->getPhiScale()->getLowEdge( gbl_phi&0xff );

    Mode =  track.first.mode();

    // To throw an error if the newer fixed L1Track::mode() isn't implemented
    //    int AddressEta = track.first.addressEta();

    Phi = phi;
    PhiPacked = track.first.localPhi();

    unsigned eta_sign = (track.first.endcap() == 1 ? 0 : 1);
    Eta = ts->getRegionalEtaScale(2)->
	getCenter( ((track.first.eta_packed()) | (eta_sign<<5)) & 0x3f );
    EtaPacked = track.first.eta_packed();

    Bx = track.first.bx();
    Halo = track.first.finehalo_packed();
    ChargePacked = track.first.charge_packed();
    if (ChargePacked==1)
		Charge=-1;
    else 
		Charge=1;

    LUTAddress = track.first.ptLUTAddress();
    FR = (track.first.ptLUTAddress() >> 21 ) & 0x1;

/*
	Phi = (2.5*( track.phi_packed() ))*(M_PI)/180 + 0.0218;
	Eta = 0.9 + 0.05*( track.eta_packed() ) +0.025;
	PtPacked = track.pt_packed();
	Pt = ptscale[PtPacked];
	Quality = track.quality_packed();
	Bx = track.bx();
	Halo = track.finehalo_packed();
	ChargePacked = track.charge_packed();
	if (ChargePacked==1)
		Charge=-1;
	else 
		Charge=1;
*/

  }
  
  TFTrack::TFTrack(L1MuGMTExtendedCand track):Track()
  	{

		
		Eta = track.etaValue();
		Phi = track.phiValue();
		Pt = track.ptValue();
		
		
		isEndcap1 = true;
		if(EtaPacked<0){isEndcap1 = false;}
		
	     
                EtaPacked = -1;
      	        PhiPacked = -1;
      	        ChargePacked = -1;
      	        Bx = -1;
      	        Charge = 0;
      	        Halo = -1;
      	        Mode = -1;
      	        Rank = -1;
      	        FR = -1;
      	        LUTAddress = -1;
		Quality =track.quality();
  
  	}

/*  
double TFTrack::distanceTo(RefTrack reftrack)
  {
	double newR;
	newR = (getPt()-reftrack->getPt())*
		(getEta()-reftrack->getEta())*
		(getPhi()-reftrack->getPhi());
	return newR;
  }
*/

  void TFTrack::print()
  {
    std::cout << "TFTrack Info" << std::endl;
    std::cout << "  Pt: "<< getPt()<< std::endl;
    std::cout << "  Phi: "<< getPhi()<< std::endl;
    std::cout << "  Eta: "<< getEta()<< std::endl;
    std::cout << "  Mode: "<< getMode()<< std::endl;
    std::cout << "  EtaPacked: "<< getEtaPacked()<< std::endl;
    std::cout << "  PhiPacked: "<< getPhiPacked()<< std::endl;
    std::cout << "  PtPacked: "<< getPtPacked()<< std::endl;
    std::cout << "  Rank: "<< getRank()<< std::endl;
    std::cout << "  FR: "<< getFR()<< std::endl;
    std::cout << "  LUTAddress: "<< getLUTAddress()<< std::endl;
  }
}
