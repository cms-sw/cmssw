//////Takes in the values of track quantites and returns an
//////L1TReginalMuonCandidate with proper values and legths
//////
//////
//////
//////


#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"



int GetPackedEta(float theta, int sector){

	float scale = 1/0.010875;

	float theta_angle = (theta*0.2851562 + 8.5)*(3.14159265359/180);
	float eta = (-1)*log(tan(theta_angle/2));
	if(sector > 5)
		eta *= -1;

	int PackedEta = eta*scale;
	if(eta < 0)
		PackedEta -= 1;

	if(PackedEta > 239)
		PackedEta = 239;

	if(PackedEta < -240)
		PackedEta = -240;

	//if(PackedEta < 0)
	//	PackedEta = 512 + PackedEta;

	return PackedEta;

}

float GetGlobalEta(float theta, int sector){

	float theta_angle = (theta*0.2851562 + 8.5)*(3.14159265359/180);
	float eta = (-1)*log(tan(theta_angle/2));
	if(sector > 5)
		eta *= -1;
		
	return eta;

}

int GetPackedPhi(int phi){

	float phiDeg = (phi*0.0166666);
	phiDeg -= 2.0;


	int PackedPhi = phiDeg/0.625;

	//if(PackedPhi < 0)
	//	PackedPhi = 256 + PackedPhi;

	return PackedPhi;

}


l1t::RegionalMuonCand MakeRegionalCand(float pt, int phi, int theta,
											   int sign, int quality,
											   int trackaddress, int sector){

	l1t::RegionalMuonCand Cand;

	int iEta = GetPackedEta(theta,sector);
	int iPhi = GetPackedPhi(phi);

	l1t::tftype TFtype = l1t::tftype::emtf_pos;
	if(sector > 5){
		TFtype = l1t::tftype::emtf_neg;
		sector -= 6;
	}

	// compressed pt = pt*2 (scale) + 1 (iPt = 0 is empty candidate)
	int iPt = pt*2 + 1;
	if(iPt > 511)
		iPt = 511;

	if(iPt < 0)
		iPt = 0;

	int iQual = quality;
	
	int LSB = quality & 3;
	
	float eta = GetGlobalEta(theta,sector);
	
	if(eta < 1.2){
	
		switch(quality){
			case(15): iQual = 8;break;
			case(14): iQual = 4;break;
			case(13): iQual = 4;break;
			case(12): iQual = 4;break;
			case(11): iQual = 4;break;
			default: iQual = 4;break;
		}
	
	}
	else{
	
		switch(quality){
			case(15): iQual = 12;break;
			case(14): iQual = 12;break;
			case(13): iQual = 12;break;
			case(12): iQual = 8;break;
			case(11): iQual = 12;break;
			case(10): iQual = 8;break;
			case(7): iQual = 8;break;
			default: iQual = 4;break;
		}
	
	}
	iQual |= LSB;

	Cand.setHwPt(iPt);
	Cand.setHwEta(iEta);
  	Cand.setHwPhi(iPhi);
  	Cand.setHwSign(1);
	Cand.setHwSignValid(0);
  	Cand.setHwQual(iQual);
  	// jl: FIXME this has to be adapted to the new schema of saving track addresses
  	//Cand.setTrackSubAddress(l1t::RegionalMuonCand::kME12, trackaddress&0xf);
	//Cand.setTrackSubAddress(l1t::RegionalMuonCand::kME22, trackaddress>>4);
	Cand.setTFIdentifiers(sector,TFtype);


	return Cand;

}
