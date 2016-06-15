//////Takes in the values of track quantites and returns an
//////L1TReginalMuonCandidate with proper values and legths
//////
//////
//////
//////


#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

int GMT_eta_from_theta[128] = {239, 235, 233, 230, 227, 224, 222, 219, 217, 214, 212, 210, 207, 205, 203, 201,
			       199, 197, 195, 193, 191, 189, 187, 186, 184, 182, 180, 179, 177, 176, 174, 172,
			       171, 169, 168, 166, 165, 164, 162, 161, 160, 158, 157, 156, 154, 153, 152, 151,
			       149, 148, 147, 146, 145, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133,
			       132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117,
			       116, 116, 115, 114, 113, 112, 111, 110, 110, 109, 108, 107, 106, 106, 105, 104,
			       103, 102, 102, 101, 100,  99,  99,  98,  97,  96,  96,  95,  94,  93,  93,  92,
			       91,  91,  90,  89,  89,  88,  87,  87,  86,  85,  84,  84,  83,  83,  82,  81};

int GetPackedEta (int theta, int sector) {

  if (theta < 0) return 0;
  if (sector > 5) {
    if (theta > 127) return -240;
    else return -1*GMT_eta_from_theta[theta] - 1;
  }
  else {
    if (theta > 127) return  239;
    else return   GMT_eta_from_theta[theta];
  }
}
 
float GetGlobalEta(int theta, int sector){ // Should use calc_eta to compute - AWB 07.05.16

  // Underlying calculation:
  // float theta_angle = (theta*0.2851562 + 8.5)*(pi/180);
  // float eta = (-1)*log(tan(theta_angle/2));

  return 0.010875 * GetPackedEta(theta, sector);
}

int GetPackedPhi(int phi){

  // Experimentally, this factor must be between 0.979752 and 0.979758 - AWB 11.05.16 
  float phiDeg = (phi / (0.625 * 60)) * 0.979755;
  int PackedPhi = floor( phiDeg - 35 );
  
  return PackedPhi;

}


l1t::RegionalMuonCand MakeRegionalCand(float pt, int phi, int theta,
				       int sign, int mode,
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

	int iQual = -999;
	
	int LSB = mode & 3;
	
	float eta = GetGlobalEta(theta,sector);
	
	if(eta < 1.2){
	
		switch(mode){
			case(15): iQual = 8;break;
			case(14): iQual = 4;break;
			case(13): iQual = 4;break;
			case(12): iQual = 4;break;
			case(11): iQual = 4;break;
			default: iQual = 4;break;
		}
	
	}
	else{
	
		switch(mode){
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
  	Cand.setHwSign(sign);
	Cand.setHwSignValid(1);
  	Cand.setHwQual(iQual);
  	// jl: FIXME this has to be adapted to the new schema of saving track addresses
  	//Cand.setTrackSubAddress(l1t::RegionalMuonCand::kME12, trackaddress&0xf);
	//Cand.setTrackSubAddress(l1t::RegionalMuonCand::kME22, trackaddress>>4);
	Cand.setTFIdentifiers(sector,TFtype);


	return Cand;

}
