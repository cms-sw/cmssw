#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBQADCUnpacker.h"
#include "FWCore/Utilities/interface/Exception.h"

// QADC channels
static const int N_QADCS_ALLOWED = 6;

// Channel to logical unit
// TB04
static const int aScint1          = 52;  // 14x14 cm
static const int aScint2          = 53;  // 4x4 cm
static const int aScint3          = 54;  // 2x2 cm
static const int aScint4          = 66;  // 14x14 cm
static const int aMuonV            = 56;  // Behind the table
static const int aMuonV3           = 64;  // on HB1
static const int aMuonV6           = 62;  // on HB2
static const int aMuonVH1          = 67;  // Oct. muon veto wall
static const int aMuonVH2          = 68;  // Oct. muon veto wall
static const int aMuonVH3          = 69;  // Oct. muon veto wall
static const int aMuonVH4          = 70;  // Oct. muon veto wall
static const int aCerenkov2        = 49;  // el id
static const int aCerenkov3        = 59;  // pi/proton separation
static const int aSCI_VLE          = 65;  // VLE line 
static const int aSCI_521          = 57;
static const int aSCI_528          = 58;
// TB06
static const int bMuonV1           = 0;  // Muon veto wall
static const int bMuonV2           = 1;  // Muon veto wall
static const int bMuonV3           = 2;  // Muon veto wall
static const int bMuonV4           = 3;  // Muon veto wall
static const int bMuonV5           = 4;  // Muon veto wall
static const int bMuonV6           = 5;  // Muon veto wall
static const int bMuonV7           = 6;  // Muon veto wall
static const int bMuonV8           = 7;  // Muon veto wall
static const int bMuonVF           = 10;  // Behind the table
static const int bMuonVB           = 11;  // Behind a beam dump
static const int bScint1           = 12;  // 14x14 cm
static const int bScint2           = 13;  // 4x4 cm
static const int bScint3           = 14;  // 2x2 cm
static const int bScint4           = 15;  // 14x14 cm
static const int bCerenkov1        = 16;  // 
static const int bCerenkov2        = 17;  // el id
static const int bCerenkov3        = 18;  // pi/proton id
static const int bTOF1             = 20;  // TOF1
static const int bTOF2             = 21;  // TOF2
static const int bSCI_521          = 22;
static const int bSCI_528          = 23;
static const int bVH1              = 28;  // beam halo up
static const int bVH2              = 29;  // beam halo left from particle view
static const int bVH3              = 30;  // beam halo right from particle view
static const int bVH4              = 31;  // beam halo down

namespace hcaltb {

HcalTBQADCUnpacker::HcalTBQADCUnpacker(){}; 

struct ClassicQADCDataFormat {
  unsigned int cdfHeader0,cdfHeader1,cdfHeader2,cdfHeader3;
  unsigned short data[N_QADCS_ALLOWED*32];
  unsigned int cdfTrailer0,cdfTrailer1;

};

 struct CombinedTDCQDCDataFormat {
    unsigned int cdfHeader0,cdfHeader1,cdfHeader2,cdfHeader3;
    unsigned int n_qdc_hits; // Count of QDC channels
    unsigned int n_tdc_hits; // upper/lower TDC counts    
    unsigned short qdc_values[4];
  };

// Sets the pedestal and gain
void HcalTBQADCUnpacker::setCalib(const vector<vector<string> >& calibLines_) {
// The default pedestal and gain
	for(int i=0;i<N_QADCS_ALLOWED*32;i++)
	 {
	  qdc_ped[i]=0.;qdc_gain[i]=1.;
	 }
// Pedestal and gains from configuration file.
	for(unsigned int ii=0;ii<calibLines_.size();ii++)
	 {
	  if(calibLines_[ii][0]=="QDC")
		{
		if(calibLines_[ii].size()==4)
		  {
		  int channel=atoi(calibLines_[ii][1].c_str());
		  qdc_ped[channel]=atof(calibLines_[ii][2].c_str());
		  qdc_gain[channel]=atof(calibLines_[ii][3].c_str());
	//	  printf("Got QDC %i ped %f , gain %f\n",channel, qdc_ped[channel],qdc_gain[channel]);
		  }
		 else
		  {
                  throw cms::Exception("Incomplete configuration") << 
		  "Wrong QADC configuration format: expected 3 parameters, got "<<calibLines_[ii].size()-1;
		  }
		}
	 } // End of calibLines.
	}

void HcalTBQADCUnpacker::unpack(const FEDRawData& raw,
  			       HcalTBBeamCounters& beamadc, bool is04) const {

  if (raw.size()<3*8) {
    throw cms::Exception("Missing Data") << "No data in the QDC block";
  }


  if(is04){ ///this is TB04
    const ClassicQADCDataFormat* qadc=(const ClassicQADCDataFormat*)raw.data();
    double qdc_calib_hits[N_QADCS_ALLOWED*32];
    // Applying mask, pedestal subtraction and gain.
	for (unsigned int i=0;i<N_QADCS_ALLOWED*32;i++)
	  qdc_calib_hits[i]=((qadc->data[i]&0xFFF)-qdc_ped[i])/qdc_gain[i];

    // Ecal energy sum should go here.
	double Ecal7x7=0.;
	for(int i=0;i<49;i++)Ecal7x7+=qdc_calib_hits[i];


    beamadc.setADCs04(qdc_calib_hits[aMuonV],qdc_calib_hits[aMuonV3],qdc_calib_hits[aMuonV6],
		    qdc_calib_hits[aMuonVH1],qdc_calib_hits[aMuonVH2],qdc_calib_hits[aMuonVH3],
		    qdc_calib_hits[aMuonVH4],qdc_calib_hits[aCerenkov2],qdc_calib_hits[aCerenkov3],
		    qdc_calib_hits[aSCI_VLE],qdc_calib_hits[aSCI_521],qdc_calib_hits[aSCI_528],
		    qdc_calib_hits[aScint1],qdc_calib_hits[aScint2],qdc_calib_hits[aScint3],
		    qdc_calib_hits[aScint4],Ecal7x7);
       }
  else{ /// this is TB06
    const CombinedTDCQDCDataFormat* qdctdc=(const CombinedTDCQDCDataFormat*)raw.data();
    double qdc_calib_hits[32];
	for (unsigned int i=0;i<qdctdc->n_qdc_hits;i++)
	  qdc_calib_hits[i]=((qdctdc->qdc_values[i]&0xFFF)-qdc_ped[i])/qdc_gain[i];

    beamadc.setADCs06( qdc_calib_hits[bMuonVF], qdc_calib_hits[bMuonVB],
                    qdc_calib_hits[bMuonV1],qdc_calib_hits[bMuonV2],qdc_calib_hits[bMuonV3], 
                    qdc_calib_hits[bMuonV4],qdc_calib_hits[bMuonV5],qdc_calib_hits[bMuonV6], 
                    qdc_calib_hits[bMuonV7], qdc_calib_hits[bMuonV8],
                    qdc_calib_hits[bCerenkov1], qdc_calib_hits[bCerenkov2], qdc_calib_hits[bCerenkov3],
                    qdc_calib_hits[bScint1], qdc_calib_hits[bScint2], 
		    qdc_calib_hits[bScint3], qdc_calib_hits[bScint4],
                    qdc_calib_hits[bTOF1], qdc_calib_hits[bTOF2],
		    qdc_calib_hits[bSCI_521],qdc_calib_hits[bSCI_528],
		    qdc_calib_hits[bVH1],qdc_calib_hits[bVH2],
		    qdc_calib_hits[bVH3],qdc_calib_hits[bVH4]);

       }
 }
  
}
