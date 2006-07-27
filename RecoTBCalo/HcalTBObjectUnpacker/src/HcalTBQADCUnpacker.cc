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
static const int bMuonV1           = 1;  // Muon veto wall
static const int bMuonV2           = 2;  // Muon veto wall
static const int bMuonV3           = 3;  // Muon veto wall
static const int bMuonV4           = 4;  // Muon veto wall
static const int bMuonV5           = 5;  // Muon veto wall
static const int bMuonV6           = 6;  // Muon veto wall
static const int bMuonV7           = 7;  // Muon veto wall
static const int bMuonV8           = 8;  // Muon veto wall
static const int bMuonVF           = 11;  // Behind the table
static const int bMuonVB           = 12;  // on HB1
static const int bScint1           = 13;  // 14x14 cm
static const int bScint2           = 14;  // 4x4 cm
static const int bScint3           = 15;  // 2x2 cm
static const int bScint4           = 16;  // 14x14 cm
static const int bTOF1             = 21;
static const int bTOF2             = 22;
static const int bCerenkov2        = 23;  // el id
static const int bCerenkov3        = 24;  // pi/proton separation
static const int bSCI_VLE          = 25;  // VLE line 

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


void HcalTBQADCUnpacker::setCalib(const vector<vector<string> >& calibLines_) {
	for(int i=0;i<128;i++)
	 {
	  qdc_ped[i]=0.;qdc_gain[i]=1.;
	 }
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
		  printf("HcalTBQADCUnpacker thinks your QADC calibration format stinks....\n");
      ///throw an exception here when we're ready with the calib file..  
		  }
		}
	 }
	}
void HcalTBQADCUnpacker::unpack(const FEDRawData& raw,
  			       HcalTBBeamCounters& beamadc, bool is04) const {

  if (raw.size()<3*8) {
    throw cms::Exception("Missing Data") << "No data in the QDC block";
  }

  const ClassicQADCDataFormat* qadc=(const ClassicQADCDataFormat*)raw.data();

  if(is04){ ///this is TB04
// to fix ped,gain,mask
    beamadc.setADCs04(qadc->data[aMuonV],qadc->data[aMuonV3],qadc->data[aMuonV6],
		    qadc->data[aMuonVH1],qadc->data[aMuonVH2],qadc->data[aMuonVH3],
		    qadc->data[aMuonVH4],qadc->data[aCerenkov2],qadc->data[aCerenkov3],
		    qadc->data[aSCI_VLE],qadc->data[aSCI_521],qadc->data[aSCI_528],
		    qadc->data[aScint1],qadc->data[aScint2],qadc->data[aScint3],qadc->data[aScint4]);
       }
  else{ /// this is TB06
    const CombinedTDCQDCDataFormat* qdctdc=(const CombinedTDCQDCDataFormat*)raw.data();
    double qdc_calib_hits[32];
	for (unsigned int i=0;i<qdctdc->n_qdc_hits;i++)
	  qdc_calib_hits[i]=((qdctdc->qdc_values[i]&0xFFF)-qdc_ped[i])*qdc_gain[i];

    beamadc.setADCs06( qdc_calib_hits[bMuonVF], qdc_calib_hits[bMuonVB],
                    qdc_calib_hits[bMuonV1],qdc_calib_hits[bMuonV2],qdc_calib_hits[bMuonV3], 
                    qdc_calib_hits[bMuonV4],qdc_calib_hits[bMuonV5],qdc_calib_hits[bMuonV6], 
                    qdc_calib_hits[bMuonV7], qdc_calib_hits[bMuonV8],
                    qdc_calib_hits[bCerenkov2], qdc_calib_hits[bCerenkov3], 
                    qdc_calib_hits[bSCI_VLE],
                    qdc_calib_hits[bScint1], qdc_calib_hits[bScint2], 
		    qdc_calib_hits[bScint3], qdc_calib_hits[bScint4],
                    qdc_calib_hits[bTOF1], qdc_calib_hits[bTOF2]);

       }
 }
  
}
