#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBQADCUnpacker.h"
#include "FWCore/Utilities/interface/Exception.h"

// QADC channels
static const int N_QADCS_ALLOWED = 6;
// Trigger Channels
static const int aScint1          = 52;  // 14x14 cm
static const int aScint2          = 53;  // 4x4 cm
static const int aScint3          = 54;  // 2x2 cm
static const int aScint4          = 66;  // 14x14 cm
// Beam Channels
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


void HcalTBQADCUnpacker::unpack(const FEDRawData& raw,
  			       HcalTBBeamCounters& beamadc, bool is04) const {

  if (raw.size()<3*8) {
    throw cms::Exception("Missing Data") << "No data in the QDC block";
  }

  const ClassicQADCDataFormat* qadc=(const ClassicQADCDataFormat*)raw.data();
  
  uint16_t dat0 = qadc->data[0] +qadc->data[1];

  if(is04){ ///this is TB04
    beamadc.setADCs(qadc->data[aMuonV],qadc->data[aMuonV3],qadc->data[aMuonV6],
		    qadc->data[aMuonVH1],qadc->data[aMuonVH2],qadc->data[aMuonVH3],
		    qadc->data[aMuonVH4],qadc->data[aCerenkov2],qadc->data[aCerenkov3],
		    qadc->data[aSCI_VLE],qadc->data[aSCI_521],qadc->data[aSCI_528],
		    qadc->data[aScint1],qadc->data[aScint2],qadc->data[aScint3],qadc->data[aScint4]);
  }
  else{
    const CombinedTDCQDCDataFormat* qdctdc=(const CombinedTDCQDCDataFormat*)raw.data();
    beamadc.setADCs(qdctdc->qdc_values[aMuonV-49], qdctdc->qdc_values[aMuonV3-49],
		    qdctdc->qdc_values[aMuonV6-49], qdctdc->qdc_values[aMuonVH1-49], 
		    qdctdc->qdc_values[aMuonVH2-49], qdctdc->qdc_values[aMuonVH3-49],
		    qdctdc->qdc_values[aMuonVH4-49], qdctdc->qdc_values[aCerenkov2-49],
		    qdctdc->qdc_values[aCerenkov3-49], qdctdc->qdc_values[aSCI_VLE-49],
		    qdctdc->qdc_values[aSCI_521-49],qdctdc->qdc_values[aSCI_528-49],
		    qdctdc->qdc_values[aScint1-49],qdctdc->qdc_values[aScint2-49],
		    qdctdc->qdc_values[aScint3-49],qdctdc->qdc_values[aScint4-49]);
  }

 }
  
}
