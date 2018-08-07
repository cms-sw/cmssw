#include <memory>

#include "EventFilter/EcalDigiToRaw/interface/TCCBlockFormatter.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "EventFilter/EcalRawToDigi/interface/DCCRawDataDefinitions.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"


using namespace std;


TCCBlockFormatter::TCCBlockFormatter(Config const& iC, Params const& iP): BlockFormatter(iC, iP) {

 AllTPsamples_ = false;
}

void TCCBlockFormatter::DigiToRaw(const EcalTriggerPrimitiveDigi& trigprim, 
				  FEDRawData& rawdata, const EcalElectronicsMapping* TheMapping)
{

  if (debug_) cout << "enter in TCCBlockFormatter::DigiToRaw " << endl;

  int HEADER_SIZE = 8 * 9;
  int bx = bx_;
  int lv1 = lv1_;


	const EcalTrigTowerDetId& detid = trigprim.id();

           if ( (detid.subDet() == EcalBarrel) && (! doBarrel_) ) return;
           if ( (detid.subDet() == EcalEndcap) && (! doEndCap_) ) return;

	int iDCC = TheMapping -> DCCid(detid);
	int TCCid = TheMapping -> TCCid(detid);


	if (TCCid < EcalElectronicsMapping::MIN_TCCID || TCCid > EcalElectronicsMapping::MAX_TCCID) 
		cout << "Wrong TCCid in TCCBlockFormatter::DigiToRaw " << endl;
	bool IsEndCap = ( (EcalElectronicsId::MIN_DCCID_EEM <= iDCC && iDCC <= EcalElectronicsId::MAX_DCCID_EEM) ||
			  (EcalElectronicsId::MIN_DCCID_EEP <= iDCC && iDCC <= EcalElectronicsId::MAX_DCCID_EEP) );

	int FEDid = FEDNumbering::MINECALFEDID + iDCC;

	// note: row is a 64 bit word
	int NTT_max = 68;	// Barrel case
	int Nrows_TCC = 17;  	// Barrel case    (without the header row)
	int NTCC = 1;        	// Barrel case; number of TCC blocks
	int itcc_block = 1;     // Barrel case

        if (IsEndCap) {
         	Nrows_TCC = 8;         	// one row is a 64 bit word
         	NTCC = 4;		// 4 TTC in EndCap case. Use some custom numbering since
		int pair = TCCid % 2;	// the TCCid is written to the RawData.
                int inner = ( detid.ietaAbs() >= 22) ? 1 : 0;
                itcc_block = 2 * pair + inner + 1;
                if (inner == 1) NTT_max = 28;
                else NTT_max = 16;
        }

	
        int nsamples = trigprim.size();
	if (! AllTPsamples_) nsamples = 1;

	int iTT = TheMapping -> iTT(detid);   // number of tp inside a fed
	if (debug_) cout << "This is a TrigTower  iDCC iTT iTCCBlock TCCid " << dec << 
		iDCC << " " << iTT << " " << itcc_block << " " << TCCid << endl;
        if (debug_) cout << "ieta iphi " << dec << detid.ieta() << " " << detid.iphi() << endl;
        if (iTT <= 0 || iTT > NTT_max)  {
		cout << "invalid iTT " << iTT << endl;
		return;
	}

	int FE_index;
	
	// rawdata points to the block which will be built for TCC data  
 	if ((int)rawdata.size() != HEADER_SIZE) {
	  FE_index = rawdata.size() / 8 - NTCC*(Nrows_TCC+1);         // as far as raw data have been generated 
	  FE_index ++;                                                // infer position in TCC block
	  if (debug_) cout << "TCCid already there. FE_index = " << FE_index << endl;
  	}
	else {
 		if (debug_) cout << "New TTCid added on Raw data, TTCid = " << dec << TCCid << " 0x" << hex << TCCid << endl;
		FE_index = rawdata.size() / 8;                    // size in unites of 64 bits word
	        int fe_index = FE_index;
		for (int iblock=0; iblock < NTCC; iblock++) {     // do this once per fed in EB, four times in EE
		   rawdata.resize (rawdata.size() + 8);
		   unsigned char* ppData = rawdata.data();        // use this to navigate and create the binary
		   ppData[8*fe_index] = TCCid & 0xFF;             // fed_index increases in units of bytes
		   ppData[8*fe_index+2] = bx & 0xFF;              // bx takes bits 0-11: 0-7+8-11
		   ppData[8*fe_index+3] = (bx & 0xF00)>>8;
		   ppData[8*fe_index+3] |= 0x60;
		   ppData[8*fe_index+4] = lv1 & 0xFF;             // same game done for lv1, which takes bits 0-11: 0-7+8-11 
	 	   ppData[8*fe_index+5] = (lv1 & 0xF00)>>8;       // lv1
		   ppData[8*fe_index+6] = NTT_max;
		   ppData[8*fe_index+6] |= ((nsamples & 0x1)<<7); // nsamples: number time samples
		   ppData[8*fe_index+7] = ((nsamples & 0xE)>>1);
		   ppData[8*fe_index+7] |= 0x60;
		   if (iblock == 0) FE_index ++;
		   fe_index += Nrows_TCC+1;
		   rawdata.resize (rawdata.size() + 8*Nrows_TCC);    // 17 lines of TPG data in EB, 8 in EE
		}
		if (debug_) cout << "Added headers and empty lines : " << endl;
		if (debug_) print(rawdata);

		// -- put the B011 already, since for Endcap there can be empty
		// -- lines in the TCC and the SRP blocks
		unsigned char* ppData = rawdata.data();
		for (int iline=FE_index-1; iline < FE_index + (Nrows_TCC+1)*NTCC -1 ; iline++) {
                 ppData[8*iline + 7] |= 0x60;
		 ppData[8*iline + 3] |= 0x60;
		}
	}

	unsigned char* pData = rawdata.data();

	// -- Now the TCC Block :

	int jTT = (iTT-1);                                // jTT is the TP number insided a block;
	int irow = jTT/4 + (itcc_block-1)*(Nrows_TCC+1);  // you fit 4 TP's per row; move forward if you're not in the first block;
	int ival = jTT % 4;                               // for each block you have to skip, move of (Nrows_TCC +1) - 1 is for the TCC header

	// RTC required TP's tp follow global phi also in EB+, thus swap them inside the single TCC 
	// here you could swap ival -> 3-ival to swap phi insied EB+ supermodules  
	if(NUMB_SM_EB_PLU_MIN <= iDCC && iDCC <= NUMB_SM_EB_PLU_MAX)
	  {ival = 3-ival;}

	FE_index += irow;                                 // ival is location inside a TP row; varies between 0-3

        if (debug_) cout << "Now add tower " << dec << iTT << " irow ival " << dec << irow << " " << dec << ival << endl;
        if (debug_) cout << "new data will be added at line " << dec << FE_index << endl;

	int fg = trigprim.fineGrain();
	int et = trigprim.compressedEt();
	int ttflag = trigprim.ttFlag();

	if (debug_ && (ttflag != 0)) {
		cout << "in TCCBlock : this tower has a non zero flag" << endl;
		cout << "Fedid  iTT  flag " << dec << FEDid << " " << iTT << " " << "0x" << hex << ttflag << endl;
	}
	pData[8*FE_index + ival*2] = et & 0xFF;                   // ival is location inside a TP row; varies between 0-3; tp goes in bits 0-7
	pData[8*FE_index + ival*2+1] = (ttflag<<1) + (fg&0x1);    // fg follows in bit 8; ttfg is in bits 9-11 
	if (IsEndCap) {
		// re-write the TCCid  and N_Tower_Max :
		int ibase = 8*(FE_index - (int)(jTT/4) -1);
		pData[ibase] = TCCid & 0xFF;
                pData[ibase+6] = NTT_max;
                pData[ibase+6] |= ((nsamples & 0x1)<<7);
                pData[ibase+7] |= ((nsamples & 0xE)>>1);
	}
	if (debug_) cout << "pData[8*FE_index + ival*2+1] = " << hex << (int)pData[8*FE_index + ival*2+1] << endl;
	if (debug_) cout << "ttflag ttflag<<1 " << hex << ttflag << " " << hex << (ttflag<<1) << endl;
	if (debug_) cout << "fg&0x1 " << hex << (fg&0x1) << endl;
	if (debug_) cout << "sum " << hex << ( (ttflag<<1) + (fg&0x1) ) << endl;
	if (ival %2 == 1) pData[8*FE_index + ival*2+1] |= 0x60;
	if (debug_) cout << "ttflag et fgbit " << hex << ttflag << " " << hex << et << " " << hex << fg << endl;
	if (debug_) print(rawdata);

	
}


