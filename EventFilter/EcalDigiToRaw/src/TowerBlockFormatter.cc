#include <memory>
#include <vector>
#include <iostream>
#include <list>

#include <FWCore/Framework/interface/Handle.h>

#include "EventFilter/EcalDigiToRaw/interface/TowerBlockFormatter.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

using namespace std;


TowerBlockFormatter::TowerBlockFormatter() {

}

TowerBlockFormatter::~TowerBlockFormatter() {

}



void TowerBlockFormatter::DigiToRaw(const EBDataFrame& dataframe, FEDRawData& rawdata,
					 const EcalElectronicsMapping* TheMapping)

{

 int bx = *pbx_;
 int lv1 = *plv1_;


  int rdsize = rawdata.size() / 8;  // size in Word64

  bool newFE = false;

        const EBDetId& ebdetid = dataframe.id();

	int DCCid = TheMapping -> DCCid(ebdetid);
	int FEDid = EcalFEDIds.first + DCCid ;


        int nsamples = dataframe.size();
                // -- FE number
    	const EcalElectronicsId& elid = TheMapping -> getElectronicsId(ebdetid);
	int iFE = elid.towerId();
        if (iFE <= 0 || iFE > 68)  throw cms::Exception("InvalidFEid") << 
		"TowerBlockFormatter::DigiToRaw : Invalid iFE " << iFE << endl;


        map<int, map<int,int> >::iterator fen = FEDorder -> find(FEDid);
        map<int, map<int,int> >::iterator fed = FEDmap -> find(FEDid);

        if (fen == FEDorder -> end()) {
                if (debug_) cout << "New FED in TowerBlockFormatter " << dec << FEDid << " 0x" << hex << FEDid << endl;
		map<int,int> FEorder;
                pair<map<int, map<int,int> >::iterator, bool> t1 = FEDorder -> insert(map<int, map<int,int> >::value_type(FEDid,FEorder));
		map<int,int> FEmap;
		pair<map<int, map<int,int> >::iterator, bool> t2 = FEDmap -> insert(map<int, map<int,int> >::value_type(FEDid,FEmap));
                fen = t1.first;
		fed = t2.first;
        }

	map<int, int>& FEorder = (*fen).second;
        map<int, int>& FEmap = (*fed).second;

        map<int,int>::iterator fe = FEorder.find(iFE);
	int FE_order;
	int FE_index;
	if (fe != FEorder.end()) {
		FE_order = (*fe).second;
		map<int,int>::iterator ff = FEmap.find(FE_order);
		if (ff == FEmap.end()) cout << "Error with maps... " << endl;
		FE_index = (*ff).second; 
                if (debug_) cout << "FE already there, FE_index = " << dec << FE_index <<  " FEorder " << FE_order << endl;
	}
	else {
                if (debug_) cout << "New FE in TowerBlockFormatter  FE " << dec << iFE << " 0x" << hex << iFE << " in FED id " << dec << FEDid << endl;
		newFE = true;
 		int inser = rdsize;
		int number_FEs = FEorder.size() -1;
                FE_order = number_FEs+1;
		pair<map<int,int>::iterator, bool> t2 = FEorder.insert(map<int,int>::value_type(iFE,FE_order)); 
		if (! t2.second) cout << " FE insertion failed...";
                pair<map<int,int>::iterator, bool> tt = FEmap.insert(map<int,int>::value_type(FE_order,inser));
                fe = tt.first;
		FE_index = (*fe).second;
		if (debug_) cout << "Build the Tower Block header for FE id " << iFE << " start at line " << rdsize << endl;
		if (debug_) cout << "This is the Fe number (order) " << number_FEs+1 << endl;
                rawdata.resize( 8*rdsize + 8);
		unsigned char* pData = rawdata.data();
		pData[8*FE_index] = iFE & 0xFF;
		pData[8*FE_index+1] = (nsamples & 0x7F);
		pData[8*FE_index+2] = bx & 0xFFF;
		pData[8*FE_index+3] = lv1 & 0xFFF;
		pData[8*FE_index+3] |= 0xa0;
		pData[8*FE_index+6] = 1;
		pData[8*FE_index+7] = 0xc0;
		if (debug_) print(rawdata);

        }


                // -- Crystal number inside the SM :

	int istrip = elid.stripId();
	int ichannel = elid.xtalId();

	if (debug_) cout << "Now add crystal : strip  channel " << dec << istrip << " " << ichannel << endl;


        unsigned char* pData = rawdata.data();

        vector<unsigned char> vv(&pData[0],&pData[rawdata.size()]);


        int n_add = 2 + 2*nsamples;     // 2 bytes per sample, plus 2 bytes before sample 0
        if (n_add % 8 != 0) n_add = n_add/8 +1;
	else
	n_add = n_add/8;
	if (debug_) cout << "will add " << n_add << " lines of 64 bits at line " << 8*(FE_index+1) << endl;
        rawdata.resize( rawdata.size() + 8*n_add );
	unsigned char* ppData = rawdata.data();

        vector<unsigned char>::iterator iter = vv.begin() + 8*(FE_index+1);

	vector<unsigned char> toadd(n_add*8);


        int tzs=0;
	toadd[0] = (istrip & 0x7) + ((ichannel & 0x7)<<4);
	toadd[1] = (tzs & 0x1) <<12;

	for (int isample=0; isample < (n_add*8-2)/2; isample++) {
		if (isample < nsamples) {
                  uint16_t word = (dataframe.sample(isample)).raw();   // 16 bits word corresponding to this sample
		  toadd[2 + isample*2] = word & 0x00FF;
		  toadd[2 + isample*2 +1] = (word & 0xFF00)>>8; 
		}
		else {
		  toadd[2 + isample*2] = 0;
		  toadd[2 + isample*2 +1] = 0;
		}
		if (isample % 2 == 0) toadd[2 + isample*2 +1] |= 0xc0;  // need to add the B11 header...
        }

	vv.insert(iter,toadd.begin(),toadd.end());


	// update the pData for this FED :
	for (int i=0; i < (int)vv.size(); i++) {
		ppData[i] = vv[i];
	}

	if (debug_) {
	  cout << "pData for this FED is now " << endl;
	  print(rawdata);
	}

	// and update the FEmap for this FED :
	for (int i=FE_order+1; i < (int)FEorder.size(); i++) {
		FEmap[i] += n_add;
		if (debug_) cout << "FEmap updated for fe number " << dec << i << endl;
		if (debug_) cout << " FEmap[" << i << "] = " << FEmap[i] << endl;
	}

	// update the block length
	int blocklength = ppData[8*FE_index+6]
		        + ((ppData[8*FE_index+7] & 0x1)<<8);
	blocklength += n_add;
	ppData[8*FE_index+6] = blocklength & 0xFF;
	ppData[8*FE_index+7] |= (blocklength & 0x100)>>8;


}



void TowerBlockFormatter::EndEvent() {

 FEDmap -> empty();
 FEDorder -> empty();
 delete FEDmap;
 delete FEDorder;
 FEDmap = 0;
 FEDorder = 0;
}

void TowerBlockFormatter::StartEvent() {

 FEDmap = new map<int, map<int,int> >;
 FEDorder = new map<int, map<int,int> >;
 
}




void TowerBlockFormatter::DigiToRaw(const EEDataFrame& dataframe, FEDRawData& rawdata, const EcalElectronicsMapping* TheMapping)

	// -- now that we have the EcalElectronicsMapping, this method could probably be
	//    merged with DigiToRaw(EBdataframe).
	//    Keep as it is for the while...
{

 debug_ = false;

 int bx = *pbx_;
 int lv1 = *plv1_;


  int rdsize = rawdata.size() / 8;  // size in Word64

  bool newFE = false;


        const EEDetId& eedetid = dataframe.id();
	EcalElectronicsId elid = TheMapping -> getElectronicsId(eedetid);
        int DCCid = elid.dccId();
	int FEDid = EcalFEDIds.first + DCCid ;
	int iFE = elid.towerId();

	if (debug_) cout << "enter in TowerBlockFormatter::DigiToRaw DCCid FEDid iFE " <<
	dec << DCCid << " " << FEDid << " " << iFE << endl;

        int nsamples = dataframe.size();

        if (iFE <= 0 || iFE > 68)  {
		cout << "invalid iFE for EndCap DCCid iFE " << DCCid << " " << iFE << endl;
		return;
	}


        map<int, map<int,int> >::iterator fen = FEDorder -> find(FEDid);
        map<int, map<int,int> >::iterator fed = FEDmap -> find(FEDid);

        if (fen == FEDorder -> end()) {
                if (debug_) cout << "New FED in TowerBlockFormatter " << dec << FEDid << " 0x" << hex << FEDid << endl;
                map<int,int> FEorder;
                pair<map<int, map<int,int> >::iterator, bool> t1 = FEDorder -> insert(map<int, map<int,int> >::value_type(FEDid,FEorder));
                map<int,int> FEmap;
                pair<map<int, map<int,int> >::iterator, bool> t2 = FEDmap -> insert(map<int, map<int,int> >::value_type(FEDid,FEmap));
                fen = t1.first;
                fed = t2.first;
        }

        map<int, int>& FEorder = (*fen).second;
        map<int, int>& FEmap = (*fed).second;

        map<int,int>::iterator fe = FEorder.find(iFE);
        int FE_order;
        int FE_index;
        if (fe != FEorder.end()) {
                FE_order = (*fe).second;
                map<int,int>::iterator ff = FEmap.find(FE_order);
                if (ff == FEmap.end()) cout << "Error with maps... " << endl;
                FE_index = (*ff).second;
                if (debug_) cout << "FE already there, FE_index = " << dec << FE_index <<  " FEorder " << FE_order << endl;
        }
        else {
                if (debug_) cout << "New FE in TowerBlockFormatter  FE " << dec << iFE << " 0x" << hex << iFE << " in FED id " << dec << FEDid << endl;
                newFE = true;
                int inser = rdsize;
                int number_FEs = FEorder.size() -1;
                FE_order = number_FEs+1;
                pair<map<int,int>::iterator, bool> t2 = FEorder.insert(map<int,int>::value_type(iFE,FE_order));
                if (! t2.second) cout << " FE insertion failed...";
                pair<map<int,int>::iterator, bool> tt = FEmap.insert(map<int,int>::value_type(FE_order,inser));
                fe = tt.first;
                FE_index = (*fe).second;
                if (debug_) cout << "Build the Tower Block header for FE id " << iFE << " start at line " << rdsize << endl;
                if (debug_) cout << "This is the Fe number (order) " << number_FEs+1 << endl;
                rawdata.resize( 8*rdsize + 8);
                unsigned char* pData = rawdata.data();

                pData[8*FE_index] = iFE & 0xFF;
                pData[8*FE_index+1] = (nsamples & 0x7F);
                pData[8*FE_index+2] = bx & 0xFFF;
                pData[8*FE_index+3] = lv1 & 0xFFF;
                pData[8*FE_index+3] |= 0xa0;
                pData[8*FE_index+6] = 1;
                pData[8*FE_index+7] = 0xc0;
                if (debug_) print(rawdata);

        }



                // -- Crystal number inside the SM :
	int istrip = elid.stripId();
	int ichannel = elid.xtalId();

        if (debug_) cout << "Now add crystal  strip  channel " << dec << istrip << " " << ichannel << endl;

        unsigned char* pData = rawdata.data();

        vector<unsigned char> vv(&pData[0],&pData[rawdata.size()]);


        int n_add = 2 + 2*nsamples;     // 2 bytes per sample, plus 2 bytes before sample 0
        if (n_add % 8 != 0) n_add = n_add/8 +1;
        else
        n_add = n_add/8;
	if (debug_) cout << "nsamples = " << dec << nsamples << endl;
        if (debug_) cout << "will add " << n_add << " lines of 64 bits at line " << 8*(FE_index+1) << endl;
        rawdata.resize( rawdata.size() + 8*n_add );
        unsigned char* ppData = rawdata.data();

        vector<unsigned char>::iterator iter = vv.begin() + 8*(FE_index+1);

        vector<unsigned char> toadd(n_add*8);


        int tzs=0;
        toadd[0] = (istrip & 0x7) + ((ichannel & 0x7)<<4);
        toadd[1] = (tzs & 0x1) <<12;

        for (int isample=0; isample < (n_add*8-2)/2; isample++) {
                if (isample < nsamples) {
                  uint16_t word = (dataframe.sample(isample)).raw();   // 16 bits word corresponding to this sample
                  toadd[2 + isample*2] = word & 0x00FF;
                  toadd[2 + isample*2 +1] = (word & 0xFF00)>>8;
                }
                else {
                  toadd[2 + isample*2] = 0;
                  toadd[2 + isample*2 +1] = 0;
                }
                if (isample % 2 == 0) toadd[2 + isample*2 +1] |= 0xc0;  // need to add the B11 header...
        }

        vv.insert(iter,toadd.begin(),toadd.end());


        // update the pData for this FED :
        for (int i=0; i < (int)vv.size(); i++) {
                ppData[i] = vv[i];
        }

        if (debug_) {
          cout << "pData for this FED is now " << endl;
          print(rawdata);
        }

        // and update the FEmap for this FED :
        for (int i=FE_order+1; i < (int)FEorder.size(); i++) {
                FEmap[i] += n_add;
                if (debug_) cout << "FEmap updated for fe number " << dec << i << endl;
                if (debug_) cout << " FEmap[" << i << "] = " << FEmap[i] << endl;
        }

        // update the block length
        int blocklength = ppData[8*FE_index+6]
                        + ((ppData[8*FE_index+7] & 0x1)<<8);
        blocklength += n_add;
        ppData[8*FE_index+6] = blocklength & 0xFF;
        ppData[8*FE_index+7] |= (blocklength & 0x100)>>8;


}










