#include <memory>
#include <list>

#include "EventFilter/EcalDigiToRaw/interface/TowerBlockFormatter.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

using namespace std;


TowerBlockFormatter::TowerBlockFormatter(Config const& iC, Params const& iP): BlockFormatter(iC, iP) {

}


void TowerBlockFormatter::DigiToRaw(const EBDataFrame& dataframe, FEDRawData& rawdata,
					 const EcalElectronicsMapping* TheMapping)

{

 int bx = bx_;
 int lv1 = lv1_ - 1;


  int rdsize = rawdata.size() / 8;  // size in Word64

        const EBDetId& ebdetid = dataframe.id();

	int DCCid = TheMapping -> DCCid(ebdetid);
	int FEDid = FEDNumbering::MINECALFEDID + DCCid ;


        int nsamples = dataframe.size();
                // -- FE number
    	const EcalElectronicsId& elid = TheMapping -> getElectronicsId(ebdetid);
	int iFE = elid.towerId();
        if (iFE <= 0 || iFE > 68)  throw cms::Exception("InvalidFEid") << 
		"TowerBlockFormatter::DigiToRaw : Invalid iFE " << iFE << endl;


        map<int, map<int,int> >::iterator fen = FEDorder.find(FEDid);
        map<int, map<int,int> >::iterator fed = FEDmap.find(FEDid);

        if (fen == FEDorder.end()) {
                if (debug_) cout << "New FED in TowerBlockFormatter " << dec << FEDid << " 0x" << hex << FEDid << endl;
		map<int,int> FEorder;
                pair<map<int, map<int,int> >::iterator, bool> t1 = FEDorder.insert(map<int, map<int,int> >::value_type(FEDid,FEorder));
		map<int,int> FEmap;
		pair<map<int, map<int,int> >::iterator, bool> t2 = FEDmap.insert(map<int, map<int,int> >::value_type(FEDid,FEmap));
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
		pData[8*FE_index+2] = bx & 0xFF;
		pData[8*FE_index+3] = (bx >>8) & 0x0F;
                pData[8*FE_index+3] |= 0xa0;
		pData[8*FE_index+4] = lv1 & 0xFF;
		pData[8*FE_index+5] = (lv1 >>8) & 0x0F;
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
	if (debug_) cout << "will add " << n_add << " lines of 64 bits at line " << (FE_index+1) << endl;
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



void TowerBlockFormatter::EndEvent(FEDRawDataCollection* productRawData) {

// -- Need to reorder the FE's in teh TowerBlock. They come in the right
//    order when reading the unsuppressed digis, but ganz durcheinander 
//    when reading the SelectiveReadout_Suppressed digis.

  if (debug_) cout << "enter in TowerBlockFormatter::EndEvent. First reorder the FE's. " << endl;

  for (int idcc=1; idcc <= 54; idcc++) {

	// debug_ = (idcc == 34);

	//if (idcc != 34) continue;

	int FEDid = FEDNumbering::MINECALFEDID + idcc;
	// cout << "Process FED " << FEDid << endl;
	FEDRawData& fedData = productRawData -> FEDData(FEDid);
	if (fedData.size() <= 16) continue;

	if (debug_) cout << "This is FEDid = " << FEDid << endl;

  	unsigned char * pData = fedData.data();
  	// Word64* words = reinterpret_cast<Word64*>(const_cast<unsigned char*>(pData));
	Word64* words = reinterpret_cast<Word64*>(pData);

	int length = fedData.size() / 8;
        int iDAQ_header(-1), iDCC_header(-1), iTCCBlock_header(-1),
                iSRBlock_header(-1), iTowerBlock_header(-1), iDAQ_trailer(-1);

        for (int i=length-1; i > -1; i--) {
        if (  ( (words[i] >> 60) & 0xF) == 0x5 ) iDAQ_header=i;
        if (  ( (words[i] >> 62) & 0x3) == 0x0 ) iDCC_header=i;
        if (  ( (words[i] >> 61) & 0x7) == 0x3 ) iTCCBlock_header=i;
        if (  ( (words[i] >> 61) & 0x7) == 0x4 ) iSRBlock_header=i;
        if (  ( (words[i] >> 62) & 0x3) == 0x3 ) iTowerBlock_header=i;
        if (  ( (words[i] >> 60) & 0xF) == 0xA ) iDAQ_trailer=i;
        }

        if (iTowerBlock_header < 0) iTowerBlock_header = iDAQ_trailer;
        if (iSRBlock_header < 0) iSRBlock_header = iTowerBlock_header;
        if (iTCCBlock_header < 0) iTCCBlock_header = iSRBlock_header;

	if (debug_) {
        cout << "iDAQ_header = " << iDAQ_header << endl;
        cout << " iDCC_header = " << iDCC_header << endl;
        cout << " iTCCBlock_header = " << iTCCBlock_header << endl;
        cout << " iSRBlock_header = " << iSRBlock_header << endl;
        cout << " iTowerBlock_header = " << iTowerBlock_header << endl;
        cout << " iDAQ_trailer = " << iDAQ_trailer << endl;
	}

	std::map<int, int> FrontEnd;
	std::map<int, std::vector<Word64> > Map_xtal_data;

	int iTowerBlock_header_keep = iTowerBlock_header;
	
	while (iTowerBlock_header < iDAQ_trailer) {
        	int fe = words[iTowerBlock_header] & 0xFF;
		int nlines = (words[iTowerBlock_header] >> 48) & 0x1FF;
		if (debug_) cout << "This is FE number " << fe << "needs nlines = " << nlines << endl;
		FrontEnd[fe] = nlines;
		std::vector<Word64> xtal_data;
		for (int j=0; j < nlines; j++) {
			Word64 ww = words[iTowerBlock_header+j];
			xtal_data.push_back(ww);
		}
		Map_xtal_data[fe] = xtal_data;
		iTowerBlock_header += nlines;
	}

	if (debug_) {
		cout << "vector of FrontEnd : " << FrontEnd.size() << endl;
		for (std::map<int, int>::const_iterator it=FrontEnd.begin();
                        it != FrontEnd.end(); it++) {
		    int fe = it -> first;
			int l = it -> second;
			cout << "FE line " << fe << " " << l << endl;
		}
	}

	iTowerBlock_header = iTowerBlock_header_keep;
	for (std::map<int, int>::const_iterator it=FrontEnd.begin();
			it != FrontEnd.end(); it++) {
		int fe = it -> first;
		int nlines = it -> second;
		if (debug_) cout << "iTowerBlock_header = " << iTowerBlock_header << endl;
		vector<Word64> xtal_data = Map_xtal_data[fe];
		for (int j=0; j < nlines; j++) {
			words[iTowerBlock_header+j] = xtal_data[j];
			if (debug_) cout << "update line " << iTowerBlock_header+j << endl;
		}
		if (debug_) {
			int jFE = pData[8*(iTowerBlock_header)];
			cout << "Front End on RD : " << jFE << endl; 
		}
		iTowerBlock_header += nlines;
	}

	// -- now the FEs are ordered. STill need to order the xtals within FEs;
	//    need : xtal 1,2,3,4, 5 in strip 1, xtal 1,2,3,4,5 in strip 2 etc..
	//    with possibly missing ones.

	if (debug_) cout << "now reorder the xtals within the FEs" << endl;

	iTowerBlock_header = iTowerBlock_header_keep;

        for (std::map<int, int>::const_iterator it=FrontEnd.begin();
                        it != FrontEnd.end(); it++) {

		int fe = it -> first;
		if (fe > 68) cout << "Problem... fe = " << fe << " in FEDid = " << FEDid << endl;
		if (debug_) cout << " This is for FE = " << fe << endl;
		int nlines = it -> second;
		int timesamples = pData[8*iTowerBlock_header+1] & 0x7F;
		int n4=timesamples-3;
        	int n_lines4 = n4/4;
        	if ( n4 % 4 != 0) n_lines4 ++;
        	if (n_lines4<0) n_lines4=0;
        	int Nxtal_max = (nlines-1)/(1+n_lines4);
		int Nxtal = 0;

		map< int, map<int, vector<Word64> > > Strip_Map;

		while (Nxtal < Nxtal_max) {

			int i_xtal = iTowerBlock_header+1 + Nxtal*(1+n_lines4);
			int strip = words[i_xtal] & 0x7;
			int xtal  = ( words[i_xtal] >>4) & 0x7;

			map< int, map<int, vector<Word64> > >::iterator iit = Strip_Map.find(strip);

			map<int, vector<Word64> > NewMap;
			map<int, vector<Word64> > Xtal_Map;

			if (iit == Strip_Map.end()) {            // new strip
				Xtal_Map = NewMap;
			}
			else {
				Xtal_Map = iit -> second;
			}

			std::vector<Word64> xtal_data;
			for (int j=0; j < n_lines4 +1; j++) {
                                Word64 ww = words[i_xtal +j];
                                xtal_data.push_back(ww);
                        }
                        Xtal_Map[xtal] = xtal_data;
                        Strip_Map[strip] = Xtal_Map;

			Nxtal ++;
		}

		// now, update the xtals for this FE :

		int idx = 0;
		for (map< int, map<int, vector<Word64> > >::const_iterator jt = Strip_Map.begin();
			jt != Strip_Map.end(); jt++) {

			int strip = jt -> first;
			if (debug_) cout << "   this is strip number " << strip << endl;
			map<int, vector<Word64> > Xtal_Map = jt -> second;

			for (map<int, vector<Word64> >::const_iterator kt = Xtal_Map.begin();
				kt != Xtal_Map.end(); kt++) {
				int xtal = kt -> first;
				if (debug_) cout << "       this is xtal number " << xtal << endl;
				vector<Word64> xtal_data = kt -> second;

				int mlines = (int)xtal_data.size();
				if (debug_) cout << "     mlines = " << mlines << endl;
				for (int j=0; j < mlines; j++) {
					int line = iTowerBlock_header+1+idx+j;
					if (line >= iDAQ_trailer) cout << "smth wrong... line " << line << " trailer " << iDAQ_trailer << endl;
					words[line] = xtal_data[j] ;
					if (debug_) cout << "      updated line " << iTowerBlock_header+idx+j << endl;
				}
				idx += mlines; 

			}  // end loop on xtals
			Xtal_Map.clear();

		}  // end loop on strips

		Strip_Map.clear();

		iTowerBlock_header += nlines;
	}  // end loop on FEs

	
	if (debug_) cout << " DONE FOR FED " << FEDid << endl;
	FrontEnd.clear();
	Map_xtal_data.clear();

  }  // end loop on DCC

  // cout << " finished reorder, now clean up " << endl;

// -- clean up

 // FEDmap.empty();
 // FEDorder.empty();

 // cout << "end of EndEvent " << endl;
}




void TowerBlockFormatter::DigiToRaw(const EEDataFrame& dataframe, FEDRawData& rawdata, const EcalElectronicsMapping* TheMapping)

	// -- now that we have the EcalElectronicsMapping, this method could probably be
	//    merged with DigiToRaw(EBdataframe).
	//    Keep as it is for the while...
{

 // debug_ = false;

 int bx = bx_;
 int lv1 = lv1_;


  int rdsize = rawdata.size() / 8;  // size in Word64

        const EEDetId& eedetid = dataframe.id();
	EcalElectronicsId elid = TheMapping -> getElectronicsId(eedetid);
        int DCCid = elid.dccId();
	int FEDid = FEDNumbering::MINECALFEDID + DCCid ;
	int iFE = elid.towerId();

	if (debug_) cout << "enter in TowerBlockFormatter::DigiToRaw DCCid FEDid iFE " <<
	dec << DCCid << " " << FEDid << " " << iFE << endl;

        int nsamples = dataframe.size();

        if (iFE <= 0 || iFE > 68)  {
		cout << "invalid iFE for EndCap DCCid iFE " << DCCid << " " << iFE << endl;
		return;
	}


        map<int, map<int,int> >::iterator fen = FEDorder.find(FEDid);
        map<int, map<int,int> >::iterator fed = FEDmap.find(FEDid);

        if (fen == FEDorder.end()) {
                if (debug_) cout << "New FED in TowerBlockFormatter " << dec << FEDid << " 0x" << hex << FEDid << endl;
                map<int,int> FEorder;
                pair<map<int, map<int,int> >::iterator, bool> t1 = FEDorder.insert(map<int, map<int,int> >::value_type(FEDid,FEorder));
                map<int,int> FEmap;
                pair<map<int, map<int,int> >::iterator, bool> t2 = FEDmap.insert(map<int, map<int,int> >::value_type(FEDid,FEmap));
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
                pData[8*FE_index+2] = bx & 0xFF;
		pData[8*FE_index+3] = (bx >>8) & 0x0F;
                pData[8*FE_index+3] |= 0xa0;
                pData[8*FE_index+4] = lv1 & 0xFF;
                pData[8*FE_index+5] = (lv1 >>8) & 0x0F;
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
        if (debug_) cout << "will add " << n_add << " lines of 64 bits at line " << (FE_index+1) << endl;
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










