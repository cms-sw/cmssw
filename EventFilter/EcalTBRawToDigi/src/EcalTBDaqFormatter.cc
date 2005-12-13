/*  
 *
 *  $Date: 2005/12/12 07:25:30 $
 *  $Revision: 1.12 $
 *  \author  N. Marinelli IASA 
 *  \author G. Della Ricca
 *  \author G. Franzoni
 *  \author A. Ghezzi
 *
 */

#include "EcalTBDaqFormatter.h"
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h>
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>

#include "DCCDataParser.h"
#include "DCCEventBlock.h"
#include "DCCTowerBlock.h"
#include "DCCXtalBlock.h"
#include "DCCDataMapper.h"


using namespace std;
#include <iostream>


EcalTBDaqFormatter::EcalTBDaqFormatter () {

  cout << " EcalTBDaqFormatter CTOR " << endl;
  vector<ulong> parameters;
  parameters.push_back(10); // parameters[0] is the xtal samples 
  parameters.push_back(1);  // parameters[1] is the number of trigger time samples
  parameters.push_back(68);  // parameters[2] is the number of TT
  parameters.push_back(68);  // parameters[3] is the number of SR Flags
  parameters.push_back(1);  // parameters[4] is the dcc id
  parameters.push_back(1);  // parameters[5] is the sr id
  parameters.push_back(1);  // parameters[6] is the tcc1 id
  parameters.push_back(2);  // parameters[7] is the tcc2 id
  parameters.push_back(3);  // parameters[8] is the tcc3 id
  parameters.push_back(4);  // parameters[9] is the tcc4 id

  theParser_ = new DCCDataParser(parameters);

}





void EcalTBDaqFormatter::interpretRawData(const FEDRawData & fedData , EBDigiCollection& digicollection, EcalPnDiodeDigiCollection & pndigicollection , EBDetIdCollection & dccsizecollection , EcalTrigTowerDetIdCollection & ttidcollection , EcalTrigTowerDetIdCollection & blocksizecollection, EBDetIdCollection & chidcollection , EBDetIdCollection & gaincollection){

  const unsigned char * pData = fedData.data();
  int length = fedData.size();
  bool shit=true;
  int tower=0;
  int ch=0;
  int strip=0;

  cout << " [EcalTBDaqFormatter][interpretRawData] size " << length << endl;
  
  theParser_->parseBuffer( reinterpret_cast<ulong*>(pData), static_cast<ulong>(length), shit );
  

  vector< DCCEventBlock * > &   dccEventBlocks = theParser_->dccEvents();
    
  // Access each DCC block
  for( vector< DCCEventBlock * >::iterator itEventBlock = dccEventBlocks.begin(); 
       itEventBlock != dccEventBlocks.end(); 
       itEventBlock++){
      
    //cout << " DCC ID " <<  (*itEventBlock)->getDataField("FED/DCC ID") << endl; 
    //cout << " BX number " << (*itEventBlock)->getDataField("BX") << endl;
    //cout << " RUN NUMBER  " <<  (*itEventBlock)->getDataField("RUN NUMBER") << endl;
      
      
    bool _displayParserMessages = false;
    if( (*itEventBlock)->eventHasErrors() && _displayParserMessages){
      cout<<"\n[EcalTBDaqFormatter][interpretRawData]: errors found from parser... "<<endl;
      cout<<"\n "<<(*itEventBlock)->eventErrorString()<<endl;
      cout<<"\n[EcalTBDaqFormatter][interpretRawData]:... errors from parser  notified\n   "<<endl;
    }
      
    short TowerStatus[71];
    char buffer[20];
    for(int i=1;i<71;i++)
      { 
	sprintf(buffer, "FE_CHSTATUS#%d", i);
	string Tower(buffer);
	TowerStatus[i]= (*itEventBlock)->getDataField(Tower);
	//cout << "tower " << i << " has status " <<  TowerStatus[i] << endl;  
      }
      
    vector< DCCTowerBlock * > dccTowerBlocks = (*itEventBlock)->towerBlocks();
    cout << "[EcalTBDaqFormatter][interpretRawData] dccTowerBlocks size " << dccTowerBlocks.size() << endl;


    // build list of expected towers
    unsigned expTowersIndex=0;
    unsigned numExpectedTowers=0;
    unsigned ExpectedTowers[71];

    for (int u=1; u<71; u++)
      {     
	// 0 = tower expected; 1 = tower not expected
	if(TowerStatus[u] == 0)
	  {ExpectedTowers[expTowersIndex]=u;
	    expTowersIndex++;numExpectedTowers++;
	  }  
      }
    // resetting counter of expected towers
    expTowersIndex=0;
      
      
    // FIXME: dccID hard coded, for now
    //unsigned dccID = 5-1;
    unsigned dccID = 1-1;// at the moment SM is 1 by default (in DetID)
    // if number of dccEventBlocks NOT same as expected stop
    if (!      (dccTowerBlocks.size() == numExpectedTowers)      )
      {
	// we probably always want to know if this happens
	cout << "[EcalTBDaqFormatter][interpretRawData]"
	     << " number of TowerBlocks found ("
	     << dccTowerBlocks.size()
	     << ") differs from expected ("
	     << numExpectedTowers 
	     << ") skipping event"   << endl; 

	EBDetId idsm(1, 1 + 20 * dccID);
	dccsizecollection.push_back(idsm);

    // cout << "ERROR 1 " << idsm.ism() << endl;

	return;
      }
      
      
    unsigned previousTT =0;
    // Access the Tower block    
    for( vector< DCCTowerBlock * >::iterator itTowerBlock = dccTowerBlocks.begin(); 
	 itTowerBlock!= dccTowerBlocks.end(); 
	 itTowerBlock++){

      tower=(*itTowerBlock)->towerID();
      previousTT = tower;
      // checking if tt in data is the same as tt expected 
      // else skip tower and increment problem counter
	    
      int etaTT = (ExpectedTowers[expTowersIndex]-1)  /4;
      int phiTT = (ExpectedTowers[expTowersIndex]-1)  %4;

      EcalTrigTowerDetId idtt(etaTT, phiTT);

      if (  !(tower == ExpectedTowers[expTowersIndex])	  )
	{	
	  cout << "[EcalTBDaqFormatter][interpretRawData] TTower id found (=" 
	       << tower 
	       << ") different from expected (=" 
	       <<  ExpectedTowers[expTowersIndex] 
	       << ") " << (expTowersIndex+1) 
	       << "th tower checked" 
	       <<  endl; 
	    
	  // report on failed tt_id - ASSUME that
	    
	  ttidcollection.push_back(idtt);
      
      // cout << "ERROR 2 " << idtt.ieta() << "  " << idtt.iphi() << endl;

	  ++ expTowersIndex;
	  continue;	
	}// if TT id found  different than expected 
	

	
      /*********************************
       //    tt: 1 ... 68: crystal data
       *********************************/
      if ( 0<  (*itTowerBlock)->towerID() &&
	   (*itTowerBlock)->towerID() < 69) {

	vector<DCCXtalBlock * > & xtalDataBlocks = (*itTowerBlock)->xtalBlocks();	
	if (xtalDataBlocks.size() != 25)
	  {     
	    cout << "[EcalTBDaqFormatter][interpretRawData]  wrong dccBlock size is: "  
		 << xtalDataBlocks.size() 
		 << " in event " << (*itEventBlock)->getDataField("LV1")
		 << " for TT " << ExpectedTowers[expTowersIndex] 
		 << endl;

            blocksizecollection.push_back(idtt);
      
            // cout << "ERROR 3 " << idtt.ieta() << "  " << idtt.iphi() << endl;
        
	    ++ expTowersIndex;
	    continue;	
	  }

	short expStripInTower;
	short expCryInStrip;
	short expCryInTower =0;

	// Access the Xstal data
	for( vector< DCCXtalBlock * >::iterator itXtalBlock = xtalDataBlocks.begin(); 
	     itXtalBlock!= xtalDataBlocks.end(); 
	     itXtalBlock++){

	  strip = (*itXtalBlock)->stripID();
	  ch    =(*itXtalBlock)->xtalID();

	  // these are the expected indices
	  expStripInTower = expCryInTower/5 +1;
	  expCryInStrip   =  expCryInTower%5 +1;
	  
	  // FIXME: waiting for geometry to do (TT, strip,chNum) <--> (SMChId)
	  // short abscissa = (ExpectedTowers[expTowersIndex]-1)  /4;
	  // short ordinate = (ExpectedTowers[expTowersIndex]-1)  %4;
	  // temporarily choosing central crystal in trigger tower
	  // int cryIdInSM  = 45 + ordinate*5 + abscissa * 100;
    
	  // comparison: expectation VS crystal in data
	  if(!	   (strip == expStripInTower &&
		    ch     == expCryInStrip )	     )
	    {

	      // 		// only for debugging purposes
	      // 		if (1){
	      // 		  cout << "[EcalTBDaqFormatter][interpretRawData]  expected:\t tt " 
	      // 		       << ExpectedTowers[expTowersIndex]
	      // 		       << "  strip " << expStripInTower 
	      // 		       << "  cry " << expCryInStrip << "\tfound: "
	      // 		       << "\t tt " << ExpectedTowers[expTowersIndex] 
	      // 		       << "  strip " <<  strip
	      // 		       << "  cry " << ch <<  endl;
	      // 		}
		
              pair<int,int> cellIndExp=cellIndex(tower, expStripInTower, expCryInStrip); 
              EBDetId  idExp(cellIndExp.first, cellIndExp.second );           

              chidcollection.push_back(idExp);

              // cout << "ERROR 4 " << idExp.ieta() << " " << idExp.iphi() << endl;

	      expCryInTower++; continue;
		
	    }


      // data  to be stored in EBDataFrame, identified by EBDetId
      pair<int,int> cellInd=cellIndex(tower, strip, ch); 
      EBDetId  id(cellInd.first, cellInd.second );           
      
	     EBDataFrame theFrame ( id );
	     vector<int> xtalDataSamples = (*itXtalBlock)->xtalDataSamples();   
	     theFrame.setSize(xtalDataSamples.size());
      
   
	     // gain cannot be 0, checking for that
	     bool        gainIsOk =true;
	     unsigned gain_mask      = 12288;    //12th and 13th bit
	     for (int i=0; i<xtalDataSamples.size(); ++i ) {
	       theFrame.setSample (i, xtalDataSamples[i] );
	       if((xtalDataSamples[i] & gain_mask) == 0)
		 {gainIsOk =false;}
	     }

	     if (! gainIsOk) {

           gaincollection.push_back(id);

           // cout << "ERROR 5 " << id.ieta() << " " << id.iphi() << endl;
         
         }

	     digicollection.push_back(theFrame);
	     
	     expCryInTower++; 
	     }// end loop on crystals

	  expTowersIndex++;
	}// end: tt1 ... tt68, crystal data
	
	
	
	
	
	
	
	

	
      /**************************************************
       //    tt 69 and 70:  two mem boxes, holding PN0 ... PN9
       *********************************************************/	
      else if (       (*itTowerBlock)->towerID() == 69 
		      ||	   (*itTowerBlock)->towerID() == 70
	       )
	
	  {// if it is a mem box
	  
	    cout << "[EcalTBDaqFormatter][interpretRawData]  processing mem box num"
		 << (*itTowerBlock)->towerID()
		 << endl;
	  
	  
	    // checking mem data size, as a tt
	    vector<DCCXtalBlock * > & xtalDataBlocks = (*itTowerBlock)->xtalBlocks();	
	    if (xtalDataBlocks.size() != 25)
	      {     
		cout << "[EcalTBDaqFormatter][interpretRawData]  wrong dccBlock size is: "  
		     << xtalDataBlocks.size() 
		     << " in event " << (*itEventBlock)->getDataField("LV1")
		     << " for mem " << ExpectedTowers[expTowersIndex] 
		     << endl;
		// fixme giofr: need monitoring element for mem integrity
		++ expTowersIndex;
		continue;	
	      }

	    // prepare the structure for the PN MEM data according to the code by P.Verrecchia
	    for(int is=0;is<5;is++){
	      for(int ic=0; ic<5; ic++){
		for(int sa =0; sa<11; sa++){
		  fem[is][ic][sa]=0;
		}
	      }
	    }


	    vector<DCCXtalBlock *> & dccXtalBlocks = (*itTowerBlock)->xtalBlocks();
	    vector<DCCXtalBlock*>::iterator itXtal; 	// itXtal = 0 - 24
	    int  cryCounter = 0;
	    int  strip_id  = 0;
	    int  xtal_id   = 0;
	    for ( itXtal = dccXtalBlocks.begin(); itXtal < dccXtalBlocks.end(); itXtal++ ) {
	      strip_id  = (*itXtal) ->getDataField("STRIP ID");
	      xtal_id   = (*itXtal) ->getDataField("XTAL ID");
	  
	      int wished_strip_id  = cryCounter/5+1;
	      int wished_ch_id     = cryCounter%5+1;
	    
	      if( wished_strip_id != ((int)strip_id) || wished_ch_id != ((int)xtal_id) )
		{
		  cout << "[EcalTBDaqFormatter][interpretRawData] in mem " 
		       <<  (*itTowerBlock)->towerID()
		       << " expected:\t strip"
		       << wished_strip_id  << "  cry " << wished_ch_id << "\tfound: "
		       << "  strip " <<  strip_id << "  cry " << xtal_id <<  endl;
		  // report on crystal with unexpected indices
		  // fixme giofr: need monitoring element for mem integrity
		}
	    
	      // Accessing the 10 time samples per Xtal:
	      fem[wished_strip_id-1][wished_ch_id-1][1] = (*itXtal)->getDataField("ADC#1");
	      fem[wished_strip_id-1][wished_ch_id-1][2] = (*itXtal)->getDataField("ADC#2");
	      fem[wished_strip_id-1][wished_ch_id-1][3] = (*itXtal)->getDataField("ADC#3");
	      fem[wished_strip_id-1][wished_ch_id-1][4] = (*itXtal)->getDataField("ADC#4");
	      fem[wished_strip_id-1][wished_ch_id-1][5] = (*itXtal)->getDataField("ADC#5");
	      fem[wished_strip_id-1][wished_ch_id-1][6] = (*itXtal)->getDataField("ADC#6");
	      fem[wished_strip_id-1][wished_ch_id-1][7] = (*itXtal)->getDataField("ADC#7");
	      fem[wished_strip_id-1][wished_ch_id-1][8] = (*itXtal)->getDataField("ADC#8");
	      fem[wished_strip_id-1][wished_ch_id-1][9] = (*itXtal)->getDataField("ADC#9");
	      fem[wished_strip_id-1][wished_ch_id-1][10] = (*itXtal)->getDataField("ADC#10");

	      cryCounter++;
	    }// end loop on cry of dccXtalBlock (=tower)
	  
	    previousTT = (*itTowerBlock)->towerID();
	    ++ expTowersIndex;;

	    // unpacks and stores samples: TT 69 data_MEM[0...249], TT 70 data_MEM[250...499], 
	    DecodeMEM(int( (*itTowerBlock)->towerID() ));
	  
	    int currentMemId       = ( (*itTowerBlock)->towerID() -69);

	    // looping on PN's of current mem box
	    for (int pnId = 1;  pnId < 6; pnId++){

	      // fixme giof: second argumenti is DCCId, to be determined
	      EcalPnDiodeDetId PnId(1, 1, pnId + 5*currentMemId);
	      EcalPnDiodeDigi thePnDigi(PnId );
	      thePnDigi.setSize(50);
	      for (int sample =0; sample<50; sample++)
		{thePnDigi.setSample(sample, data_MEM[(pnId-1)*50 + sample ] );  
		
		
//		  if (pnId==1){
//		    cout << "[Formatter] sample: " << sample
//			 << " " 
//			 <<data_MEM[(pnId-1)*50 + sample ];
//		  }
		
		
		}
	      pndigicollection.push_back(thePnDigi);
	    }
	    cout << endl;
	  }// end of < if it is a mem box>
	



	// wrong tt id
	else  {
	  cout <<"[EcalTBDaqFormatter][interpretRawData]  wrong tt id ( "
	       <<  (*itTowerBlock)->towerID() << ")"  << endl;
	  ++ expTowersIndex;continue; 
	}// end: tt id error



      }// end loop on trigger towers
      
    }// end loop on events
  }



  // Code from P.Verrecchia to store sequentially the 10 PNs with 50 samples each into data_MEM[500]
  // DecodeMEM will soon be put in a separate class dedicated to PN unpacking
  void EcalTBDaqFormatter::DecodeMEM(int tower_id){
    // cout<<"enetering in DecodeMEM for TT "<<tower_id<<endl;
    if(tower_id != 69 && tower_id != 70)
      {cout << "[EcalTBDaqFormatter][DecodeMEM] this is not a mem box tower ("
	    << tower_id << ")"<< endl;
	return;
      }

    int XSAMP = 10;
    int mem_id= tower_id-69;
    int tmp_data=0;
    int indbuf=0;
    int ipn=0;
    for(int st=0;st<5;st++) {
      for(int ch=0;ch<5;ch++) {
	if(st%2 == 0) ipn= mem_id*5+ch;
	else ipn=mem_id*5+4-ch;
	for(int ec=0;ec<XSAMP;ec++) {
	  tmp_data= fem[st][ch][ec+1];
	
	  int new_data=0;
	  if(st%2 == 1) {
	    for(int ib=0;ib<14;ib++)
	      {
		new_data <<= 1;
		new_data=new_data | (tmp_data&1);
		tmp_data >>= 1;
	      }
	  } else {
	    new_data=tmp_data;
	  }
	  // Reinvert bit 11 for AD9052 still there on MEM !
	  new_data = (new_data ^ 0x800) & 0x1fff;
	  indbuf= ipn*50+st*XSAMP+ec;
	  data_MEM[indbuf]= new_data;
	}
      }
    }
  }// end DecodeMEM

  
  pair<int,int>  EcalTBDaqFormatter::cellIndex(int tower_id, int strip, int ch) {
 
    int xtal= (strip-1)*5+ch-1;
    //  cout << " cellIndex input xtal " << xtal << endl;
    pair<int,int> ind;

    int eta = (tower_id - 1)/kTowersInPhi*kCardsPerTower;
    int phi = (tower_id - 1)%kTowersInPhi*kChannelsPerCard;

    if (rightTower(tower_id))
      eta += xtal/kCardsPerTower;
    else
      eta += (kCrystalsPerTower - 1 - xtal)/kCardsPerTower;

    if (rightTower(tower_id) && (xtal/kCardsPerTower)%2 == 1 ||
	!rightTower(tower_id) && (xtal/kCardsPerTower)%2 == 0)

      phi += (kChannelsPerCard - 1 - xtal%kChannelsPerCard);
    else
      phi += xtal%kChannelsPerCard;


    ind.first =eta+1;  
    ind.second=phi+1; 

    //  cout << "  EcalTBDaqFormatter::cell_index eta " << ind.first << " phi " << ind.second << " " << endl;

    return ind;

  }


  bool EcalTBDaqFormatter::rightTower(int tower) const {
  
    if ((tower>12 && tower<21) || (tower>28 && tower<37) ||
	 (tower>44 && tower<53) || (tower>60 && tower<69))
      return true;
    else
      return false;
  }



  bool EcalTBDaqFormatter::leftTower(int tower) const
  {
    return !rightTower(tower);
  }


