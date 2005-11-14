/*  
 *
 *  $Date: 2005/10/18 09:06:15 $
 *  $Revision: 1.7 $
 *  \author  N. Marinelli IASA 
 *  \author G. Della Ricca
 *  \author G. Franzoni
 *  \author A. Ghezzi
 *
 */

#include "EcalTBDaqFormatter.h"
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>

#include "DCCDataParser.h"
#include "DCCEventBlock.h"
#include "DCCTowerBlock.h"
#include "DCCXtalBlock.h"
#include "DCCDataMapper.h"


using namespace std;
#include <iostream>


EcalTBDaqFormatter::EcalTBDaqFormatter (DaqMonitorBEInterface *dbe) {

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

  //! booking histograms to collect data integrity errors

  Char_t histo[20];

  // checking when gain=0
  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel");
    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity");
    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity/Gain");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EI gain SM%02d", i+1);
      meIntegrityGain[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    } 
    
    // checking when channel has unexpected or invalid ID
    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity/ChId");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EI ChId SM%02d", i+1);
      meIntegrityChId[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    } 

    // checking when trigger tower has unexpected or invalid ID
    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity/TTId");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EI TTId SM%02d", i+1);
      meIntegrityTTId[i] = dbe->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
    }

    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity/TTBlockSize");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EI TTBlockSize SM%02d", i+1);
      meIntegrityTTBlockSize[i] = dbe->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
    } 

    // checking when number of towers in data different than expected from header
    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity");
    sprintf(histo, "DCC size error");
    meIntegrityDCCSize = dbe->book1D(histo, histo, 36, 1, 37.);

  }

}

void EcalTBDaqFormatter::interpretRawData(const FEDRawData & fedData , EBDigiCollection& digicollection ){

  const unsigned char * pData = fedData.data();
  int length = fedData.size();
  bool shit=true;
  int tower=0;
  int ch=0;
  int strip=0;

  cout << " EcalTBDaqFormatter::interpretRawData size " << length << endl;
  
  theParser_->parseBuffer( reinterpret_cast<ulong*>(pData), static_cast<ulong>(length), shit );
  
  /// just for debug
    /*
      ulong * temp = (ulong*)(pData);
      for ( int i=0; i<20; ++i, temp++)
      cout << "i)"<< i << " " << hex<<(*temp)<< dec << endl;
    */

    vector< DCCEventBlock * > &   dccEventBlocks = theParser_->dccEvents();
    //  cout << " EcalTBDaqFormatter::interpretRawData  dccDataBlocks size " << dccEventBlocks.size() << endl;
    
    // Access each DCC block
    for( vector< DCCEventBlock * >::iterator itEventBlock = dccEventBlocks.begin(); 
	 itEventBlock != dccEventBlocks.end(); 
	 itEventBlock++){
      
      //  cout << " DCC ID " <<  (*itEventBlock)->getDataField("FED/DCC ID") << endl; 
      //cout << " BX number " << (*itEventBlock)->getDataField("BX") << endl;
      //cout << " RUN NUMBER  " <<  (*itEventBlock)->getDataField("RUN NUMBER") << endl;
      
      
      //!See if we have errors form parser. To be better organized with verbosity
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
      cout << " EcalTBDaqFormatter::unFormatMe dccTowerBlocks size " << dccTowerBlocks.size() << endl;

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
	       << ") skipping event" 
	       << endl; 
	  if ( meIntegrityDCCSize ) meIntegrityDCCSize->Fill(dccID+1);
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
	    
	    short abscissa = (ExpectedTowers[expTowersIndex]-1)  /4;
	    short ordinate = (ExpectedTowers[expTowersIndex]-1)  %4;
	    
	    if ( meIntegrityTTId[dccID] ) meIntegrityTTId[dccID]->Fill(abscissa,ordinate);
	    ++ expTowersIndex;
	    continue;	
	  }// if TT id found  different than expected 
	


	// FIXME: MEM boxes to be taken care of
	// The last two towers does not contain Xtal data. Do not store in the digis
	if (  (*itTowerBlock)->towerID() > 68 ) {++ expTowersIndex;continue; } 

	
	short expStripInTower;
	short expCryInStrip;
	short expCryInTower =0;

	//      cout << " Tower ID " << (*itTowerBlock)->towerID() << endl;
	vector<DCCXtalBlock * > & xtalDataBlocks = (*itTowerBlock)->xtalBlocks();	
	if (xtalDataBlocks.size() != 25)
	  {     
	    // we probably always want to know if this has happened
	    cout << "[EcalTBDaqFormatter][interpretRawData]  wrong dccBlock size is: "  
		 << xtalDataBlocks.size() 
		 << " in event " << (*itEventBlock)->getDataField("LV1")
		 << " for TT " << ExpectedTowers[expTowersIndex] 
		 << endl;
	  short abscissa = (ExpectedTowers[expTowersIndex]-1)  /4;
	  short ordinate = (ExpectedTowers[expTowersIndex]-1)  %4;
	  if ( meIntegrityTTBlockSize[dccID] ) meIntegrityTTBlockSize[dccID]->Fill(abscissa,ordinate);
	  ++ expTowersIndex;
	    continue;	
	  }


	// Access the Xstal data
	for( vector< DCCXtalBlock * >::iterator itXtalBlock = xtalDataBlocks.begin(); 
	     itXtalBlock!= xtalDataBlocks.end(); 
	     itXtalBlock++){
	  	  //cout << " Xtal ID " << (*itXtalBlock)->xtalID() << " Strip ID " << (*itXtalBlock)->stripID() <<   endl;
	  strip = (*itXtalBlock)->stripID();
	  ch    =(*itXtalBlock)->xtalID();

	  // these are the expected indices
	  expStripInTower = expCryInTower/5 +1;
	  expCryInStrip   =  expCryInTower%5 +1;
	  
	  // FIXME: waiting for geometry to do (TT, strip,chNum) <--> (SMChId)
	  //	  int cryIdInSM =_mySMGeom.getSMCrystalNumber(tower_id,expStripInTower,expCryInStrip);
	  short abscissa = (ExpectedTowers[expTowersIndex]-1)  /4;
	  short ordinate = (ExpectedTowers[expTowersIndex]-1)  %4;
	  // temporarily choosing central crystal in trigger tower
	  int cryIdInSM  = 45 + ordinate*5 + abscissa * 100;
    
	  // comparison: expectation VS crystal in data
	  if(!	   (strip == expStripInTower &&
		    ch     == expCryInStrip )	     )
	    {
	      // only for debugging purposes
	      if (1){
		cout << "[MonitorEventFiller][checkIndex] expected:\t tt " 
		     << ExpectedTowers[expTowersIndex]
		     << "  strip " << expStripInTower 
		     << "  cry " << expCryInStrip << "\tfound: "
		     << "\t tt " << ExpectedTowers[expTowersIndex] 
		     << "  strip " <<  strip
		     << "  cry " << ch <<  endl;
	      }
	      // filling histogram reporting chID errors
	      if ( meIntegrityChId[dccID] ) meIntegrityChId[dccID]->Fill(cryIdInSM /20, cryIdInSM %20);  

	      expCryInTower++; continue;
	    }



        
	  pair<int,int> cellInd=cellIndex(tower, strip, ch); 
	  EBDetId  id(cellInd.first, cellInd.second );           
	  //   cout << " Unformatter Eta " << id.ieta() << " phi " << id.iphi() << endl;
	  EBDataFrame theFrame ( id );
	  vector<int> xtalDataSamples = (*itXtalBlock)->xtalDataSamples();   
	  theFrame.setSize(xtalDataSamples.size());
      
   
      
	  // cout << " Store the Adc counts " ;
	  bool        gainIsOk =true;
	  unsigned gain_mask      = 12288;    //12th and 13th bit
	  for (int i=0; i<xtalDataSamples.size(); ++i ) {
	    theFrame.setSample (i, xtalDataSamples[i] );
	    if((xtalDataSamples[i] & gain_mask) == 0)
	      {gainIsOk =false;}
	  }

	  if (! gainIsOk) 
	      { if ( meIntegrityGain[dccID] ) meIntegrityGain[dccID]->Fill(cryIdInSM /20, cryIdInSM %20);}
 
	  digicollection.push_back(theFrame);
	  
	  expCryInTower++; 
	}// end loop on crystals

	expTowersIndex++;
      }// end loop on trigger towers
    }// end loop on events
}

  
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


