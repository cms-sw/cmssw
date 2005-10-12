/*  
 *
 *  $Date: 2005/10/06 17:49:04 $
 *  $Revision: 1.5 $
 *  \author  N. Marinelli IASA 
 *
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


EcalTBDaqFormatter::EcalTBDaqFormatter (TFile* rootFile) {

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
 
 Char_t histo[20];

 if ( rootFile ) {
   rootFile->mkdir("EcalIntegrity");
   rootFile->cd("EcalIntegrity");
   for (int i = 0; i < 36 ; i++) {
     sprintf(histo, "EI SM%02d", i+1);
     meIntegrity[i] = new TH2D(histo, histo, 20, 0., 20., 85, 0., 85.);
   } 
 }

 theParser_ = new DCCDataParser(parameters);

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
  for( vector< DCCEventBlock * >::iterator itEventBlock = dccEventBlocks.begin(); itEventBlock != dccEventBlocks.end(); itEventBlock++){
    //  cout << " DCC ID " <<  (*itEventBlock)->getDataField("FED/DCC ID") << endl; 
    //cout << " BX number " << (*itEventBlock)->getDataField("BX") << endl;
    //cout << " RUN NUMBER  " <<  (*itEventBlock)->getDataField("RUN NUMBER") << endl;
    vector< DCCTowerBlock * > dccTowerBlocks = (*itEventBlock)->towerBlocks();
    cout << " EcalTBDaqFormatter::unFormatMe dccTowerBlocks size " << dccTowerBlocks.size() << endl;

    // Access the Tower block
    
    for( vector< DCCTowerBlock * >::iterator itTowerBlock = dccTowerBlocks.begin(); itTowerBlock!= dccTowerBlocks.end(); itTowerBlock++){
      tower=(*itTowerBlock)->towerID();
      if (  (*itTowerBlock)->towerID() > 68 ) continue;  // The last two towers does not contain Xtal data. Do not store in the digis


      //      cout << " Tower ID " << (*itTowerBlock)->towerID() << endl;
      vector<DCCXtalBlock * > & xtalDataBlocks = (*itTowerBlock)->xtalBlocks();
      

      
      // Access the Xstal data
      for( vector< DCCXtalBlock * >::iterator itXtalBlock = xtalDataBlocks.begin(); itXtalBlock!= xtalDataBlocks.end(); itXtalBlock++){
	//cout << " Xtal ID " << (*itXtalBlock)->xtalID() << " Strip ID " << (*itXtalBlock)->stripID() <<   endl;
      strip= (*itXtalBlock)->stripID();
      ch=(*itXtalBlock)->xtalID();

        
      pair<int,int> cellInd=cellIndex(tower, strip, ch); 
      EBDetId  id(cellInd.first, cellInd.second );           
      //   cout << " Unformatter Eta " << id.ieta() << " phi " << id.iphi() << endl;
      EBDataFrame theFrame ( id );
      vector<int> xtalDataSamples = (*itXtalBlock)->xtalDataSamples();   
      theFrame.setSize(xtalDataSamples.size());
      
   
      
      // cout << " Store the Adc counts " ;
      for (int i=0; i<xtalDataSamples.size(); ++i ) {
	theFrame.setSample (i, xtalDataSamples[i] );
      }
      
      
      digicollection.push_back(theFrame);
      


      }

      
    }
    
    
  }
 
  
  
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


