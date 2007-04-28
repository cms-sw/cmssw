/*  
 *
 *  $Date: 2007/04/10 10:45:37 $
 *  $Revision: 1.12 $
 *  \author  N. Marinelli IASA 
 *  \author G. Della Ricca
 *  \author G. Franzoni
 *  \author A. Ghezzi
 *
 */


#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h>
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>

#include "EventFilter/EcalRawToDigi/src/EcalDCCDaqFormatter.h"
#include "EventFilter/EcalRawToDigi/interface/EcalDCCHeaderRuntypeDecoder.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataParser.h"
#include "EventFilter/EcalRawToDigi/src/DCCEventBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCTowerBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCXtalBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataMapper.h"
#include "EventFilter/EcalRawToDigi/src/DCCMapper.h"

using namespace edm;
using namespace std;

#include <iostream>

EcalDCCDaqFormatter::EcalDCCDaqFormatter () {

  LogDebug("EcalRawToDigi") << "@SUB=EcalDCCDaqFormatter";
  vector<ulong> parameters;
  parameters.push_back(10); // parameters[0] is the xtal samples 
  parameters.push_back(1);  // parameters[1] is the number of trigger time samples
  parameters.push_back(68); // parameters[2] is the number of TT
  parameters.push_back(68); // parameters[3] is the number of SR Flags
  parameters.push_back(1);  // parameters[4] is the dcc id
  parameters.push_back(1);  // parameters[5] is the sr id
  parameters.push_back(1);  // parameters[6] is the tcc1 id
  parameters.push_back(2);  // parameters[7] is the tcc2 id
  parameters.push_back(3);  // parameters[8] is the tcc3 id
  parameters.push_back(4);  // parameters[9] is the tcc4 id

  theParser_ = new DCCDataParser(parameters);

}

void EcalDCCDaqFormatter::interpretRawData(const FEDRawData & fedData , 
					   EBDigiCollection& digicollection, EcalPnDiodeDigiCollection & pndigicollection , 
					   EcalRawDataCollection& DCCheaderCollection, 
					   EBDetIdCollection & dccsizecollection , 
					   EcalTrigTowerDetIdCollection & ttidcollection , EcalTrigTowerDetIdCollection & blocksizecollection,
					   EBDetIdCollection & chidcollection , EBDetIdCollection & gaincollection, 
					   EBDetIdCollection & gainswitchcollection, EBDetIdCollection & gainswitchstaycollection, 
					   EcalElectronicsIdCollection & memttidcollection,  EcalElectronicsIdCollection &  memblocksizecollection,
					   EcalElectronicsIdCollection & memgaincollection,  EcalElectronicsIdCollection & memchidcollection)
{

  const unsigned char * pData = fedData.data();
  int length = fedData.size();
  if (!length)
    return;
  if (!theMapper_)
    return;

  bool shit=true;
  unsigned int tower=0;
  int ch=0;
  int strip=0;
  
  LogDebug("EcalRawToDigi") << "@SUB=EcalDCCDaqFormatter::interpretRawData"
			    << "size " << length;
  
 
  // mean + 3sigma estimation needed when switching to 0suppressed data
  //  digicollection.reserve(kCrystals);
  pnAllocated = false;
  


  theParser_->parseBuffer( reinterpret_cast<ulong*>(const_cast<unsigned char*>(pData)), static_cast<ulong>(length), shit );
  
  vector< DCCEventBlock * > &   dccEventBlocks = theParser_->dccEvents();

  // Access each DCC block
  for( vector< DCCEventBlock * >::iterator itEventBlock = dccEventBlocks.begin(); 
       itEventBlock != dccEventBlocks.end(); 
       itEventBlock++){

    LogDebug("EcalRawToDigi") << " DCC ID " <<  (*itEventBlock)->getDataField("FED/DCC ID"); 
    LogDebug("EcalRawToDigi") << " BX number " << (*itEventBlock)->getDataField("BX");
    LogDebug("EcalRawToDigi") << " RUN NUMBER  " <<  (*itEventBlock)->getDataField("RUN NUMBER");

    int fedId=(*itEventBlock)->getDataField("FED/DCC ID");      // fed id as found in raw data (0... 35 at tb, 610... 645 for EB at CMS )
    int SMid=theMapper_->getSMId(fedId);                        // global unpacker: numbers ranging from 1 to 36 as EB slots
    
    bool _displayParserMessages = false;
    if( (*itEventBlock)->eventHasErrors() && _displayParserMessages)
      {
        LogWarning("EcalRawToDigi") << "@SUB=EcalDCCDaqFormatter::interpretRawData"
				    << "errors found from parser... ";
        LogWarning("EcalRawToDigi") << (*itEventBlock)->eventErrorString();
        LogWarning("EcalRawToDigi") << "@SUB=EcalDCCDaqFormatter::interpretRawData"
				    << "... errors from parser notified";
      }

    // getting the fields of the DCC header
    EcalDCCHeaderBlock theDCCheader;
    theDCCheader.setId( SMid );          // geometrical slot in CMS ECAL barrael (EB in pre-commissioning conventions: 1... 36)
    theDCCheader.setFedId( fedId );      // fed id as found in raw data (0... 35 at tb, 610... 645 for EB in CMS  )

    theDCCheader.setRunNumber((*itEventBlock)->getDataField("RUN NUMBER"));
    short trigger_type = (*itEventBlock)->getDataField("TRIGGER TYPE");
    if(trigger_type >0 && trigger_type <5){theDCCheader.setBasicTriggerType(trigger_type);}

    else{ LogWarning("EcalRawToDigiTriggerType") << "@SUB=EcalDCCDaqFormatter::interpretRawData"
					<< "unrecognized TRIGGER TYPE: "<<trigger_type;}

    theDCCheader.setLV1((*itEventBlock)->getDataField("LV1"));
    theDCCheader.setBX((*itEventBlock)->getDataField("BX"));
    theDCCheader.setErrors((*itEventBlock)->getDataField("DCC ERRORS"));
    theDCCheader.setSelectiveReadout((*itEventBlock)->getDataField("SR"));
    theDCCheader.setZeroSuppression((*itEventBlock)->getDataField("ZS"));
    theDCCheader.setTestZeroSuppression((*itEventBlock)->getDataField("TZS"));
    theDCCheader.setSrpStatus((*itEventBlock)->getDataField("SR_CHSTATUS"));

    short zs  = (*itEventBlock)->getDataField("ZS");
    short tzs = (*itEventBlock)->getDataField("TZS");
    bool  dataIsSuppressed;
    // if zs&&tzs the suppression algo is used in DCC, the data are not suppressed and zs-bits are set
    if ( zs && !(tzs) ) dataIsSuppressed = true;
    else  dataIsSuppressed = false;


    vector<short> theTCCs;
    for(int i=0; i<MAX_TCC_SIZE; i++){
      std::ostringstream TCCnum;
      TCCnum << "TCC_CHSTATUS#" << i+1;
      string TCCnumS(TCCnum.str());
      theTCCs.push_back ((*itEventBlock)->getDataField(TCCnumS) );
    }
    theDCCheader.setTccStatus(theTCCs);

    short TowerStatus[MAX_TT_SIZE+1];
    vector<short> theTTstatus;
    for(int i=1;i<MAX_TT_SIZE+1;i++)
      { 
	std::ostringstream buffer;
	buffer << "FE_CHSTATUS#" << i;
	string Tower(buffer.str());
 	TowerStatus[i]= (*itEventBlock)->getDataField(Tower);
	theTTstatus.push_back(TowerStatus[i]);
	LogDebug("EcalRawToDigi") << "tower " << i << " has status " <<  TowerStatus[i];  
      }

    theDCCheader.setTriggerTowerStatus(theTTstatus);
    
    EcalDCCHeaderRuntypeDecoder theRuntypeDecoder;
    ulong DCCruntype = (*itEventBlock)->getDataField("RUN TYPE");
    theRuntypeDecoder.Decode(DCCruntype, &theDCCheader);

    // DCCHeader filled 
    DCCheaderCollection.push_back(theDCCheader);
    
    vector< DCCTowerBlock * > dccTowerBlocks = (*itEventBlock)->towerBlocks();
    LogDebug("EcalRawToDigi") << "@SUBS=EcalDCCDaqFormatter::interpretRawData"
			      << "dccTowerBlocks size " << dccTowerBlocks.size();



    _expTowersIndex=0;_numExpectedTowers=0;
    for (int v=0; v<71; v++){
      _ExpectedTowers[v]=0;
    }
    // staus==0:  tower expected; status==1: tower not expected
    for (int u=1; u< (kTriggerTowersAndMem+1); u++)
      {
        if(TowerStatus[u] == 0)
          {_ExpectedTowers[_expTowersIndex]=u;
	    _expTowersIndex++;
	    _numExpectedTowers++;
          }  
      }
    // resetting counter of expected towers
    _expTowersIndex=0;
      

    // if number of dccEventBlocks NOT same as expected stop
    if (!      (dccTowerBlocks.size() == _numExpectedTowers)      )
      {
        // we probably always want to know if this happens

        LogWarning("EcalRawToDigiNumTowerBlocks") << "@SUB=EcalDCCDaqFormatter::interpretRawData"
				      << "number of TowerBlocks found (" << dccTowerBlocks.size()
				      << ") differs from expected (" << _numExpectedTowers 
				      << ") skipping event"; 

        EBDetId idsm(SMid, 1,EBDetId::SMCRYSTALMODE);
        dccsizecollection.push_back(idsm);

        return;
      }
      

  


    // Access the Tower block    
    for( vector< DCCTowerBlock * >::iterator itTowerBlock = dccTowerBlocks.begin(); 
         itTowerBlock!= dccTowerBlocks.end(); 
         itTowerBlock++){

      tower = (*itTowerBlock)->towerID();
      
      // checking if tt in data is the same as tt expected 
      // else skip tower and increment problem counter
	    
      // compute eta/phi in order to have iTT = _ExpectedTowers[_expTowersIndex]
      // for the time being consider only zside>0

      // report on failed tt_id
      int zIndex = (SMid < 19 ? 1 : -1 );
      int etaTT = (_ExpectedTowers[_expTowersIndex]-1)  / kTowersInPhi +1;
      int phiTT = 0;
      
      if (zIndex > 0)
	phiTT=(SMid - 1) * kTowersInPhi + (kTowersInPhi - ((_ExpectedTowers[_expTowersIndex]-1)  % kTowersInPhi));
      else
	phiTT=(SMid - 19)* kTowersInPhi + (kTowersInPhi - ((_ExpectedTowers[_expTowersIndex]-1)  % kTowersInPhi));
      
      EcalTrigTowerDetId idtt(zIndex, EcalBarrel, etaTT, phiTT, 0);
      
      if (  !(tower == _ExpectedTowers[_expTowersIndex])	  )
        {	

	  if (_ExpectedTowers[_expTowersIndex] <= 68){
	    edm::LogWarning("EcalRawToDigiTowerId") << "@SUBS=EcalDCCDaqFormatter::interpretRawData"
						      << "TTower id found (=" << tower 
						      << ") different from expected (=" <<  _ExpectedTowers[_expTowersIndex] 
						      << ") " << (_expTowersIndex+1) << "-th tower checked"; 
	    
	    //  report on failed tt_id for regular tower block
	    ttidcollection.push_back(idtt);
	  }
	  else
	    {
	      edm::LogWarning("EcalRawToDigiTowerId") << "@SUB=EcalDCCDaqFormatter:interpretRawData"
							<< "DecodeMEM: tower " << tower  
							<< " is not the same as expected " << ((int)_ExpectedTowers[_expTowersIndex])
							<< " (according to DCC header channel status)";
	      
	      // report on failed tt_id for mem tower block
	      // chosing channel 1 as representative
	      EcalElectronicsId id(1, (int)_ExpectedTowers[_expTowersIndex], 1, 1);
	      memttidcollection.push_back(id);
	    }

          ++ _expTowersIndex;
          continue;	
        }// if TT id found  different than expected 
	



      /*********************************
       //    tt: 1 ... 68: crystal data
       *********************************/
      if (  0<  (*itTowerBlock)->towerID() &&
	    (*itTowerBlock)->towerID() < (kTriggerTowers+1) )
 	{
	  
	  vector<DCCXtalBlock * > & xtalDataBlocks = (*itTowerBlock)->xtalBlocks();	
	  if (xtalDataBlocks.size() != kChannelsPerTower)
	    {     
	      LogWarning("EcalRawToDigiTowerSize") << "@SUB=EcalDCCDaqFormatter::interpretRawData"
					    << "wrong dccBlock size is: "  << xtalDataBlocks.size() 
					    << " at LV1 " << (*itEventBlock)->getDataField("LV1")
					    << " for TT " << _ExpectedTowers[_expTowersIndex];
	      // report on wrong tt block size
	      blocksizecollection.push_back(idtt);

	      ++ _expTowersIndex; 	      continue;	
	    }

	
	  short expStripInTower;
	  short expCryInStrip;
	  short expCryInTower =0;
	  
	  // Access the Xstal data
	  for( vector< DCCXtalBlock * >::iterator itXtalBlock = xtalDataBlocks.begin(); 
	       itXtalBlock!= xtalDataBlocks.end(); 
	       itXtalBlock++){
	    
	    strip = (*itXtalBlock)->stripID();
	    ch    = (*itXtalBlock)->xtalID();
	    
	    if (! dataIsSuppressed )
	      {

		// these are the expected indices
		expStripInTower =  expCryInTower/5 +1;
		expCryInStrip   =  expCryInTower%5 +1;
		
		
		// FIXME: waiting for geometry to do (TT, strip,chNum) <--> (SMChId)
		// short abscissa = (_ExpectedTowers[_expTowersIndex]-1)  /4;
		// short ordinate = (_ExpectedTowers[_expTowersIndex]-1)  %4;
		// temporarily choosing central crystal in trigger tower
		// int cryIdInSM  = 45 + ordinate*5 + abscissa * 100;
		
		// comparison: expectation VS crystal in data
		if(!	   (strip == expStripInTower &&
			    ch    == expCryInStrip )	     
		   )
		  {
		    
		    int ic    = cryIc(tower, expStripInTower,  expCryInStrip) ;
		    EBDetId   idExp(SMid, ic,EBDetId::SMCRYSTALMODE);
		    
		    LogWarning("EcalRawToDigiChId") << "EcalTBDaqFormatter::interpretRawData no zero suppression "
						    << " wrong channel id for channel: "  << expCryInStrip
						    << "\t strip: " << expStripInTower
						    << "\t in TT: " << _ExpectedTowers[_expTowersIndex]
						    << "\t at LV1: " << (*itEventBlock)->getDataField("LV1")
						    << "\t   (in the data, found channel:  " << ch
						    << "\t strip:  " << strip << " ).";
		    
		    
		    // report on wrong channel id
		    chidcollection.push_back(idExp);
		    
		    // there has been unexpected crystal id, dataframe not to go to the Event		  
		    expCryInTower++; 		  continue;
		    
		  }
          
	      }//end of in case of non zero suppression
	    else
	      {
		LogInfo("EcalRawToDigi") << "@SUB=EcalDCCDaqFormatter::interpretRawData "
					 << " this unpacker does not support 0suppressed data";
	      }
	        
	    // data  to be stored in EBDataFrame, identified by EBDetId
	    int  ic        = cryIc(tower, strip,  ch) ;
	    EBDetId  id(SMid, ic,1);                 
	    
	    EBDataFrame theFrame ( id );
	    vector<int> xtalDataSamples = (*itXtalBlock)->xtalDataSamples();   
	    theFrame.setSize(xtalDataSamples.size());
      
      



	    // gain cannot be 0, checking for that
	    bool        gainIsOk =true;
	    unsigned gain_mask      = 12288;    //12th and 13th bit
	    vector <int> xtalGain;
      
	    for (unsigned short i=0; i<xtalDataSamples.size(); ++i ) {
	      theFrame.setSample (i, xtalDataSamples[i] );
	      if((xtalDataSamples[i] & gain_mask) == 0)
		{gainIsOk =false;}
	      xtalGain.push_back(0);
	      xtalGain[i] |= (xtalDataSamples[i] >> 12);
	    }
      
	    if (! gainIsOk) {
	      
	      LogWarning("EcalRawToDigiGainZero") << "@SUB=EcalDCCDaqFormatter::interpretRawData"
					  << " gain==0 for strip: "  << strip
					  << "\t channel: " << ch
					  << "\t in TT: " << _ExpectedTowers[_expTowersIndex]
					  << "\t ic: " << ic
					  << "\t at LV1: " << (*itEventBlock)->getDataField("LV1");
	      
	      // report on gain==0
	      gaincollection.push_back(id);

	      // there has been a gain==0, dataframe not to go to the Event
	      expCryInTower++; 	      continue;
	    }





	    // looking for forbidden gain transitions
	    
	    short firstGainWrong=-1;
	    short numGainWrong=0;
	    for (unsigned short i=0; i<xtalGain.size(); i++ ) {
	      
	      if (i>0 && xtalGain[i-1]>xtalGain[i]) {
		numGainWrong++;
		
		if (firstGainWrong == -1) {
		  firstGainWrong=i;
		  LogWarning("EcalRawToDigiGainSwitch") << "@SUB=EcalDCCDaqFormatter::interpretRawData"
							<< "channelHasGainSwitchProblem: crystal eta = " 
							<< id.ieta() << " phi = " << id.iphi();
		}
		
		LogWarning("EcalRawToDigiGainSwitch") << "@SUB=EcalDCCDaqFormatter::interpretRawData"
						      << "channelHasGainSwitchProblem: sample = " << (i-1) 
						      << " gain: " << xtalGain[i-1] << " sample: "
						      << i << " gain: " << xtalGain[i];
	      }
	    }


	    // only discriminating if gain stays the same after the forbidden gain transition
	    bool wrongGainStaysTheSame=false;
	    if (firstGainWrong!=-1 && firstGainWrong<9){
	      short gainWrong = xtalGain[firstGainWrong];
    
	      // does wrong gain stay the same after the forbidden transition?
	      for (unsigned short u=firstGainWrong+1; u<xtalGain.size(); u++){

		if( gainWrong == xtalGain[u]) 
		  wrongGainStaysTheSame=true; 
		else
		  wrongGainStaysTheSame=false; 

	      }// END loop on samples after forbidden transition
            
	    }// if firstGainWrong!=0 && firstGainWrong<8

	    if (numGainWrong>0) {
	      gainswitchcollection.push_back(id);

	      if (numGainWrong == 1 && (wrongGainStaysTheSame)) {
              
		LogWarning("EcalRawToDigiGainSwitch") << "@SUB=EcalDCCDaqFormatter:interpretRawData"
						      << "channelHasGainSwitchProblem: wrong transition stays till last sample";
              
	      }
	      else if (numGainWrong>1) {
		
		LogWarning("EcalRawToDigiGainSwitch") << "@SUB=EcalDCCDaqFormatter:interpretRawData"
						      << "channelHasGainSwitchProblem: more than 1 wrong transition";
              
		for (unsigned short i1=0; i1<xtalDataSamples.size(); ++i1 ) {
		  int countADC = 0x00000FFF;
		  countADC &= xtalDataSamples[i1];
		  LogWarning("EcalRawToDigi") << "Sample " << i1 << " ADC " << countADC << " Gain " << xtalGain[i1];
		}
	      }// end 'if there is multiple transition'

	      // there has been a forbidden gain transition,  dataframe not to go to the Event
	      expCryInTower++;  	    continue;

	    }// END of:   'if there is a forbidden gain transition'

	    // This unpacker does not handle 0suppressed data, as of now
	    // if (!theDCCheader.getZeroSuppression()) 	      expCryInTower++; 
	    expCryInTower++; 

	    // here (already continued if gain==0 or if forbidden-gain-switch),
	    // data frame needs go to the Event
	    digicollection.push_back(theFrame);

	  }// end loop on crystals within a tower block
	  
	  
	  _expTowersIndex++;
	}// end: tt1 ... tt68, crystal data
      


      
      
      /******************************************************************
       //    tt 69 and 70:  two mem boxes, holding PN0 ... PN9
       ******************************************************************/	
      else if (       (*itTowerBlock)->towerID() == 69 
                      ||	   (*itTowerBlock)->towerID() == 70       )	
	{
	  
	  LogDebug("EcalRawToDigi") << "@SUB=EcalDCCDaqFormatter::interpretRawData"
				    << "processing mem box num: " << (*itTowerBlock)->towerID();

	  // if tt 69 or 70 found, allocate Pn digi collection
	  if(! pnAllocated) 
	    {
	      //	      pndigicollection.reserve(kPns);
	      pnAllocated = true;
	    }

	  // using SMid=slot-in-CMS, not fed
	  DecodeMEM( SMid , (*itTowerBlock),  pndigicollection ,
		     memttidcollection,  memblocksizecollection,
		     memgaincollection,  memchidcollection);
	  
	}// end of < if it is a mem box>
      
      
    


      // wrong tt id
      else  {
        LogWarning("EcalRawToDigiTowerId") <<"@SUB=EcalDCCDaqFormatter::interpretRawData"
				      << " processing tt with ID not existing ( "
				      <<  (*itTowerBlock)->towerID() << ")";
        ++ _expTowersIndex;continue; 
      }// end: tt id error

    }// end loop on trigger towers
      
  }// end loop on events
}








void EcalDCCDaqFormatter::DecodeMEM( int SMid, DCCTowerBlock *  towerblock,  EcalPnDiodeDigiCollection & pndigicollection ,
				     EcalElectronicsIdCollection & memttidcollection,  EcalElectronicsIdCollection &  memblocksizecollection,
				     EcalElectronicsIdCollection & memgaincollection,  EcalElectronicsIdCollection & memchidcollection)
{
  
  LogDebug("EcalRawToDigi") << "@SUB=EcalDCCDaqFormatter::DecodeMEM"
			    << "in mem " << towerblock->towerID();  
  
  int  tower_id = towerblock ->towerID() ;
  int  mem_id   = tower_id-69;

  // initializing container
  for (int st_id=0; st_id< kStripsPerTower; st_id++){
    for (int ch_id=0; ch_id<kChannelsPerStrip; ch_id++){
      for (int sa=0; sa<11; sa++){      
	memRawSample_[st_id][ch_id][sa] = -1;}    } }

  
  // check that tower block id corresponds to mem boxes
  if(tower_id != 69 && tower_id != 70) 
    {
      LogWarning("EcalRawToDigiTowerId") << "@SUB=EcalDCCDaqFormatter:decodeMem"
					 << "DecodeMEM: this is not a mem box tower (" << tower_id << ")";
      ++ _expTowersIndex;
      return;
    }


     
  /******************************************************************************
   // getting the raw hits from towerBlock while checking tt and ch data structure 
   ******************************************************************************/
  vector<DCCXtalBlock *> & dccXtalBlocks = towerblock->xtalBlocks();
  vector<DCCXtalBlock*>::iterator itXtal;

  // checking mem tower block fo size
  if (dccXtalBlocks.size() != kChannelsPerTower)
    {     
      LogDebug("EcalRawToDigiDccBlockSize") << "@SUB=EcalDCCDaqFormatter:decodeMem"
				  << " wrong dccBlock size, namely: "  << dccXtalBlocks.size() 
				  << ", for mem " << _ExpectedTowers[_expTowersIndex];

      // reporting mem-tt block size problem
      // chosing channel 1 as representative as a dummy...
      EcalElectronicsId id(SMid, (int)_ExpectedTowers[_expTowersIndex], 1, 1);
      memblocksizecollection.push_back(id);

      ++ _expTowersIndex;
      return;  // if mem tt block size not ok - do not build any Pn digis
    }
  

  // loop on channels of the mem block
  int  cryCounter = 0;   int  strip_id  = 0;   int  xtal_id   = 0;  

  for ( itXtal = dccXtalBlocks.begin(); itXtal < dccXtalBlocks.end(); itXtal++ ) {
    strip_id             = (*itXtal) ->getDataField("STRIP ID");
    xtal_id              = (*itXtal) ->getDataField("XTAL ID");
    int wished_strip_id  = cryCounter/ kStripsPerTower;
    int wished_ch_id     = cryCounter% kStripsPerTower;
    
    if( (wished_strip_id+1) != ((int)strip_id) ||
	(wished_ch_id+1) != ((int)xtal_id) )
      {
	
	LogDebug("EcalRawToDigiChId") << "@SUB=EcalDCCDaqFormatter:decodeMem"
				    << " in mem " <<  towerblock->towerID()
				    << ", expected:\t strip"
				    << (wished_strip_id+1)  << " cry " << (wished_ch_id+1) << "\tfound: "
				    << "  strip " <<  strip_id << "  cry " << xtal_id;
	
	// report on crystal with unexpected indices
	EcalElectronicsId id(SMid, (int)_ExpectedTowers[_expTowersIndex], wished_strip_id, wished_ch_id );
	memchidcollection.push_back(id);
      }
    
    
    // Accessing the 10 time samples per Xtal:
    memRawSample_[wished_strip_id][wished_ch_id][1] = (*itXtal)->getDataField("ADC#1");
    memRawSample_[wished_strip_id][wished_ch_id][2] = (*itXtal)->getDataField("ADC#2");
    memRawSample_[wished_strip_id][wished_ch_id][3] = (*itXtal)->getDataField("ADC#3");
    memRawSample_[wished_strip_id][wished_ch_id][4] = (*itXtal)->getDataField("ADC#4");
    memRawSample_[wished_strip_id][wished_ch_id][5] = (*itXtal)->getDataField("ADC#5");
    memRawSample_[wished_strip_id][wished_ch_id][6] = (*itXtal)->getDataField("ADC#6");
    memRawSample_[wished_strip_id][wished_ch_id][7] = (*itXtal)->getDataField("ADC#7");
    memRawSample_[wished_strip_id][wished_ch_id][8] = (*itXtal)->getDataField("ADC#8");
    memRawSample_[wished_strip_id][wished_ch_id][9] = (*itXtal)->getDataField("ADC#9");
    memRawSample_[wished_strip_id][wished_ch_id][10] = (*itXtal)->getDataField("ADC#10");
      
    cryCounter++;
  }// end loop on crystals of mem dccXtalBlock
  
  // tower accepted and digi read from all 25 channels.
  // Increase counter of expected towers before unpacking in the 5 PNs
  ++ _expTowersIndex;



  /************************************************************
   // unpacking and 'cooking' the raw numbers to get PN sample
   ************************************************************/
  int tempSample=0;
  int memStoreIndex=0;
  int ipn=0;
  for (memStoreIndex=0; memStoreIndex<500; memStoreIndex++)    {
    data_MEM[memStoreIndex]= -1;   }
  
  
  for(int strip=0; strip<kStripsPerTower; strip++) {// loop on strips
    for(int channel=0; channel<kChannelsPerStrip; channel++) {// loop on channels

      if(strip%2 == 0) 
	{ipn= mem_id*5+channel;}
      else 
	{ipn=mem_id*5+4-channel;}

      for(int sample=0;sample< kSamplesPerChannel ;sample++) {
	tempSample= memRawSample_[strip][channel][sample+1];

	int new_data=0;
	if(strip%2 == 1) {
	  // 1) if strip number is even, 14 bits are reversed in order
	  for(int ib=0;ib<14;ib++)
	    { 
	      new_data <<= 1;
	      new_data=new_data | (tempSample&1);
	      tempSample >>= 1;
	    }
	} else {
	  new_data=tempSample;
	}

	// 2) flip 11th bit for AD9052 still there on MEM !
	// 3) mask with 1 1111 1111 1111
	new_data = (new_data ^ 0x800) & 0x3fff;       // (new_data  XOR 1000 0000 0000) & 11 1111 1111 1111
	// new_data = (new_data ^ 0x800) & 0x1fff;    // (new_data  XOR 1000 0000 0000) & 1 1111 1111 1111

	// (Bit 12) == 1 -> Gain 16;    (Bit 12) == 0 -> Gain 1	
	// gain in mem can be 1 or 16 encoded resp. with 0 ir 1 in the 13th bit.
	// checking and reporting if there is any sample with gain==2,3
	short sampleGain = (new_data &0x3000)/4096;
	if (  sampleGain==2 || sampleGain==3) 
	  {
	    EcalElectronicsId id(SMid, (int)_ExpectedTowers[_expTowersIndex], strip, channel );
	    memgaincollection.push_back(id);
	    
	    LogWarning("EcalRawToDigiGainZero")  << "@SUB=EcalDCCDaqFormatter:decodeMem"
					   << "in mem " <<  towerblock->towerID()
					   << " :\t strip: "
					   << (strip +1)  << " cry: " << (channel+1) 
					   << " has 14th bit non zero! Gain results: "
					   << sampleGain << ".";
	    
	    continue;
	  }// end 'if gain is zero'

	memStoreIndex= ipn*50+strip*kSamplesPerChannel+sample;
	// storing in data_MEM also the gain bits
	data_MEM[memStoreIndex]= new_data & 0x3fff;

      }// loop on samples
    }// loop on strips
  }// loop on channels
  



  for (int pnId=0; pnId<kPnPerTowerBlock; pnId++) pnIsOkInBlock[pnId]=true;
  // if anything was wrong with mem_tt_id or mem_tt_size: you would have already exited
  // otherwise, if any problem with ch_gain or ch_id: must not produce digis for the pertaining Pn

  if (!      (memgaincollection.size()==0 && memchidcollection.size()==0)          )
    {
      for ( EcalElectronicsIdCollection::const_iterator idItr = memgaincollection.begin();
	    idItr != memgaincollection.end();
	    ++ idItr ) {
	int ch = (*idItr).channelId();
	ch = (ch-1)/5;
	pnIsOkInBlock [ch] = false;
      }

      for ( EcalElectronicsIdCollection::const_iterator idItr = memchidcollection.begin();
	    idItr != memchidcollection.end();
	    ++ idItr ) {
	int ch = (*idItr).channelId();
	ch = (ch-1)/5;
	pnIsOkInBlock [ch] = false;
      }

    }// end: if any ch_gain or ch_id problems exclude the Pn's from digi production




  // looping on PN's of current mem box
  for (int pnId = 1;  pnId <  (kPnPerTowerBlock+1); pnId++){

    // if present Pn has any of its 5 channels with problems, do not produce digi for it
    if (! pnIsOkInBlock [pnId-1] ) continue;

    // second argumenti is SMid, namely geometrical slot in CMS ECAL barrel (1.. 36)
    EcalPnDiodeDetId PnId(EcalBarrel, SMid, pnId +  kPnPerTowerBlock*mem_id);
    EcalPnDiodeDigi thePnDigi(PnId );

    thePnDigi.setSize(kSamplesPerPn);

    //     for (int sample =0; sample<kSamplesPerPn; sample++)
    //       {thePnDigi.setSample(sample, data_MEM[(mem_id)*250 + (pnId-1)*kSamplesPerPn + sample ] );  }
    //     pndigicollection.push_back(thePnDigi);

    for (int sample =0; sample<kSamplesPerPn; sample++)
      {
	//	int adc  = (data_MEM[(mem_id)*250 + (pnId-1)*kSamplesPerPn + sample ] & 0xfff);
	//	int gain = (data_MEM[(mem_id)*250 + (pnId-1)*kSamplesPerPn + sample ] & 0x3000) /4096;;
	//	EcalFEMSample thePnSample(adc, gain);
	EcalFEMSample thePnSample( data_MEM[(mem_id)*250 + (pnId-1)*kSamplesPerPn + sample ] );
	thePnDigi.setSample(sample,  thePnSample );  
      }
    pndigicollection.push_back(thePnDigi);
  }
 
  
}

pair<int,int>  EcalDCCDaqFormatter::cellIndex(int tower_id, int strip, int ch) {
  
  int xtal= (strip-1)*5+ch-1;
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

  return ind;

}



int  EcalDCCDaqFormatter::cryIc(int tower, int strip, int ch) {

  if ( strip < 1 || 5<strip || ch <1 || 5 < ch || 68<tower)
    {
      edm::LogWarning("EcalRawToDigiChId") << "EcalDCCDaqFormatter::interpretRawData (cryIc) "
					     << " wrong channel id, since out of range: "
					     << "\t strip: "  << strip  << "\t channel: " << ch
					     << "\t in TT: " << tower;
      return -1;
    }

  pair<int,int> cellInd= EcalDCCDaqFormatter::cellIndex(tower, strip, ch); 
  return cellInd.second + (cellInd.first-1)*kCrystalsInPhi;
}



bool EcalDCCDaqFormatter::rightTower(int tower) const {
  
  if ((tower>12 && tower<21) || (tower>28 && tower<37) ||
      (tower>44 && tower<53) || (tower>60 && tower<69))
    return true;
  else
    return false;
}



bool EcalDCCDaqFormatter::leftTower(int tower) const
{
  return !rightTower(tower);
}


