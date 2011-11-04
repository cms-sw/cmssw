/*
 * L1TdeCSCTF.cc v1.0
 * written by J. Gartner
 */
 
#include "DQM/L1TMonitor/interface/L1TdeCSCTF.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h>

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <iomanip>
#include <memory>


using namespace std;
using namespace edm;

L1TdeCSCTF::L1TdeCSCTF(edm::ParameterSet const& pset):edm::EDAnalyzer(){
	dataTrackProducer = pset.getParameter<edm::InputTag>("dataTrackProducer");
	emulTrackProducer = pset.getParameter<edm::InputTag>("emulTrackProducer");
	lctProducer       = pset.getParameter<edm::InputTag>("lctProducer");
	
	m_dirName         = pset.getUntrackedParameter("DQMFolder", 
                             std::string("L1TEMU/CSCTFexpert"));

	ts=0;
	eventNum = 0;
	
	ptLUTset = pset.getParameter<edm::ParameterSet>("PTLUT");
	
	dbe = NULL;
	if(pset.getUntrackedParameter<bool>("DQMStore", false) )
	{
		dbe = Service<DQMStore>().operator->();
		dbe->setVerbose(0);
		dbe->setCurrentFolder(m_dirName);
	}
	
	outFile = pset.getUntrackedParameter<string>("outFile", "");
	if( outFile.size() != 0 )
	{
	  edm::LogWarning("L1TdeCSCTF")
	    << "L1T Monitoring histograms will be saved to " 
	    << outFile.c_str() 
	    << endl;
	}
	
	bool disable = pset. getUntrackedParameter<bool>("disableROOToutput", false);
	if(disable){
		outFile="";
	}
	
	bzero(srLUTs_, sizeof(srLUTs_));
	//int endcap =1, sector =1;
	bool TMB07=true;
	edm::ParameterSet srLUTset;
	srLUTset.addUntrackedParameter<bool>("ReadLUTs", false);
  	srLUTset.addUntrackedParameter<bool>("Binary",   false);
  	srLUTset.addUntrackedParameter<std::string>("LUTPath", "./");
  	for(int endcapItr = CSCDetId::minEndcapId(); endcapItr <= CSCDetId::maxEndcapId(); endcapItr++)
  	{
  		for(int sectorItr = CSCTriggerNumbering::minTriggerSectorId();sectorItr <= CSCTriggerNumbering::maxTriggerSectorId();sectorItr++)
    	{
	  		for(int stationItr = 1; stationItr <= 4; stationItr++)
      		{
	    		if(stationItr == 1)
        		{
					for(int subsectorItr = 0; subsectorItr < 2; subsectorItr++)
   					{
		      			srLUTs_[endcapItr-1][sectorItr-1][subsectorItr] = new CSCSectorReceiverLUT(endcapItr, sectorItr, subsectorItr+1, stationItr, srLUTset, TMB07); 
        			}
        		} else {
		  			srLUTs_[endcapItr-1][sectorItr-1][stationItr] = new CSCSectorReceiverLUT(endcapItr, sectorItr, 0, stationItr, srLUTset, TMB07); 
       			} //if for station 1 or 234
      		} // stationItr loop
   		} // sectorItr loop
  	} // endcapItr loop
}

void L1TdeCSCTF::beginJob()
{


	/////////////////////////////
	// DQM Directory Structure //
	/////////////////////////////
	DQMStore * dbe = 0;
	dbe = Service<DQMStore>().operator->();
	if( dbe ){
		dbe->setCurrentFolder(m_dirName);
		/////////////////////////////
		// Define Monitor Elements //
		/////////////////////////////

		
		pt1Comp = dbe->book2D("pt1Comp","Hardware Vs. Emulator #Delta #phi_{12}",256,0,256,256,0,256);
		pt1Comp->setAxisTitle("Hardware #Delta #phi_{12}",1);
		pt1Comp->setAxisTitle("Emulator #Delta #phi_{12}",2);
		pt2Comp = dbe->book2D("pt2Comp","Hardware Vs. Emulator #Delta #phi_{23}",16,0,16,16,0,16);
		pt2Comp->setAxisTitle("Hardware #Delta #phi_{23}",1);
		pt2Comp->setAxisTitle("Emulator #Delta #phi_{23}",2);
		pt3Comp = dbe->book2D("pt3Comp","Hardware Vs. Emulator #eta",16,0,16,16,0,16);
		pt3Comp->setAxisTitle("Hardware #eta",1);
		pt3Comp->setAxisTitle("Emulator #eta",2);
		pt4Comp = dbe->book2D("pt4Comp","Hardware Vs. Emulator Mode",16,0,16,16,0,16);
		pt4Comp->setAxisTitle("Hardware Mode",1);
		pt4Comp->setAxisTitle("Emulator Mode",2);
		pt5Comp = dbe->book2D("pt5Comp","Hardware Vs. Emulator Sign",2,0,2,2,0,2);
		pt5Comp->setAxisTitle("Hardware Sign",1);
		pt5Comp->setAxisTitle("Emulator Sign",2);
		pt6Comp = dbe->book2D("pt6Comp","Hardware Vs. Emulator FR bit",2,0,2,2,0,2);
		pt6Comp->setAxisTitle("Hardware FR bit",1);
		pt6Comp->setAxisTitle("Emulator FR bit",2);
		
		badBitMode1 = dbe->book1D("badBitMode1","P_{t} LUT address bit differences, emulator mode 1", 21,0,21);
		badBitMode1->setAxisTitle("P_{t} Address Bit",1);
		badBitMode2 = dbe->book1D("badBitMode2","P_{t} LUT address bit differences, emulator mode 2", 21,0,21);
		badBitMode2->setAxisTitle("P_{t} Address Bit",1);
		badBitMode3 = dbe->book1D("badBitMode3","P_{t} LUT address bit differences, emulator mode 3", 21,0,21);
		badBitMode3->setAxisTitle("P_{t} Address Bit",1);
		badBitMode4 = dbe->book1D("badBitMode4","P_{t} LUT address bit differences, emulator mode 4", 21,0,21);
		badBitMode4->setAxisTitle("P_{t} Address Bit",1);
		badBitMode5 = dbe->book1D("badBitMode5","P_{t} LUT address bit differences, emulator mode 5", 21,0,21);
		badBitMode5->setAxisTitle("P_{t} Address Bit",1);
		badBitMode6 = dbe->book1D("badBitMode6","P_{t} LUT address bit differences, emulator mode 6", 21,0,21);
		badBitMode6->setAxisTitle("P_{t} Address Bit",1);
		badBitMode7 = dbe->book1D("badBitMode7","P_{t} LUT address bit differences, emulator mode 7", 21,0,21);
		badBitMode7->setAxisTitle("P_{t} Address Bit",1);
		badBitMode8 = dbe->book1D("badBitMode8","P_{t} LUT address bit differences, emulator mode 8", 21,0,21);
		badBitMode8->setAxisTitle("P_{t} Address Bit",1);
		badBitMode9 = dbe->book1D("badBitMode9","P_{t} LUT address bit differences, emulator mode 9", 21,0,21);
		badBitMode9->setAxisTitle("P_{t} Address Bit",1);
		badBitMode10 = dbe->book1D("badBitMode10","P_{t} LUT address bit differences, emulator mode 10", 21,0,21);
		badBitMode10->setAxisTitle("P_{t} Address Bit",1);
		badBitMode11 = dbe->book1D("badBitMode11","P_{t} LUT address bit differences, emulator mode 11", 21,0,21);
		badBitMode11->setAxisTitle("P_{t} Address Bit",1);
		badBitMode12 = dbe->book1D("badBitMode12","P_{t} LUT address bit differences, emulator mode 12", 21,0,21);
		badBitMode12->setAxisTitle("P_{t} Address Bit",1);
		badBitMode13 = dbe->book1D("badBitMode13","P_{t} LUT address bit differences, emulator mode 13", 21,0,21);
		badBitMode13->setAxisTitle("P_{t} Address Bit",1);
		badBitMode14 = dbe->book1D("badBitMode14","P_{t} LUT address bit differences, emulator mode 14", 21,0,21);
		badBitMode14->setAxisTitle("P_{t} Address Bit",1);
		badBitMode15 = dbe->book1D("badBitMode15","P_{t} LUT address bit differences, emulator mode 15", 21,0,21);
		badBitMode15->setAxisTitle("P_{t} Address Bit",1);
		
		trackCountComp = dbe->book2D("trackCountComp","Hardware Vs. Emulator track Multiplicity",4,-0.5,3.5,4,-0.5,3.5);
		trackCountComp->setAxisTitle("Hardware track count",1);
		trackCountComp->setAxisTitle("Emulator track count",2);
		
		ptLUTOutput = dbe->book2D("ptLUTOutput","Comparison of P_{t}LUT Output for spy muon",127,0,127,127,0,127);
		ptLUTOutput->setAxisTitle("Hardware P_{t} LUT Out",1);
		ptLUTOutput->setAxisTitle("Emulator P_{t} LUT Out",2);
		
		mismatchSector = dbe->book1D("mismatchSector","LCT Sector for mismatched tracks",6,0,6);
		mismatchTime   = dbe->book1D("mismatchTime","LCT Time bin for mismatched tracks", 7,3,10);
		mismatchEndcap = dbe->book1D("mismatchEndcap","LCT endcap for mismatched tracks", 2,0,2);
		mismatchPhi		 = dbe->book1D("mismatchPhi","LCT #phi for mismatched tracks",4096,0,4096);
		mismatchEta		 = dbe->book1D("mismatchEta","LCT #eta for mismatched tracks",128,0,128);
		
		mismatchDelPhi12 = dbe->book1D("mismatchDelPhi12", "LCT #Delta #phi_{12}", 4096,0,4096);
		mismatchDelPhi13 = dbe->book1D("mismatchDelPhi13", "LCT #Delta #phi_{13}", 4096,0,4096);
		mismatchDelPhi14 = dbe->book1D("mismatchDelPhi14", "LCT #Delta #phi_{14}", 4096,0,4096);
		mismatchDelPhi23 = dbe->book1D("mismatchDelPhi23", "LCT #Delta #phi_{23}", 4096,0,4096);
		mismatchDelPhi24 = dbe->book1D("mismatchDelPhi24", "LCT #Delta #phi_{24}", 4096,0,4096);
		mismatchDelPhi34 = dbe->book1D("mismatchDelPhi34", "LCT #Delta #phi_{34}", 4096,0,4096);
		
		mismatchDelEta12 = dbe->book1D("mismatchDelEta12", "LCT #Delta #eta_{12}", 128,0,128);
		mismatchDelEta13 = dbe->book1D("mismatchDelEta13", "LCT #Delta #eta_{13}", 128,0,128);
		mismatchDelEta14 = dbe->book1D("mismatchDelEta14", "LCT #Delta #eta_{14}", 128,0,128);
		mismatchDelEta23 = dbe->book1D("mismatchDelEta23", "LCT #Delta #eta_{23}", 128,0,128);
		mismatchDelEta24 = dbe->book1D("mismatchDelEta24", "LCT #Delta #eta_{24}", 128,0,128);
		mismatchDelEta34 = dbe->book1D("mismatchDelEta34", "LCT #Delta #eta_{34}", 128,0,128);
		
		endTrackBadSector = dbe->book1D("endTrackBadSector", "Sector for PtLUT misalignment", 12,0,12);
		endTrackBadFR     = dbe->book1D("endTrackBadFR", "FR bit for PtLUT misalignment", 2, 0, 2);
		endTrackBadEta    = dbe->book1D("endTrackBadEta", "Eta for PtLUT misalignment", 16,0,16);
		endTrackBadMode   = dbe->book1D("endTrackBadMode", "Mode of track for PtLUT misalingment", 16,0,16);
		
		bxData = dbe->book1D("bxData", "Bunch Crossing Number for data track", 15, -5, 10);
		bxEmu = dbe->book1D("bxEmu", "Bunch Crossing Number for emu track", 15, -5, 10);
		
		allLctBx = dbe->book1D("allLctBx", "Bunch Crossing Number for all Lcts", 15,-3,12);
	}
	
}

void L1TdeCSCTF::endJob(void){
	
	if(ptLUT_) delete ptLUT_;
	
	if ( outFile.size() != 0  && dbe ) dbe->save(outFile);	
	return;
}

void L1TdeCSCTF::analyze(edm::Event const& e, edm::EventSetup const& es){
	// Get LCT information
	//////////////////////
	int lctArray[20][7];
	short nLCTs=0;
	for(int oj=0; oj<20; oj++) lctArray[oj][0]=0;
	if( lctProducer.label() != "null" )
	{
		edm::Handle<CSCCorrelatedLCTDigiCollection> LCTs;
		e.getByLabel(lctProducer.label(),lctProducer.instance(), LCTs);

		// check validity of input collection
		if(!LCTs.isValid()) {
		  edm::LogWarning("L1TdeCSCTF")
		    << "\n No valid [lct] product found: "
		    << " CSCCorrelatedLCTDigiCollection"
		    << std::endl;
		  return;
		}
		
		edm::ESHandle< L1MuTriggerScales > scales ;
		es.get< L1MuTriggerScalesRcd >().get( scales ) ;
		edm::ESHandle< L1MuTriggerPtScale > ptScale ;
		es.get< L1MuTriggerPtScaleRcd >().get( ptScale ) ;
		ptLUT_ = new CSCTFPtLUT(ptLUTset, scales.product(), ptScale.product() );

		for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc=LCTs.product()->begin(); csc!=LCTs.product()->end(); csc++)
		{
			int lctId=0;

			CSCCorrelatedLCTDigiCollection::Range range1 = LCTs.product()->get((*csc).first);
			for(CSCCorrelatedLCTDigiCollection::const_iterator lct=range1.first; lct!=range1.second; lct++,lctId++)
			{
				CSCCorrelatedLCTDigiCollection::Range range1 = LCTs.product()->get((*csc).first);
				CSCCorrelatedLCTDigiCollection::const_iterator lct;
				for( lct = range1.first; lct!=range1.second; lct++)
				{	
					int station = (*csc).first.station()-1;
					int cscId   = (*csc).first.triggerCscId()-1;
					int sector  = (*csc).first.triggerSector()-1;
					int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*csc).first);
					int tbin    = lct->getBX();
					int fpga    = ( subSector ? subSector-1 : station+1 );
					int endcap = (*csc).first.endcap()-1;

					lclphidat lclPhi;
					gblphidat gblPhi;
					gbletadat gblEta;

					try{
						lclPhi = srLUTs_[endcap][sector][fpga]->localPhi(lct->getStrip(), lct->getPattern(), lct->getQuality(), lct->getBend());
					} catch ( cms::Exception &e ) {
						bzero(&lclPhi, sizeof(lclPhi));
						edm::LogWarning("L1TdeCSCTF:analyze()") << "Exception from LocalPhi LUT in endCap: " << endcap << ", sector: " << sector << ", fpga: " << fpga 
							<< "(strip:" << lct->getStrip() << ", pattern:"<< lct->getPattern() << ", Q:" << lct->getQuality() << ", bend:" << lct->getBend() << std::endl;
					}

					try{
						gblPhi = srLUTs_[endcap][sector][fpga]->globalPhiME( lclPhi.phi_local, lct->getKeyWG(),cscId+1);
					} catch  ( cms::Exception &e ) {
						bzero(&gblPhi,sizeof(gblPhi));
						edm::LogWarning("L1TdeCSCTF:analyze()") << "Exception from GlobalPhi LUT in endCap: " << endcap << ", sector: " << sector << ", fpga: " << fpga 
							<< "(local phi:" << lclPhi.phi_local << ", keyWG:" << lct->getKeyWG() << ",cscID:" << cscId+1 << std::endl;
					}
					try{
						gblEta = srLUTs_[endcap][sector][fpga]->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local,lct->getKeyWG(),cscId+1);
					} catch  ( cms::Exception &e ) {
						bzero(&gblEta,sizeof(gblEta));
						edm::LogWarning("L1TdeCSCTF:analyze()") << "Exception from GlobalEta LUT in endCap: " << endcap << ", sector: " << sector << ", fpga: " << fpga
							<< "(local phi bend:" << lclPhi.phi_bend_local << ", local phi:" <<  lclPhi.phi_local << ", keyWG: " << lct->getKeyWG() << ", cscID: " << cscId+1 << std::endl;
					}

					allLctBx->Fill(tbin);
					
					if((nLCTs < 20))
					{
						lctArray[nLCTs][0] = 1;
						lctArray[nLCTs][1] = sector;
						lctArray[nLCTs][2] = tbin;
						lctArray[nLCTs][3] = endcap;
						lctArray[nLCTs][4] = gblPhi.global_phi;
						lctArray[nLCTs][5] = gblEta.global_eta;
						lctArray[nLCTs][6] = station;
						nLCTs++;
					}
				}
			}
		}
	}

	// Initialize Arrays
	////////////////////
	nDataMuons = 0; nEmulMuons = 0;
	int dataMuonArray[8][3], emuMuonArray[8][3];
	for( int joe=0; joe<8; joe++){
		for( int rules=0; rules<3; rules++ )
		{
			dataMuonArray[joe][rules] = 0;
			emuMuonArray[joe][rules] = 0;
		}
	}
	// Get Hardware information, and check output of PtLUT
	//////////////////////////////////////////////////////
	if( dataTrackProducer.label() != "null" )
	{
		edm::Handle<L1CSCTrackCollection> tracks;
		e.getByLabel(dataTrackProducer.label(),dataTrackProducer.instance(),tracks);

		// check validity of input collection
		if(!tracks.isValid()) {
		  edm::LogWarning("L1TdeCSCTF")
		    << "\n No valid [data tracks] product found: "
		    << " L1CSCTrackCollection"
		    << std::endl;
		  return;
		}


		for(L1CSCTrackCollection::const_iterator trk=tracks.product()->begin(); trk!=tracks.product()->end(); trk++)
		{
			bxData->Fill(trk->first.BX());
			if( /*(((0x0f0000 &(trk->first.ptLUTAddress()) ) >> 16) != 0xb) &&*/  (nDataMuons < 8) && (trk->first.BX() <2) && (trk->first.BX() > -2) )
			{
				dataMuonArray[nDataMuons][0]  = 0; //matched?
				dataMuonArray[nDataMuons][1]  = trk->first.ptLUTAddress();  
				dataMuonArray[nDataMuons][2]  = trk->first.sector();
				nDataMuons++;
			}
			/*
			if( trk->first.outputLink() == 1)
			{
				int frBit = (0x200000 &(trk->first.ptLUTAddress()) ) >> 21;
				int dataRank = trk->first.rank();
				ptadd thePtAdd(trk->first.ptLUTAddress());
				ptdat thePtDat = ptLUT_->Pt(thePtAdd);
				int emuRank = thePtDat.front_rank;
				if(frBit == 0) emuRank = thePtDat.rear_rank;
				ptLUTOutput->Fill(dataRank, emuRank);
				if(dataRank != emuRank)
				{
					endTrackBadSector->Fill(trk->first.sector());
					endTrackBadFR->Fill(frBit);
					int etaP = (0xf000 &(trk->first.ptLUTAddress()) ) >> 12;
					endTrackBadEta->Fill(etaP);
					int modeP = (0xf0000 &(trk->first.ptLUTAddress()) ) >> 16;
					endTrackBadMode->Fill(modeP);
				}
			}*/
		}
	}
	// Get Emulator information
	///////////////////////////
	if( emulTrackProducer.label() != "null" )
	{
		edm::Handle<L1CSCTrackCollection> tracks;
		e.getByLabel(emulTrackProducer.label(),emulTrackProducer.instance(),tracks);

		// check validity of input collection
		if(!tracks.isValid()) {
		  edm::LogWarning("L1TdeCSCTF")
		    << "\n No valid [emulator tracks] product found: "
		    << " L1CSCTrackCollection"
		    << std::endl;
		  return;
		}

		for(L1CSCTrackCollection::const_iterator trk=tracks.product()->begin(); trk!=tracks.product()->end(); trk++)
		{
			bxEmu->Fill(trk->first.BX());
			if( /*(((0x0f0000&trk->first.ptLUTAddress())>>16) != 0xb) &&*/  (nEmulMuons<8) && (trk->first.BX() <2) && (trk->first.BX() >-2))
			{
				emuMuonArray[nEmulMuons][0]  = 0;
				emuMuonArray[nEmulMuons][1]  = trk->first.ptLUTAddress();
				emuMuonArray[nEmulMuons][2]  = trk->first.sector();
				nEmulMuons++;
			}
		}
	}
	// Match Tracks by sector & mode in the case of multiple tracks
	///////////////////////////////////////////////////////////////
	short int rightIndex[8];
	for(int jjj=0; jjj<8; jjj++) rightIndex[jjj]= 20;
	trackCountComp->Fill(nDataMuons,nEmulMuons);
	if( (nDataMuons==nEmulMuons) && (nDataMuons!=0) )
	{
		for(int hw=0; hw<nDataMuons; hw++)
		{
			int addDiffWin = 5;
			for(int sw=0; sw<nEmulMuons; sw++)
			{
				if( emuMuonArray[sw][2] == dataMuonArray[hw][2] ) //make sure triggers are coming from same sector
				{
					int addDiff = 0;
					int hwMode = (0x0f0000 & dataMuonArray[hw][1] )>>16;
					int swMode = (0x0f0000 & emuMuonArray[sw][1] )>>16;
					int thor = hwMode^swMode;
					for(int jax=0; jax<4; jax++)
					{
						addDiff += 1 & (thor>>jax);
					}
					if( (addDiff <addDiffWin) && (emuMuonArray[sw][0]!=1) )
					{
						addDiffWin = addDiff;
						rightIndex[hw] = sw;
					}
				}
				
				if( sw==nEmulMuons )
				{
					emuMuonArray[rightIndex[hw]][0]=1;
				}
			}
		}
		
		for( int i=0; i<nEmulMuons; i++)
		{
			int hwModeM = (0x0f0000 & dataMuonArray[i][1] )>>16;
			int swModeM = (0x0f0000 & emuMuonArray[rightIndex[i]][1] )>>16;
			int thorM = hwModeM^swModeM;
			for( int q=0; q<=22; q++)
			{
				int addDiff = (1<<q)&thorM;
				if(addDiff != 0) 
				{	
					if(hwModeM == 0x1) badBitMode1->Fill(q);
					if(hwModeM == 0x2) badBitMode2->Fill(q);
					if(hwModeM == 0x3) badBitMode3->Fill(q);
					if(hwModeM == 0x4) badBitMode4->Fill(q);
					if(hwModeM == 0x5) badBitMode5->Fill(q);
					if(hwModeM == 0x6) badBitMode6->Fill(q);
					if(hwModeM == 0x7) badBitMode7->Fill(q);
					if(hwModeM == 0x8) badBitMode8->Fill(q);
					if(hwModeM == 0x9) badBitMode9->Fill(q);
					if(hwModeM == 0xa) badBitMode10->Fill(q);
					if(hwModeM == 0xb) badBitMode11->Fill(q);
					if(hwModeM == 0xc) badBitMode12->Fill(q);
					if(hwModeM == 0xd) badBitMode13->Fill(q);
					if(hwModeM == 0xe) badBitMode14->Fill(q);
					if(hwModeM == 0xf) badBitMode15->Fill(q);
				}
			}
			
			pt4Comp->Fill(hwModeM,swModeM);
			if( hwModeM == swModeM)
			{
				int hwPhi1 = (0x0000ff & dataMuonArray[i][1]);
				int swPhi1 = (0x0000ff & emuMuonArray[rightIndex[i]][1]);
				int hwPhi2 = (0x000f00 & dataMuonArray[i][1])>>8;
				int swPhi2 = (0x000f00 & emuMuonArray[rightIndex[i]][1])>>8;
				int hwEta  = (0x00f000 & dataMuonArray[i][1])>>12;
				int swEta  = (0x00f000 & emuMuonArray[rightIndex[i]][1])>>12;
				int hwSign = (0x100000 & dataMuonArray[i][1])>>20;
				int swSign = (0x100000 & emuMuonArray[rightIndex[i]][1])>>20;
				int hwFr   = (0x200000 & dataMuonArray[i][1])>>21;
				int swFr   = (0x200000 & emuMuonArray[rightIndex[i]][1])>>21;
				pt1Comp->Fill(hwPhi1,swPhi1);
				pt2Comp->Fill(hwPhi2,swPhi2);
				pt3Comp->Fill(hwEta,swEta);
				pt5Comp->Fill(hwSign,swSign);
				pt6Comp->Fill(hwFr,swFr);
			} else {
				for(int ak=0; ak<=nLCTs; ak++)
				{
					if(lctArray[ak][1] == dataMuonArray[i][2])
					{
						mismatchSector->Fill(lctArray[ak][1]);
						mismatchTime  ->Fill(lctArray[ak][2]);
						mismatchEndcap->Fill(lctArray[ak][3]);
						mismatchPhi		->Fill(lctArray[ak][4]);
						mismatchEta		->Fill(lctArray[ak][5]);
					}
					for(int akk=ak+1; akk<=nLCTs; akk++)
					{
						if(lctArray[ak][1] == lctArray[akk][1])
						{
							int delPhi = abs(lctArray[ak][4] - lctArray[akk][4]);
							int delEta = abs(lctArray[ak][5] - lctArray[akk][5]);
							int lowSta = (lctArray[ak][1] < lctArray[akk][1]) ? lctArray[ak][1] : lctArray[akk][1];
							int hiSta = (lctArray[ak][1] > lctArray[akk][1]) ? lctArray[ak][1] : lctArray[akk][1];
							switch(lowSta)
							{
								case 0:
									switch(hiSta)
									{
										case 1:
											mismatchDelPhi12->Fill(delPhi);
											mismatchDelEta12->Fill(delEta);
										case 2:
											mismatchDelPhi13->Fill(delPhi);
											mismatchDelEta13->Fill(delEta);
										case 3:
											mismatchDelPhi14->Fill(delPhi);
											mismatchDelEta14->Fill(delEta);
										break;
									}
								case 1:
									switch(hiSta)
									{
										case 2:
											mismatchDelPhi23->Fill(delPhi);
											mismatchDelEta23->Fill(delEta);
										case 3:
											mismatchDelPhi24->Fill(delPhi);
											mismatchDelEta24->Fill(delEta);
										break;
									}
								case 2:
									if(hiSta ==3)
									{
											mismatchDelPhi34->Fill(delPhi);
											mismatchDelEta34->Fill(delEta);
									}
								break;
							}
						}
					}
				}
			}
		}
	}
}		
