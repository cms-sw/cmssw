/*
 *  CsctfDatEmu v 1.0
 *  Written by J. Gartner
 *
 *	v1.0 Basic Comparison Plots, drop DQM framework
 */
 
#include "L1Trigger/CSCTrackFinder/test/analysis/CsctfDatEmu.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"

#include <TCanvas.h>
#include <iostream>
#include <iomanip>
#include <memory>

using namespace std;
using namespace edm;

CsctfDatEmu::CsctfDatEmu(ParameterSet const& pset){
	dataTrackProducer = pset.getParameter<edm::InputTag>("dataTrackProducer");
	emulTrackProducer = pset.getParameter<edm::InputTag>("emulTrackProducer");
	lctProducer       = pset.getParameter<edm::InputTag>("lctProducer");
  
	outFile = pset.getUntrackedParameter<std::string>("outFile");//c
	dtCount  = 0;
	cscCount = 0;
	dcCount  = 0;
	
	bzero(srLUTs_,sizeof(srLUTs_));
  int endcap=1, sector=1; // assume SR LUTs are all same for every sector in either of endcaps
  bool TMB07=true; // specific TMB firmware
  // Create a dumy pset for SR LUTs
  edm::ParameterSet srLUTset;
  srLUTset.addUntrackedParameter<bool>("ReadLUTs", false);
  srLUTset.addUntrackedParameter<bool>("Binary",   false);
  srLUTset.addUntrackedParameter<std::string>("LUTPath", "./");
  for(int station=1,fpga=0; station<=4 && fpga<5; station++)
  {
  	if(station==1)
		for(int subSector=0; subSector<2 && fpga<5; subSector++)
			srLUTs_[fpga++] = new CSCSectorReceiverLUT(endcap, sector, subSector+1, station, srLUTset, TMB07);
    else
		srLUTs_[fpga++] = new CSCSectorReceiverLUT(endcap, sector, 0, station, srLUTset, TMB07);
	}

	my_dtrc = new CSCTFDTReceiver();
	//edm::ParameterSet ptLUTset = pset.getParameter<edm::ParameterSet>("PTLUT");
  //ptLUT_ = new CSCTFPtLUT(ptLUTset, scales, ptScale);
}

void CsctfDatEmu::beginJob()
{
	modeComp   = new TH2F("modeComp",  "Hardware Vs. Emulator Mode", 16,0,16, 16,0,16);
	phi12Comp  = new TH2F("phi12Comp", "Hardware Vs. Emulator #phi_{12}", 256,0,256, 256,0,256);
	phi23Comp  = new TH2F("phi23Comp", "Hardware Vs. Emulator #phi_{23}", 16,0,16, 16,0,16);
	etaComp 	 = new TH2F("etaComp",   "Hardware Vs. Emulator #eta", 16,0,16, 16,0,16);
	signFrComp = new TH2F("signFrComp","Hardware Vs. Emulator sign/Fr", 4,0,4, 4,0,4);
	bxComp     = new TH2F("bxComp",    "Hardware Vs. Emulator Track BX",7,-3,4,7,-3,4);
	rankComp   = new TH2F("rankComp",  "Hardware Vs. Emulator Rank",127,0,127,127,0,127);
	qualComp   = new TH2F("qualComp",  "Hardware Vs. Emulator Quality",4,0,4,4,0,4);
	ptComp		 = new TH2F("ptComp",    "Hardware Vs. Emulator Pt", 32,0,32,32,0,32);
	phiVComp	= new TH2F("phiVComp", "Hardware Vs. Emulator Track #phi",32,0,32,32,0,32);
	etaVComp	= new TH2F("etaVComp", "Hardware Vs. Emulator Track #eta",32,0,32,32,0,32);
	
	numTrkComp = new TH2F("numTrkComp","Hardware Vs. Emulator Track Occupancy", 5,0,5, 5,0,5);
	badSectorEnd = new TH2F("badSectorEnd","Endcap Vs. Sector for mismatched tracks",6,1,7,2,1,3);
	
	badPhiMode = new TH1F("badPhiMode","Mode_{S.W.} for mismatched phi",16,0,16);
	badPhiEta  = new TH1F("badPhiEta", "#eta_{S.W.} value for mismatched #phi",16,0,16);
	badEtaMode = new TH1F("badEtaMode",  "Mode_{S.W.} for mismatched #eta",16,0,16);
	badEtaPhi  = new TH1F("badEtaPhi",   "#phi_{S.W.} value for mismatched #eta",256,0,256);
	badModePhi = new TH1F("badModePhi", "#phi_{S.W.} for mismatched mode", 256,0,256);
	badModeEta = new TH1F("badModeEta", "#eta_{S.W.} for mismatched mode", 16, 0, 16);
	badBxMode  = new TH1F("badBxMode", "Mode for mismatched B.X.", 16, 0 ,16);
	badRankMode= new TH1F("badRankMode","Mode for mismatched Rank", 16,0,16);
	badRankEta = new TH1F("badRankEta", "#eta for mismatched Rank", 16,0,16);
	badRankOcc = new TH1F("badRankOcc", "Track Occupancy for mismatched Rank", 10,0,10);
	
	moreDatModeE = new TH1F("moreDatMode","Mode for emulator tracks when N_{H.W.}>N_{S.W.}",16,0,16);
	moreEmuModeE = new TH1F("moreEmuMode","Mode for emulator tracks when N_{S.W.}>N_{H.W.}",16,0,16);
	moreDatModeD = new TH1F("moreDatMode","Mode for data tracks when N_{H.W.}>N_{S.W.}",16,0,16);
	moreEmuModeD = new TH1F("moreEmuMode","Mode for data tracks when N_{S.W.}>N_{H.W.}",16,0,16);
	
	dtBBx    = new TH1F("dtBBx","DT stub BX", 7, 3, 10);
	dtBBx0   = new TH1F("dtBBx0","DT stub BX0", 4,0,4);
	dtBMlink = new TH1F("dtBMlink", "DT stub MPC link", 4,0,4);
	dtBQ     = new TH1F("dtBQ", "DT stub Quality",10,0,10);
	dtBBend  = new TH1F("dtBBend", "DT stub Bend",32,0,32);
	dtBStrip = new TH1F("dtBStrip", "DT strip", 3,0,3);
	
	badRankLink = new TH1F("badRankLink","Bad Rank Link",4,0,4);
	
	stubDtPhi = new TH2F("stubDtPhi","Data vs Emulator Dt stub #phi", 200,400,2400,200,400,2400);
	stubDtStrip = new TH2F("stubDtStrip","Data vs Emulator Dt stub Strip", 3,0,3,3,0,3);
	stubDtBx = new TH2F("stubDtBx","Data vs Emulator Dt stub Bx",7, 3, 10,7, 3, 10);
	stubDtBX0 = new TH2F("stubDtBX0","Data vs Emulator Dt stub Bx0",4,0,4,4,0,4);
	stubDtMpc = new TH2F("stubDtMpc","Data vs Emulator Dt stub MPC Link",4,0,4,4,0,4);
	stubDtQual = new TH2F("stubDtQual","Data vs Emulator Dt stub Quality",10,0,10,10,0,10);
	stubDtBend = new TH2F("stubDtBend","Data vs Emulator Dt stub Bend", 32,0,32,32,0,32);
	stubDtoTrack = new TH1F("stubDtoTrack","#Delta #phi from DT recorded stub to track",5000,-50,50);
	stubCtoTrack = new TH1F("stubCtoTrack","#Delta #phi from CSC recorded stub to track",5000,-50,50);
	badRankSecEnd = new TH2F("badRankSecEnd","Bad Sector, Endcap",6,1,7,2,1,3);
	stubBadPhiSec = new TH2F("stubBadPhiSec","Bad Sector, Endcap for mismatched dt stub",12,1,13,2,1,3);
	stubEmuBx = new TH1F("stubEmuBx","BX number of emulator stubs", 10,0,10);
	stubDatBx = new TH1F("stubDatBx","BX number of data stubs",10,0,10);
	stubDifBx = new TH1F("stubDifBx","BX dat - emu plots",10,0,10);
}

void CsctfDatEmu::analyze(Event const& e, EventSetup const& es)
{
	//ptLUT_ = new CSCTFPtLUT(es);
	bool cscThis = false;
	edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts;
	e.getByLabel(lctProducer.label(),lctProducer.instance(),corrlcts);
	for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc=corrlcts.product()->begin(); csc!=corrlcts.product()->end(); csc++)
	{
	  CSCCorrelatedLCTDigiCollection::Range range1 = corrlcts.product()->get((*csc).first);
	  for(CSCCorrelatedLCTDigiCollection::const_iterator lct=range1.first; lct!=range1.second; lct++)
	  {
				cscThis = true;
	      //int endcap  = (*csc).first.endcap()-1;
	      int station = (*csc).first.station()-1;
	      //int sector  = (*csc).first.triggerSector()-1;
	      int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*csc).first);
	      int cscId   = (*csc).first.triggerCscId()-1;
	      int fpga    = ( subSector ? subSector-1 : station+1 );
				lclphidat lclPhi;
				try{
					lclPhi = srLUTs_[fpga]->localPhi(lct->getStrip(), lct->getPattern(), lct->getQuality(), lct->getBend());
				} catch(...){
					bzero(&lclPhi,sizeof(lclPhi));
				}
				
	      gblphidat gblPhi;
				try{
					gblPhi = srLUTs_[fpga]->globalPhiME(lclPhi.phi_local, lct->getKeyWG(), cscId+1);
				} catch(...){
					bzero(&gblPhi,sizeof(gblPhi));
				}
				
	      gbletadat gblEta;
				try{
					gblEta = srLUTs_[fpga]->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, lct->getKeyWG(), cscId+1);
				} catch(...){
					bzero(&gblEta,sizeof(gblEta));
				}
				//std::cout << "LCT endcap, station, sector, sub, cscId, phi, eta: " << endcap << ", " << station << ", " << sector << ", " << subSector << ", " << cscId << ", " << gblPhi.global_phi << ", " << gblEta.global_eta << std::endl;
		}
	}

	// Initialize Arrays
	////////////////////
	int nDataMuons = 0; 
	int nEmulMuons = 0;
	int dataMuonArray[8][9], emuMuonArray[8][9];
	for(int muon=0; muon<8; muon++)
	{
		for(int par=0; par<3; par++)
		{
			dataMuonArray[muon][par]=0;
			emuMuonArray[muon][par] =0;
		}
		emuMuonArray[muon][3] =-1;
		dataMuonArray[muon][3]=-1;
		
		emuMuonArray[muon][4]=7;
		dataMuonArray[muon][4]=7;
		
		for(int par2=5; par2<9; par2++)
		{
			emuMuonArray[muon][par2]= -1;
			dataMuonArray[muon][par2]= -1;
		}
	}
	// Get Hardware information, and check output of PtLUT
	//////////////////////////////////////////////////////
	if( dataTrackProducer.label() != "null" )
	{
		edm::Handle<L1CSCTrackCollection> tracks;
		e.getByLabel(dataTrackProducer.label(),dataTrackProducer.instance(),tracks);
		// check validity of input collection
		/////////////////////////////////////
		if(!tracks.isValid()) {
		  cout
		    << "\n No valid [data tracks] product found: "
		    << " L1CSCTrackCollection"
		    << endl;
		  return;
		}
		for(L1CSCTrackCollection::const_iterator trk=tracks.product()->begin(); trk!=tracks.product()->end(); trk++)
		{
			if( (nDataMuons < 8) && (trk->first.BX() <2) && (trk->first.BX() > -2) )
			{
				//int mOdE = (trk->first.ptLUTAddress()>>16)&0xf; 
				//std::cout << "D->Mode: " << mOdE << ", Rank " << trk->first.rank() << std::endl;
				dataMuonArray[nDataMuons][0] = trk->first.ptLUTAddress();  
				dataMuonArray[nDataMuons][1] = trk->first.sector();
				dataMuonArray[nDataMuons][2] = trk->first.endcap();
				dataMuonArray[nDataMuons][8] = trk->first.outputLink();
				dataMuonArray[nDataMuons][4] = trk->first.BX();
				dataMuonArray[nDataMuons][5] = trk->first.rank();
				dataMuonArray[nDataMuons][6] = trk->first.localPhi();
				dataMuonArray[nDataMuons][7] = trk->first.eta_packed();
				nDataMuons++;
			}
		}
	}
	// Get Emulator information
	///////////////////////////
	if( emulTrackProducer.label() != "null" )
	{
		edm::Handle<L1CSCTrackCollection> tracks;
		e.getByLabel(emulTrackProducer.label(),emulTrackProducer.instance(),tracks);
		// check validity of input collection
		/////////////////////////////////////
		if(!tracks.isValid()) {
		  edm::LogWarning("L1TdeCSCTF")
		    << "\n No valid [emulator tracks] product found: "
		    << " L1CSCTrackCollection"
		    << std::endl;
		  return;
		}
		for(L1CSCTrackCollection::const_iterator trk=tracks.product()->begin(); trk!=tracks.product()->end(); trk++)
		{
			if((nEmulMuons<8) && (trk->first.BX() <2) && (trk->first.BX() >-2))
			{
				//int mOdE = (trk->first.ptLUTAddress()>>16)&0xf; 
				//std::cout << "E->Mode: " << mOdE << ", Rank " << trk->first.rank() << std::endl;
				emuMuonArray[nEmulMuons][0] = trk->first.ptLUTAddress();
				emuMuonArray[nEmulMuons][1] = trk->first.sector();
				emuMuonArray[nEmulMuons][2] = trk->first.endcap();
				emuMuonArray[nEmulMuons][4] = trk->first.BX();
				emuMuonArray[nEmulMuons][5] = trk->first.rank();
				emuMuonArray[nEmulMuons][6] = trk->first.localPhi();
				emuMuonArray[nEmulMuons][7] = trk->first.eta_packed();
				nEmulMuons++;
			}
		}
	}
	
	if((nDataMuons!=0)||(nEmulMuons!=0)) numTrkComp->Fill(nDataMuons,nEmulMuons);
	
	// Match tracks by sector, endcap, and ptLUT address and Fill histograms
	////////////////////////////////////////////////////////////////////////
	if(nDataMuons==nEmulMuons)
	{
		//First, find EXACT address matches in a given sector, endcap
		for(int mu1=0; mu1<nDataMuons; mu1++)
		{
			for(int mu2=0; mu2<nEmulMuons; mu2++)
			if((emuMuonArray[mu2][1]==dataMuonArray[mu1][1])&&(emuMuonArray[mu2][2]==dataMuonArray[mu1][2]))
			{
				if(emuMuonArray[mu2][0]==dataMuonArray[mu1][0])
				{
					emuMuonArray[mu2][3]=mu1;
					dataMuonArray[mu1][3]=1;
				}
			}
		}
		//Next, try to match unmapped 
		for(int c2a=0; c2a<nEmulMuons; c2a++)
		{
			if(emuMuonArray[c2a][3]==-1)
			{
				for(int cor_a=0; cor_a<nDataMuons; cor_a++)
				{
					if( (dataMuonArray[cor_a][1]==emuMuonArray[c2a][1]) && (dataMuonArray[cor_a][2]==emuMuonArray[c2a][2]))// && (dataMuonArray[cor_a][3]==-1))
					{
						emuMuonArray[c2a][3]=cor_a;
						dataMuonArray[cor_a][3]=1;
					}
				}
			}
		}
		//Check that a single emulator track is not mapped to multiple data tracks
		bool multiMap = false;
		if(nEmulMuons!=1)
		{
			for(int c1a=0; c1a<(nEmulMuons-1); c1a++) 
			{
				for(int c1b=(c1a+1); c1b<nEmulMuons; c1b++)
				{
					if(emuMuonArray[c1a][3]==emuMuonArray[c1b][3])
					{
						//std::cout << "Error: Multiple Emulator Muons Mapped to the same Data Muon." << std::endl;
						multiMap = true;
					}
				}
			}
		}
		//Fill histograms based on matched Tracks
		for(int mu3=0; mu3<nEmulMuons; mu3++) 
		{
			int mapping = emuMuonArray[mu3][3];
			if((mapping!=-1)&&(multiMap==false))
			{
				//Decode LUT Address for more meaningful comparison
				int emuPhi12 = (0x0000ff & emuMuonArray[mu3][0]);
				int datPhi12 = (0x0000ff & dataMuonArray[mapping][0]);
				int emuPhi23 = (0x000f00 & emuMuonArray[mu3][0])>>8;
				int datPhi23 = (0x000f00 & dataMuonArray[mapping][0])>>8;
				int emuEta   = (0x00f000 & emuMuonArray[mu3][0])>>12;
				int datEta   = (0x00f000 & dataMuonArray[mapping][0])>>12;
				int emuMode = (0x0f0000 & emuMuonArray[mu3][0])>>16;
				int datMode = (0x0f0000 & dataMuonArray[mapping][0])>>16;
				int emuFrSin = (0xf00000 & emuMuonArray[mu3][0])>>20;
				int datFrSin = (0xf00000 & dataMuonArray[mapping][0])>>20;
				//Decode Rank for more meaningful comparison
				int emuQual  = emuMuonArray[mu3][5]>>5;
				int datQual  = dataMuonArray[mapping][5]>>5;
				int emuPt    = 0x1f & emuMuonArray[mu3][5];
				int datPt    = 0x1f & dataMuonArray[mapping][5];
				
				modeComp->Fill(datMode,emuMode);
				if(emuMode==datMode)
				{
					//Compare Pt LUT address fields
					phi12Comp->Fill(datPhi12,emuPhi12);
					phi23Comp->Fill(datPhi23,emuPhi23);
					etaComp->Fill(datEta,emuEta);
					signFrComp->Fill(datFrSin,emuFrSin);
					bxComp->Fill(dataMuonArray[mapping][4],emuMuonArray[mu3][4]);
					if(dataMuonArray[mapping][8]==1) // Rank Comp for Link 1
					{
						rankComp->Fill(dataMuonArray[mapping][5],emuMuonArray[mu3][5]);
						qualComp->Fill(datQual, emuQual);
						ptComp->Fill(datPt, emuPt);
					}
					//Compare Track Fields
					phiVComp->Fill(dataMuonArray[mapping][6],emuMuonArray[mu3][6]);
					etaVComp->Fill(dataMuonArray[mapping][7],emuMuonArray[mu3][7]);
					
					//Bad Match Debuging
					if(dataMuonArray[mapping][4]!=emuMuonArray[mu3][4]) // Bad Bx Match
					{
						badBxMode->Fill(emuMode);
						//std::cout << "Bad Bx Match, event: " << e.id().event() << std::endl;
					}
					if(datPhi12!=emuPhi12) // Bad Phi Match Plots
					{
						badPhiMode->Fill(emuMode);
						badPhiEta->Fill(emuEta);
						//std::cout << "Bad Phi Match, event: " << e.id().event() << std::endl;
					}
					if(datEta!=emuEta) // Bad Eta Match Plots
					{
						badEtaMode->Fill(emuMode);
						badEtaPhi ->Fill(emuPhi12);
						//std::cout << "Bad Eta Match, event: " << e.id().event() << std::endl;
					}
					if(dataMuonArray[mapping][5]!=emuMuonArray[mu3][5]&&(dataMuonArray[mapping][8]==1)) // Bad Rank Match Plots
					{
						badRankMode->Fill(emuMode);
						badRankEta->Fill(emuEta);
						if(datMode!=15) badRankOcc->Fill(nDataMuons);
						badRankLink->Fill(dataMuonArray[mapping][3]);
						badRankSecEnd->Fill(emuMuonArray[mu3][1],emuMuonArray[mu3][2]);
						
						std::cout << "Rank Disagreement event: " << e.id().event() << ", Lumi: " << e.luminosityBlock() << std::endl;
						
					}
				} else { //bad mode match
					badSectorEnd->Fill(emuMuonArray[mu3][1],emuMuonArray[mu3][2]);
					badModePhi->Fill(emuPhi12);
					badModeEta->Fill(emuEta);
					std::cout << "Bad Mode Match, event: " << e.id().event() << std::endl;
				}
			}//mapping != -1
		}//mu3
				
	} else { //Create Histograms to debug Track Occupancy Problems
		if(nDataMuons>nEmulMuons)
		{
			for(int muE3=0; muE3<nEmulMuons; muE3++)
			{
				int emuMode = (0x0f0000 & emuMuonArray[muE3][0] )>>16;
				moreDatModeE->Fill(emuMode);
			}
			for(int muD3=0; muD3<nEmulMuons; muD3++)
			{
				int datMode = (0x0f0000 & emuMuonArray[muD3][0] )>>16;
				moreDatModeD->Fill(datMode);
			}
		} else {
			for(int muE3=0; muE3<nEmulMuons; muE3++)
			{
				int emuMode = (0x0f0000 & emuMuonArray[muE3][0] )>>16;
				moreEmuModeE->Fill(emuMode);
			}
			for(int muD3=0; muD3<nEmulMuons; muD3++)
			{
				int datMode = (0x0f0000 & emuMuonArray[muD3][0] )>>16;
				moreEmuModeD->Fill(datMode);
			}	
		}
	}
	
	// Infostructure for DT comparison
	int eDtStub[7][15];
	int dDtStub[8][15];
	int eDtCounter = 0;
	int dDtCounter = 0;
	for(int dJ=0; dJ<7; dJ++)
	{
		for(int dK=0; dK<15; dK++)
		{
			eDtStub[dJ][dK] = -55;
			dDtStub[dJ][dK] = -55;
			dDtStub[7][dK] = -55;
		}
	}
	// Get Daq Recorded Stub Information
	edm::Handle<CSCTriggerContainer<csctf::TrackStub> > dtTrig;
	e.getByLabel("csctfunpacker","DT",dtTrig);
	const CSCTriggerContainer<csctf::TrackStub>* dt_stubs = dtTrig.product();
	CSCTriggerContainer<csctf::TrackStub> stub_list;
	stub_list.push_many(*dt_stubs);
	std::vector<csctf::TrackStub> stuList = stub_list.get();
	std::vector<csctf::TrackStub>::const_iterator stu= stuList.begin();
	for(; stu!=stuList.end(); stu++)
	{
	
		stubDatBx->Fill(stu->BX());
	
		if(dDtCounter<15 && (stu->BX()>4) && (stu->BX()<9))
		{
			dDtStub[0][dDtCounter] = stu->phiPacked();
			dDtStub[1][dDtCounter] = stu->getQuality();
			dDtStub[2][dDtCounter] = stu->endcap();
			dDtStub[3][dDtCounter] = stu->sector();
			dDtStub[4][dDtCounter] = stu->subsector();
			
			std::cout << 
				"Dat Stub. Phi: " << stu->phiPacked() << 
				", Qual: " << stu->getQuality() <<
				", endcap: " << stu->endcap() <<
				", sector: " << stu->sector() <<
				", subSec: " << stu->subsector() <<
				std::endl;
			
			dDtCounter++;
		}
	}
	// Get Emulated Stub Information
	edm::Handle<L1MuDTChambPhContainer> pCon;
	e.getByLabel("dttfunpacker", pCon);
	CSCTriggerContainer<csctf::TrackStub> emulStub = my_dtrc->process(pCon.product());
	std::vector<csctf::TrackStub> emuList = emulStub.get();
	std::vector<csctf::TrackStub>::const_iterator eStu=emuList.begin();
	for(; eStu!=emuList.end(); eStu++)
	{
		
		stubEmuBx->Fill(eStu->BX());
	
		if( (eDtCounter<15) && (eStu->BX()>4) && (eStu->BX()<9) )
		{
			eDtStub[0][eDtCounter] = eStu->phiPacked();
			eDtStub[1][eDtCounter] = eStu->getQuality();
			eDtStub[2][eDtCounter] = eStu->endcap();
			eDtStub[3][eDtCounter] = eStu->sector();
			eDtStub[4][eDtCounter] = eStu->subsector();

			std::cout << 
				"Emu Stub. Phi: " << eDtStub[0][eDtCounter] << 
				", Qual: " << eDtStub[1][eDtCounter] <<
				", endcap: " << eDtStub[2][eDtCounter] <<
				", sector: " << eDtStub[3][eDtCounter] <<
				", subSec: " << eDtStub[4][eDtCounter] <<
				std::endl;
			
			eDtCounter++;
		}
	}
	//Match Stubs if numbers are equal
	//if(eDtCounter==dDtCounter)
	//{
		//std::cout << "Num Tracks match eDtCounter: " << eDtCounter << ", dDt: " << dDtCounter << std::endl;
		//First find perfect matches
		for(int eS=0; eS<eDtCounter; eS++)
		{
			//std::cout << "es Loop" << std:: endl;
			for(int dS=0; dS<dDtCounter; dS++)
			{
				//std::cout << "ds Loop" << std::endl;
				if(eDtStub[2][eS]==dDtStub[2][dS])
				{
					//std::cout << "end match" << std::endl;
					if(eDtStub[3][eS]==dDtStub[3][dS])
					{
						//std::cout << "sec match" << std::endl;
						if(eDtStub[4][eS]==dDtStub[4][dS]) 
						{
							//std::cout << "First match loop, eS: " << eS << ", dS" << dS << std::endl;
							if( (eDtStub[0][eS]==dDtStub[0][dS]) && (eDtStub[1][eS]==dDtStub[1][dS]) && (eDtStub[6][eS]!=1) && (dDtStub[6][dS]!=1) )
							{
								std::cout << "Passed fist matching." << std::endl;
								eDtStub[5][eS] = dS;
								eDtStub[6][eS] = 1;
								dDtStub[5][dS] = eS;
								dDtStub[6][dS] = 1;
							}
						}
					}
				}
			}
		}
		//Now find imperfect matches
		for(int eS2=0; eS2<eDtCounter; eS2++)
		{
			for(int dS2=0; dS2<dDtCounter; dS2++)
			{
				std::cout << "1: " << eDtStub[2][eS2] << ", " << dDtStub[2][dS2] << ", " << eDtStub[3][eS2] << ", " << dDtStub[3][dS2] << ", " << eDtStub[4][eS2] << ", " << dDtStub[4][dS2] << std::endl;
				if( (eDtStub[2][eS2]==dDtStub[2][dS2]) && (eDtStub[3][eS2]==dDtStub[3][dS2]) && (eDtStub[4][eS2]==dDtStub[4][dS2]) )
				{
					std::cout << "2: " << dDtStub[7][eS2] << ", " << dDtStub[7][dS2] << ", " << abs(eDtStub[0][eS2]-dDtStub[0][dS2]) << ", " << ", " << eDtStub[6][eS2] << ", " << dDtStub[6][dS2] << std::endl;
					if( ((dDtStub[7][eS2]==-55) || (dDtStub[7][dS2]>(abs(eDtStub[0][eS2]-dDtStub[0][dS2]))) ) && (eDtStub[6][eS2]!=1) && (dDtStub[6][dS2]!=1)  )
					{
						std::cout << "Imperfect match found" << std::endl;
						dDtStub[5][dS2] = eS2;
						dDtStub[6][dS2] = 2;
						eDtStub[5][eS2] = dS2;
						eDtStub[6][eS2] = 2;
						dDtStub[7][dS2] = abs(eDtStub[0][eS2]-dDtStub[0][dS2]);
					}
				}
			}
		}
		
		//Debug time!
		bool dtSMulti = false;
		int dtUnmap  = 0;
		if(eDtCounter>1)
			for(int eS3a=0; eS3a<eDtCounter-1; eS3a++)
				for(int eS3b=eS3a+1; eS3b<eDtCounter; eS3b++)
				{
					if( eDtStub[5][eS3a]==eDtStub[5][eS3b] ) dtSMulti=true;
					if( eDtStub[5][eS3a]==-55 || eDtStub[5][eS3b]==-55 ) dtUnmap++;
				}
				
		if(dDtCounter>1)
			for(int dS3a=0; dS3a<dDtCounter-1; dS3a++)
				for(int dS3b=dS3a+1; dS3b<dDtCounter; dS3b++)
				{
					if( dDtStub[5][dS3a]==dDtStub[5][dS3b] ) dtSMulti=true;
					if( dDtStub[5][dS3a]==-55||dDtStub[5][dS3b]==-55 ) dtUnmap++;
				}
		if(dtSMulti==true)
			std::cout << "Multiple DT stubs mapped to the same stub" << std::endl;
		if(dtUnmap!=0)
			std::cout << "Unmapped DT stubs:" << dtUnmap << std::endl;
			
		if(dtSMulti==false && dtUnmap==0)
		{
			for(int phil=0; phil<eDtCounter; phil++)
			{
				if(eDtStub[6][phil]==1 || eDtStub[6][phil]==2)
				{
					int indexFil = (eDtStub[3][phil]-1)*2+eDtStub[4][phil];
					stubDtPhi->Fill(eDtStub[0][phil],  dDtStub[0][ eDtStub[5][phil] ]);
					stubDtQual->Fill(eDtStub[1][phil], dDtStub[1][ eDtStub[5][phil] ]);
					if( eDtStub[0][phil] != dDtStub[0][ eDtStub[5][phil] ])
						stubBadPhiSec->Fill(indexFil,eDtStub[2][phil]);
				}
			}
		}
		
	//}
}

void CsctfDatEmu::endJob()
{
	//std::cout << "Number of events with DTs:" << dtCount << ", CSCs: " << cscCount << ", and both: " << dcCount << std::endl;
	//Label Axis
	////////////
	modeComp->GetXaxis()->SetTitle("H.W. Mode");
	modeComp->GetYaxis()->SetTitle("S.W. Mode");
	numTrkComp->GetXaxis()->SetTitle("H.W. Occ");
	numTrkComp->GetYaxis()->SetTitle("S.W. Occ");
	bxComp->GetXaxis()->SetTitle("H.W. BX");
	bxComp->GetYaxis()->SetTitle("S.W. BX");
	rankComp->GetXaxis()->SetTitle("H.W. Rank");
	rankComp->GetYaxis()->SetTitle("S.W. Rank");
	ptComp->GetXaxis()->SetTitle("H.W. P_{t}");
	ptComp->GetYaxis()->SetTitle("S.W. P_{t}");
	qualComp->GetXaxis()->SetTitle("H.W. Quality");
	qualComp->GetYaxis()->SetTitle("S.W. Quality");
	etaComp->GetXaxis()->SetTitle("H.W. #eta");
	etaComp->GetYaxis()->SetTitle("S.W. #eta");
	phi12Comp->GetXaxis()->SetTitle("H.W. #Delta #phi_{12}");
	phi12Comp->GetYaxis()->SetTitle("S.W. #Delta #phi_{12}");  
	phi23Comp->GetXaxis()->SetTitle("H.W. #Delta #phi_{23}");
	phi23Comp->GetYaxis()->SetTitle("S.W. #Delta #phi_{23}");
	phiVComp->GetXaxis()->SetTitle("H.W. Track #phi_{local}");
	phiVComp->GetYaxis()->SetTitle("S.W. Track #phi_{local}");
	etaVComp->GetXaxis()->SetTitle("H.W. Track #eta");
	etaVComp->GetYaxis()->SetTitle("S.W. Track #eta");
	
	stubDifBx->Add(stubEmuBx,stubDatBx,1,-1);
	
	dtBBx->SetLineColor(2);
	dtBBx0->SetLineColor(2);
	dtBMlink->SetLineColor(2);
	dtBQ->SetLineColor(2);
	dtBBend->SetLineColor(2);
	dtBStrip->SetLineColor(2);
	
	//////////////////////
	//// Print Histos ////
	//////////////////////
	fAnalysis = new TFile(outFile.c_str(), "RECREATE");//c
	TObjArray Hlist(0);
	Hlist.Add(stubDtPhi);
	Hlist.Add(stubDtStrip);
	Hlist.Add(stubDtBx);
	Hlist.Add(stubDtBX0);
	Hlist.Add(stubDtMpc);
	Hlist.Add(stubDtQual);
	Hlist.Add(stubDtBend);
	Hlist.Add(stubDtoTrack);
	Hlist.Add(stubCtoTrack);
	Hlist.Add(badRankLink);
	Hlist.Add(etaVComp);
	Hlist.Add(phiVComp);
	Hlist.Add(badRankMode);
	Hlist.Add(badRankEta);
	Hlist.Add(badRankOcc);
	Hlist.Add(rankComp);
	Hlist.Add(qualComp);
	Hlist.Add(ptComp);
	Hlist.Add(modeComp);
	Hlist.Add(numTrkComp);
	Hlist.Add(phi12Comp );
	Hlist.Add(phi23Comp );
	Hlist.Add(etaComp   );
	Hlist.Add(signFrComp);
	Hlist.Add(badPhiMode);
	Hlist.Add(moreDatModeE);
	Hlist.Add(moreEmuModeE);
	Hlist.Add(moreDatModeD);
	Hlist.Add(moreEmuModeD);
	Hlist.Add(badEtaMode);
	Hlist.Add(badEtaPhi);
	Hlist.Add(badPhiEta);
	Hlist.Add(badSectorEnd);
	Hlist.Add(badModePhi);
	Hlist.Add(badModeEta);
	Hlist.Add(badBxMode);
	Hlist.Add(bxComp);
	Hlist.Add(stubBadPhiSec);
	Hlist.Add(stubEmuBx);
	Hlist.Add(stubDatBx);
	Hlist.Add(stubDifBx);
	//Hlist.Add(dtC);
	Hlist.Write();
	delete fAnalysis;
}
