/*
 * L1TdeCSCTF.cc v1.0
 * written by J. Gartner
 *
 * 2011.03.11 expanded by GP Di Giovanni
 * 
 * There is quality test allowing to check elements outside the
 * diagonal, so I need to add the 1D plot with all elements in the diagonal
 * in the first bin and all elements outside the diagonal in the second bin
 * 
 * In such way we can run the ContentsXRange quality test...
 */
 
#include "DQM/L1TMonitor/interface/L1TdeCSCTF.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h>

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"


#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <iostream>
#include <iomanip>
#include <memory>


using namespace std;
using namespace edm;

L1TdeCSCTF::L1TdeCSCTF(ParameterSet const& pset) {
  dataTrackProducer = consumes<L1CSCTrackCollection>(pset.getParameter<InputTag>("dataTrackProducer"));
  emulTrackProducer = consumes<L1CSCTrackCollection>(pset.getParameter<InputTag>("emulTrackProducer"));
  dataStubProducer  = consumes<CSCTriggerContainer<csctf::TrackStub> >(pset.getParameter<InputTag>("dataStubProducer"));
  emulStubProducer  = consumes<L1MuDTChambPhContainer>(pset.getParameter<InputTag>("emulStubProducer"));
	
  m_dirName         = pset.getUntrackedParameter("DQMFolder", string("L1TEMU/CSCTFexpert"));

  ts=0;
  ptLUT_ = 0;
	
  ptLUTset = pset.getParameter<ParameterSet>("PTLUT");
	
  outFile = pset.getUntrackedParameter<string>("outFile", "");
  if( outFile.size() != 0 )
  {
      LogWarning("L1TdeCSCTF")
	    << "L1T Monitoring histograms will be saved to " 
	    << outFile.c_str() 
	    << endl;
  }
	
  bool disable = pset. getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
		outFile="";
  }
	
	/*bzero(srLUTs_, sizeof(srLUTs_));
	//int endcap =1, sector =1;
	bool TMB07=true;
	ParameterSet srLUTset;
	srLUTset.addUntrackedParameter<bool>("ReadLUTs", false);
  	srLUTset.addUntrackedParameter<bool>("Binary",   false);
  	srLUTset.addUntrackedParameter<string>("LUTPath", "./");
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
	*/
  my_dtrc = new CSCTFDTReceiver();
}
void L1TdeCSCTF::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c){
}

void L1TdeCSCTF::beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c){
}

void L1TdeCSCTF::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&) 
{
  //Histograms booking

  /////////////////////////////
  // DQM Directory Structure //
  /////////////////////////////
  
  ibooker.setCurrentFolder(m_dirName);

  /////////////////////////////
  // Define Monitor Elements //
  /////////////////////////////
  //Monitor Elements for Pt Lut Address Field
  pt1Comp = ibooker.book2D("pt1Comp","Hardware Vs. Emulator #Delta #phi_{12}",256,0,256,256,0,256);
  pt1Comp->setAxisTitle("Hardware #Delta #phi_{12}",1);
  pt1Comp->setAxisTitle("Emulator #Delta #phi_{12}",2);
  pt2Comp = ibooker.book2D("pt2Comp","Hardware Vs. Emulator #Delta #phi_{23}",16,0,16,16,0,16);
  pt2Comp->setAxisTitle("Hardware #Delta #phi_{23}",1);
  pt2Comp->setAxisTitle("Emulator #Delta #phi_{23}",2);
  pt3Comp = ibooker.book2D("pt3Comp","Hardware Vs. Emulator #eta",16,0,16,16,0,16);
  pt3Comp->setAxisTitle("Hardware #eta",1);
  pt3Comp->setAxisTitle("Emulator #eta",2);
  pt4Comp = ibooker.book2D("pt4Comp","Hardware Vs. Emulator Mode",19,0,19,19,0,19);
  pt4Comp->setAxisTitle("Hardware Mode",1);
  pt4Comp->setAxisTitle("Emulator Mode",2);
  //Hardware Bin Titles
  pt4Comp->setBinLabel(1,"No Track",1);
  pt4Comp->setBinLabel(2,"Bad Phi/Single",1);
  pt4Comp->setBinLabel(3,"ME1-2-3",1);
  pt4Comp->setBinLabel(4,"ME1-2-4",1);
  pt4Comp->setBinLabel(5,"ME1-3-4",1);
  pt4Comp->setBinLabel(6,"ME2-3-4",1);
  pt4Comp->setBinLabel(7,"ME1-2",1);
  pt4Comp->setBinLabel(8,"ME1-3",1);
  pt4Comp->setBinLabel(9,"ME2-3",1);
  pt4Comp->setBinLabel(10,"ME2-4",1);
  pt4Comp->setBinLabel(11,"ME3-4",1);
  pt4Comp->setBinLabel(12,"MB1-ME3",1);
  pt4Comp->setBinLabel(13,"MB1-ME2",1);
  pt4Comp->setBinLabel(14,"ME1-4",1);
  pt4Comp->setBinLabel(15,"MB1-ME1",1);
  pt4Comp->setBinLabel(16,"Halo Trigger",1);
  pt4Comp->setBinLabel(17,"MB1-ME1-2",1);
  pt4Comp->setBinLabel(18,"MB1-ME1-3",1);
  pt4Comp->setBinLabel(19,"MB1-ME2-3",1);
  //Emu Bin Titles
  pt4Comp->setBinLabel(1,"No Track",2);
  pt4Comp->setBinLabel(2,"Bad Phi/Single",2);
  pt4Comp->setBinLabel(3,"ME1-2-3",2);
  pt4Comp->setBinLabel(4,"ME1-2-4",2);
  pt4Comp->setBinLabel(5,"ME1-3-4",2);
  pt4Comp->setBinLabel(6,"ME2-3-4",2);
  pt4Comp->setBinLabel(7,"ME1-2",2);
  pt4Comp->setBinLabel(8,"ME1-3",2);
  pt4Comp->setBinLabel(9,"ME2-3",2);
  pt4Comp->setBinLabel(10,"ME2-4",2);
  pt4Comp->setBinLabel(11,"ME3-4",2);
  pt4Comp->setBinLabel(12,"MB1-ME3",2);
  pt4Comp->setBinLabel(13,"MB1-ME2",2);
  pt4Comp->setBinLabel(14,"ME1-4",2);
  pt4Comp->setBinLabel(15,"MB1-ME1",2);
  pt4Comp->setBinLabel(16,"Halo Trigger",2);
  pt4Comp->setBinLabel(17,"MB1-ME1-2",2);
  pt4Comp->setBinLabel(18,"MB1-ME1-3",2);
  pt4Comp->setBinLabel(19,"MB1-ME2-3",2);

  pt5Comp = ibooker.book2D("pt5Comp","Hardware Vs. Emulator Sign, FR",4,0,4,4,0,4);
  pt5Comp->setAxisTitle("Hardware Sign<<1|FR",1);
  pt5Comp->setAxisTitle("Emulator Sign<<1|FR",2);
		
  //Monitor Elements for track variables
  phiComp = ibooker.book2D("phiComp","Hardware Vs. Emulator Track #phi",32,0,32,32,0,32);
  phiComp->setAxisTitle("Hardware #phi",1);
  phiComp->setAxisTitle("Emulator #phi",2);
  etaComp = ibooker.book2D("etaComp","Hardware Vs. Emulator Track #eta",32,0,32,32,0,32);
  etaComp->setAxisTitle("Hardware #eta",1);
  etaComp->setAxisTitle("Emulator #eta",2);
  occComp = ibooker.book2D("occComp","Hardware Vs. Emulator Track Occupancy",5,0,5,5,0,5);
  occComp->setAxisTitle("Hardware Occupancy",1);
  occComp->setAxisTitle("Emulator Occupancy",2);
  ptComp  = ibooker.book2D("ptComp","Hardware Vs. Emulator Pt",32,0,32,32,0,32);
  ptComp->setAxisTitle("Hardware P_{t}",1);
  ptComp->setAxisTitle("Emulator P_{t}",2);
  qualComp= ibooker.book2D("qualComp","Hardware Vs. Emulator Quality",4,0,4,4,0,4);
  qualComp->setAxisTitle("Hardware Quality",1);
  qualComp->setAxisTitle("Emulator Quality",2);
		
		
  //Monitor Elemens for Dt Stubs
  dtStubPhi = ibooker.book2D("dtStubPhi","Hardware Vs. Emulator DT Stub #phi",200,400,2400,200,400,2400);
  dtStubPhi->setAxisTitle("Hardware Stub #phi",1);
  dtStubPhi->setAxisTitle("Emulator Stub #phi",2);
  badDtStubSector = ibooker.book2D("badDtStubSector","Dt Sector for bad Dt stub #phi",6,1,7,2,1,3);
  badDtStubSector->setAxisTitle("Dt stub sector, subsector",1);
  badDtStubSector->setAxisTitle("Dt Stub Endcap",2);

  //***********************************//
  //* F O R   Q U A L I T Y   T E S T *//
  //***********************************//
  //1D plots for the quality test
  //Monitor Elements for Pt Lut Address Field
  pt1Comp_1d = ibooker.book1D("pt1Comp_1d","Hardware Vs. Emulator #Delta #phi_{12}",2,0,2);
  pt1Comp_1d->setAxisTitle("#Delta #phi_{12}",1);
  pt1Comp_1d->setBinLabel(1, "Agree", 1);
  pt1Comp_1d->setBinLabel(2, "Disagree", 1);

  pt2Comp_1d = ibooker.book1D("pt2Comp_1d","Hardware Vs. Emulator #Delta #phi_{23}",2,0,2);
  pt2Comp_1d->setAxisTitle("#Delta #phi_{23}",1);
  pt2Comp_1d->setBinLabel(1, "Agree", 1);
  pt2Comp_1d->setBinLabel(2, "Disagree", 1);

  pt3Comp_1d = ibooker.book1D("pt3Comp_1d","Hardware Vs. Emulator #eta",2,0,2);
  pt3Comp_1d->setAxisTitle("#eta",1);
  pt3Comp_1d->setBinLabel(1, "Agree", 1);
  pt3Comp_1d->setBinLabel(2, "Disagree", 1);

  pt4Comp_1d = ibooker.book1D("pt4Comp_1d","Hardware Vs. Emulator Mode",2,0,2);
  pt4Comp_1d->setAxisTitle("Mode",1);
  pt4Comp_1d->setBinLabel(1, "Agree", 1);
  pt4Comp_1d->setBinLabel(2, "Disagree", 1);

  pt5Comp_1d = ibooker.book1D("pt5Comp_1d","Hardware Vs. Emulator Sign, FR",2,0,2);
  pt5Comp_1d->setAxisTitle("Sign<<1|FR",1);
  pt5Comp_1d->setBinLabel(1, "Agree", 1);
  pt5Comp_1d->setBinLabel(2, "Disagree", 1);
               
		
  //Monitor Elements for track variables
  phiComp_1d = ibooker.book1D("phiComp_1d","Hardware Vs. Emulator Track #phi",2,0,2);
  phiComp_1d->setAxisTitle("#phi",1);
  phiComp_1d->setBinLabel(1, "Agree", 1);
  phiComp_1d->setBinLabel(2, "Disagree", 1);

  etaComp_1d = ibooker.book1D("etaComp_1d","Hardware Vs. Emulator Track #eta",2,0,2);
  etaComp_1d->setAxisTitle("#eta",1);
  etaComp_1d->setBinLabel(1, "Agree", 1);
  etaComp_1d->setBinLabel(2, "Disagree", 1);

  occComp_1d = ibooker.book1D("occComp_1d","Hardware Vs. Emulator Track Occupancy",2,0,2);
  occComp_1d->setAxisTitle("Occupancy",1);
  occComp_1d->setBinLabel(1, "Agree", 1);
  occComp_1d->setBinLabel(2, "Disagree", 1);

  ptComp_1d  = ibooker.book1D("ptComp_1d","Hardware Vs. Emulator Pt",2,0,2);
  ptComp_1d->setAxisTitle("P_{t}",1);
  ptComp_1d->setBinLabel(1, "Agree", 1);
  ptComp_1d->setBinLabel(2, "Disagree", 1);

  qualComp_1d= ibooker.book1D("qualComp_1d","Hardware Vs. Emulator Quality",2,0,2);
  qualComp_1d->setAxisTitle("Quality",1);
  qualComp_1d->setBinLabel(1, "Agree", 1);
  qualComp_1d->setBinLabel(2, "Disagree", 1);

  //Monitor Elemens for Dt Stubs
  dtStubPhi_1d = ibooker.book1D("dtStubPhi_1d","Hardware Vs. Emulator DT Stub #phi",2,0,2);
  dtStubPhi_1d->setAxisTitle("DT Stub #phi",1);
  dtStubPhi_1d->setBinLabel(1, "Agree", 1);
  dtStubPhi_1d->setBinLabel(2, "Disagree", 1); 
}

void L1TdeCSCTF::analyze(Event const& e, EventSetup const& es){
   // Initialize Arrays
  ////////////////////
  unsigned int nDataMuons = 0;
  unsigned int nEmulMuons = 0;
  int dataMuonArray[8][10], emuMuonArray[8][10];
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
		
    for(int par2=5; par2<10; par2++)
    {
      emuMuonArray[muon][par2]= -1;
      dataMuonArray[muon][par2]= -1;
    }
  }
    // Get Hardware information, and check output of PtLUT
    //////////////////////////////////////////////////////
  if( !dataTrackProducer.isUninitialized())
  {
    Handle<L1CSCTrackCollection> tracks;
    e.getByToken(dataTrackProducer,tracks);

    // check validity of input collection
    if(!tracks.isValid()) {
      LogWarning("L1TdeCSCTF")
	   << "\n No valid [data tracks] product found: "
           << " L1CSCTrackCollection"
           << endl;
      return;
  }


  for(L1CSCTrackCollection::const_iterator trk=tracks.product()->begin(); trk!=tracks.product()->end(); trk++)
  {
    if (nDataMuons>=8)
      break;
      if ( (trk->first.BX() <2) && (trk->first.BX() > -1) )
        {
	  //int mOdE = (trk->first.ptLUTAddress()>>16)&0xf; 
	  //cout << "D->Mode: " << mOdE << ", Rank " << trk->first.rank() << endl;
	  dataMuonArray[nDataMuons][0] = trk->first.ptLUTAddress();  
          dataMuonArray[nDataMuons][1] = trk->first.sector();
          dataMuonArray[nDataMuons][2] = trk->first.endcap();
          dataMuonArray[nDataMuons][8] = trk->first.outputLink();
          dataMuonArray[nDataMuons][4] = trk->first.BX();
          dataMuonArray[nDataMuons][5] = trk->first.rank();
          dataMuonArray[nDataMuons][6] = trk->first.localPhi();
          dataMuonArray[nDataMuons][7] = trk->first.eta_packed();
          dataMuonArray[nDataMuons][9] = trk->first.modeExtended();
          nDataMuons++;
        }
    }
  }
  // Get Emulator information
  ///////////////////////////
  if( !emulTrackProducer.isUninitialized() )
  {
    Handle<L1CSCTrackCollection> tracks;
    e.getByToken(emulTrackProducer,tracks);

    // check validity of input collection
    if(!tracks.isValid()) {
      LogWarning("L1TdeCSCTF")
	   << "\n No valid [emulator tracks] product found: "
           << " L1CSCTrackCollection"
           << endl;
      return;
  }

  for(L1CSCTrackCollection::const_iterator trk=tracks.product()->begin(); trk!=tracks.product()->end(); trk++)
  {
     if(nEmulMuons>=8)
       break;
       if((trk->first.BX() <2) && (trk->first.BX() >-1))
         {
	    //int mOdE = (trk->first.ptLUTAddress()>>16)&0xf; 
	    //cout << "E->Mode: " << mOdE << ", Rank " << trk->first.rank() << endl;
	    emuMuonArray[nEmulMuons][0] = trk->first.ptLUTAddress();
	    emuMuonArray[nEmulMuons][1] = trk->first.sector();
	    emuMuonArray[nEmulMuons][2] = trk->first.endcap();
	    emuMuonArray[nEmulMuons][4] = trk->first.BX();
	    emuMuonArray[nEmulMuons][5] = trk->first.rank();
	    emuMuonArray[nEmulMuons][6] = trk->first.localPhi();
	    emuMuonArray[nEmulMuons][7] = trk->first.eta_packed();
	    emuMuonArray[nEmulMuons][9] = trk->first.modeExtended();
	    nEmulMuons++;
	 }
    }
  }
	//Fill Occupancy M.E.
  if( (nDataMuons!=0)||(nEmulMuons!=0) ) {
    occComp->Fill(nDataMuons,nEmulMuons);
    (nDataMuons==nEmulMuons) ? occComp_1d->Fill(0) : occComp_1d->Fill(1);
  }
	// Match Tracks by sector & mode in the case of multiple tracks
	///////////////////////////////////////////////////////////////
  if(nDataMuons==nEmulMuons)
  {
    //First, find EXACT address matches in a given sector, endcap
    for(unsigned int mu1=0; mu1<nDataMuons; mu1++)
    {
      for(unsigned int mu2=0; mu2<nEmulMuons; mu2++)
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
    for(unsigned int c2a=0; c2a<nEmulMuons; c2a++)
    {
      if(emuMuonArray[c2a][3]==-1)
      {
        for(unsigned int cor_a=0; cor_a<nDataMuons; cor_a++)
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
    if(nEmulMuons>1)
    {
      for(unsigned int c1a=0; c1a<(nEmulMuons-1); c1a++)
      {
        for(unsigned int c1b=(c1a+1); c1b<nEmulMuons; c1b++)
        {
          if(emuMuonArray[c1a][3]==emuMuonArray[c1b][3])
          {
	    //cout << "Error: Multiple Emulator Muons Mapped to the same Data Muon." << endl;
	    multiMap = true;
	    break;
	  }
	}
	if (multiMap)
	  break;
      }
    }
    //Fill histograms based on matched Tracks
    for(unsigned int mu3=0; mu3<nEmulMuons; mu3++)
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
	//int emuMode = (0x0f0000 & emuMuonArray[mu3][0])>>16;
	//int datMode = (0x0f0000 & dataMuonArray[mapping][0])>>16;
        int emuFrSin = (0xf00000 & emuMuonArray[mu3][0])>>20;
        int datFrSin = (0xf00000 & dataMuonArray[mapping][0])>>20;
	//Decode Rank for more meaningful comparison
        int emuQual  = emuMuonArray[mu3][5]>>5;
        int datQual  = dataMuonArray[mapping][5]>>5;
        int emuPt    = 0x1f & emuMuonArray[mu3][5];
        int datPt    = 0x1f & dataMuonArray[mapping][5];
        int emuModeExtended = emuMuonArray[mu3][9];
        int datModeExtended = dataMuonArray[mapping][9];
				
        //Fill mode M.E., one of (the most important) PtLUT address field
        pt4Comp->Fill(datModeExtended,emuModeExtended);
        (datModeExtended==emuModeExtended) ? pt4Comp_1d->Fill(0) : pt4Comp_1d->Fill(1);
        //To disentagle problems, only fill histograms if mode matches
	if(emuModeExtended==datModeExtended)
	{
	  //Fill Pt LUT address field M.E.
	  pt1Comp->Fill(datPhi12,emuPhi12); (datPhi12==emuPhi12) ? pt1Comp_1d->Fill(0) : pt1Comp_1d->Fill(1);
	  pt2Comp->Fill(datPhi23,emuPhi23); (datPhi23==emuPhi23) ? pt2Comp_1d->Fill(0) : pt2Comp_1d->Fill(1);
	  pt3Comp->Fill(datEta,emuEta);     (datEta==emuEta)     ? pt3Comp_1d->Fill(0) : pt3Comp_1d->Fill(1);
	  pt5Comp->Fill(datFrSin,emuFrSin); (datFrSin==emuFrSin) ? pt5Comp_1d->Fill(0) : pt5Comp_1d->Fill(1);
					//Fill Track value M.E.
	  if(dataMuonArray[mapping][8]==1) //Rank Comparison available for Link 1 only due to readout limitation
	  {
	    ptComp->Fill(datPt,emuPt);      (datPt==emuPt)     ? ptComp_1d->Fill(0)   : ptComp_1d->Fill(1);
	    qualComp->Fill(datQual,emuQual);(datQual==emuQual) ? qualComp_1d->Fill(0) : qualComp_1d->Fill(1);
          }
          phiComp->Fill(dataMuonArray[mapping][6],emuMuonArray[mu3][6]);
          etaComp->Fill(dataMuonArray[mapping][7],emuMuonArray[mu3][7]);

          (dataMuonArray[mapping][6]==emuMuonArray[mu3][6]) ? phiComp_1d->Fill(0) : phiComp_1d->Fill(1); 
          (dataMuonArray[mapping][7]==emuMuonArray[mu3][7]) ? etaComp_1d->Fill(0) : etaComp_1d->Fill(1);
        }
      }
    }
  }
	
  //Compare DT stubs to check transmission quality
  ////////////////////////////////////////////////
  //Declare arrays, initialize
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
  if( !dataStubProducer.isUninitialized() )
  {
    Handle<CSCTriggerContainer<csctf::TrackStub> > dtTrig;
    e.getByToken(dataStubProducer,dtTrig);
    // check validity of input collection
    if(!dtTrig.isValid()) {
      LogWarning("L1TdeCSCTF")
             << "\n No valid [Data Stubs] product found: "
             << " L1CSCTrackCollection"
             << endl;
      return;
    }
    const CSCTriggerContainer<csctf::TrackStub>* dt_stubs = dtTrig.product();
    CSCTriggerContainer<csctf::TrackStub> stub_list;
    stub_list.push_many(*dt_stubs);
    vector<csctf::TrackStub> stuList = stub_list.get();
    vector<csctf::TrackStub>::const_iterator stu= stuList.begin();
    for(; stu!=stuList.end(); stu++)
    {
      if(dDtCounter>=15)
        break;
        if((stu->BX()>4) && (stu->BX()<9))
        {
          dDtStub[0][dDtCounter] = stu->phiPacked();
	  dDtStub[1][dDtCounter] = stu->getQuality();
	  dDtStub[2][dDtCounter] = stu->endcap();
	  dDtStub[3][dDtCounter] = stu->sector();
	  dDtStub[4][dDtCounter] = stu->subsector();
	  dDtCounter++;
	}
    }
  }
	
  // Get Daq Recorded Stub Information
  if( !emulStubProducer.isUninitialized() )
  {
      // Get Emulated Stub Information
    Handle<L1MuDTChambPhContainer> pCon;
    e.getByToken(emulStubProducer,pCon);
    // check validity of input collection
    if(!pCon.isValid()) {
      LogWarning("L1TdeCSCTF")
           << "\n No valid [Data Stubs] product found: "
           << " L1CSCTrackCollection"
           << endl;
      return;
    }
    CSCTriggerContainer<csctf::TrackStub> emulStub = my_dtrc->process(pCon.product());
    vector<csctf::TrackStub> emuList = emulStub.get();
    vector<csctf::TrackStub>::const_iterator eStu=emuList.begin();
    for(; eStu!=emuList.end(); eStu++)
    {		
      if (eDtCounter>=15)
        break;
	if((eStu->BX()>4) && (eStu->BX()<9) )
	{
	  eDtStub[0][eDtCounter] = eStu->phiPacked();
	  eDtStub[1][eDtCounter] = eStu->getQuality();
	  eDtStub[2][eDtCounter] = eStu->endcap();
	  eDtStub[3][eDtCounter] = eStu->sector();
	  eDtStub[4][eDtCounter] = eStu->subsector();
	  eDtCounter++;
	}
    }
  }
	
  //cout << "Num Tracks match eDtCounter: " << eDtCounter << ", dDt: " << dDtCounter << endl;
  //First find perfect matches
  for(int eS=0; eS<eDtCounter; eS++)
  {
    //cout << "es Loop" <<  endl;
    for(int dS=0; dS<dDtCounter; dS++)
    {
      //cout << "ds Loop" << endl;
      if(eDtStub[2][eS]==dDtStub[2][dS])
      {
        //cout << "end match" << endl;
	if(eDtStub[3][eS]==dDtStub[3][dS])
	{
	  //cout << "sec match" << endl;
	  if(eDtStub[4][eS]==dDtStub[4][dS]) 
	  {
	    //cout << "First match loop, eS: " << eS << ", dS" << dS << endl;
            if( (eDtStub[0][eS]==dDtStub[0][dS]) && (eDtStub[1][eS]==dDtStub[1][dS]) && (eDtStub[6][eS]!=1) && (dDtStub[6][dS]!=1) )
            {
              //cout << "Passed fist matching." << endl;
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
			//cout << "1: " << eDtStub[2][eS2] << ", " << dDtStub[2][dS2] << ", " << eDtStub[3][eS2] << ", " << dDtStub[3][dS2] << ", " << eDtStub[4][eS2] << ", " << dDtStub[4][dS2] << endl;
      if( (eDtStub[2][eS2]==dDtStub[2][dS2]) && (eDtStub[3][eS2]==dDtStub[3][dS2]) && (eDtStub[4][eS2]==dDtStub[4][dS2]) )
      {
				//cout << "2: " << dDtStub[7][eS2] << ", " << dDtStub[7][dS2] << ", " << abs(eDtStub[0][eS2]-dDtStub[0][dS2]) << ", " << ", " << eDtStub[6][eS2] << ", " << dDtStub[6][dS2] << endl;
        if( ((dDtStub[7][eS2]==-55) || (dDtStub[7][dS2]>(abs(eDtStub[0][eS2]-dDtStub[0][dS2]))) ) && (eDtStub[6][eS2]!=1) && (dDtStub[6][dS2]!=1)  )
        {
          //cout << "Imperfect match found" << endl;
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
	/*if(dtSMulti==true)
		cout << "Multiple DT stubs mapped to the same stub" << endl;
	if(dtUnmap!=0)
		cout << "Unmapped DT stubs:" << dtUnmap << endl;*/
	
  if(dtSMulti==false && dtUnmap==0)
  {
    for(int phil=0; phil<eDtCounter; phil++)
    {
      if(eDtStub[6][phil]==1 || eDtStub[6][phil]==2)
      {
        int indexFil = eDtStub[3][phil]*2+eDtStub[4][phil]-1;
        dtStubPhi->Fill(eDtStub[0][phil],  dDtStub[0][ eDtStub[5][phil] ]);
        (eDtStub[0][phil] ==  dDtStub[0][ eDtStub[5][phil] ]) ? dtStubPhi_1d->Fill(0) : dtStubPhi_1d->Fill(1);
        if( eDtStub[0][phil] != dDtStub[0][ eDtStub[5][phil] ])
					badDtStubSector->Fill(indexFil,eDtStub[2][phil]);
      }
    }
  }
  if(ptLUT_) delete ptLUT_;
}		
