/****************************************************************************
 *																		   																		*
 *   CSCTFEfficiencies.cc												   													*
 *   Original Code by L. Grey											   												*
 *   Version 2.0 written by J. Gartner									   									*
 *   V 2.0 - 1/17/08 == Basic efficiency plots and verbose track matching. 	*
 *																		   																		*
 *   For Questions or Comments email: joseph.anthony.gartner.iii@cern.ch   	*
 *   Please use this code for good, not awesome.						   							*
 *																		   																		*
 ***************************************************************************/

#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>

#include <L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h>
#include <L1Trigger/CSCTrackFinder/test/analysis/CSCTFEfficiencies.h>

#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/L1CSCTrackFinder/interface/TrackStub.h>
#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h>
#include <DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h>
#include <DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <TMath.h>
#include <TCanvas.h>
#include <TLorentzVector.h>

#include <TStyle.h>
#include <TLegend.h>
#include <TF1.h>
#include <TH2.h>

namespace csctf_analysis
{
  Double_t thresh(Double_t* pt, Double_t* par)
  {
    Double_t fitval = (0.5*TMath::Erf((pt[0]/par[0] + 1.0)/(TMath::Sqrt(2.0)*par[1])) +
		       0.5*TMath::Erf((pt[0]/par[0] - 1.0)/(TMath::Sqrt(2.0)*par[1])) )
           *(par[2] + par[3]*pt[0]);
    return fitval;
  }
}

CSCTFEfficiencies::CSCTFEfficiencies(edm::ParameterSet const& conf)
{
  cnttrk = 0;
  cntGen = 0;

  outFile = conf.getUntrackedParameter<std::string>("OutFile");//c
	
	my_dtrc = new CSCTFDTReceiver();
	
	bzero(srLUTs_,sizeof(srLUTs_));
  bool TMB07=true;
  edm::ParameterSet srLUTset;
  srLUTset.addUntrackedParameter<bool>("ReadLUTs", false);
  srLUTset.addUntrackedParameter<bool>("Binary",   false);
  srLUTset.addUntrackedParameter<std::string>("LUTPath", "./");
  for(int endcap = 1; endcap<=2; endcap++)
  {
		for(int sector=1; sector<=6; sector++)
		{
     	for(int station=1,fpga=0; station<=4 && fpga<5; station++)
     	{
 				if(station==1)
        	for(int subSector=0; subSector<2; subSector++)
        		srLUTs_[fpga++][sector-1][endcap-1] = new CSCSectorReceiverLUT(endcap, sector, subSector+1, station, srLUTset, TMB07);
   			else
				{
   				srLUTs_[fpga++][sector-1][endcap-1] = new CSCSectorReceiverLUT(endcap, sector, 0, station, srLUTset, TMB07);
				}
     	}
		}
  }
}

void CSCTFEfficiencies::DeleteHistos()
{
	if(simEta) delete simEta;
	if(simPhi) delete simPhi;
	if(simPt) delete simPt;
}

void CSCTFEfficiencies::beginJob()
{
	modeOcc = new TH1F("modeOcc","Mode Occupancy",15,1,16);

	simEta = new TH1F("simEta","Simulated Track #eta",500,0, 2.5); 
	simPhi = new TH1F("simPhi","Simulated Track #phi",128,0,6.4);
	simPt = new TH1F("simPt","Simulated Track Transverse Momentum", 140, 0, 7000);
	simPz = new TH1F("simPz", "Simulated Track Z Momentum", 140, 0, 7000);
	simEHalo = new TH1F("simEHalo", "Simulated Track Energy", 140, 0, 7000);
	simP =  new TH1F("simP", "Simulated Track Total Momentum", 140, 0, 7000);
	
	trackedEta = new TH1F("trackedEta","Track Finder #eta", 500, 0, 2.5);
	trackedPhi = new TH1F("trackedPhi","Track Finder #phi", 80, 0, 6.4);
	trackedPt = new TH1F("trackedPt","Track Finder Pt",101,-0.5,100.5);
	trackedBx = new TH1F("trackedBx","Track Finder Bx Assignment", 14, -6.5, 7.5);
	trackedEHalo = new TH1F("trackedEHalo","Simulated Track Energy for Halo Trigger Muons", 140, 0, 7000);
	trackedPtHalo = new TH1F("trackedPtHalo", "Tf Track Pt for Halo Trigger Muons", 141, -0.5, 140.5);
	HaloPRes = new TH1F("HaloPRes", "Momentum Resolution for Halo Tracks", 1000, -1, 1);
	
	matchedPhi = new TH1F("matchedPhi","Matched Track #phi", 128, 0, 6.4);
	matchedEta = new TH1F("matchedEta","Matched Track #eta",500,0, 2.5);
	matchedPt = new TH1F("matchedPt","Matched Track Pt",101,-0.5,100.5);
	matchedPt10 = new TH1F("matchedPt10","Matched Track Pt Tf > 10",101,-0.5,100.5); 
	matchedPt20 = new TH1F("matchedPt20","Matched Track Pt Tf > 20",101,-0.5,100.5);
	matchedPt40 = new TH1F("matchedPt40","Matched Track Pt Tf > 40",101,-0.5,100.5);
	matchedPt60 = new TH1F("matchedPt60","Matched Track Pt Tf > 60",101,-0.5,100.5);
	fidPtDen = new TH1F("fidPtDen", "Fiducial Sim Track Pt", 101,-0.5, 100.5);	
	
	EffPhi = new TH1F("EffPhi","Efficiency v #phi",128,0,6.4);
	EffPt = new TH1F("EffPt","Efficiency v Pt",101,-0.5,100.5);
	EffEn = new TH1F("EffEn", "Efficiency v Simualted Muon Eneregy",120, 1000, 7000);
	EffPt10 = new TH1F("EffPt10","Efficiency v Pt Tf > 10",101,-0.5,100.5);
	EffPt20 = new TH1F("EffPt20","Efficiency v Pt Tf > 20",101,-0.5,100.5);
	EffPt40 = new TH1F("EffPt40","Efficiency v Pt Tf > 40",101,-0.5,100.5);
	EffPt60 = new TH1F("EffPt60","Efficiency v Pt Tf > 60",101,-0.5,100.5);
	EffEtaAll = new TH1F("EffEtaAll","Efficiency v eta for all Tracks", 500, 0, 2.5);
	EffEtaQ3 = new TH1F("EffEtaQ3","Efficiency v #eta for Quality 3 Tracks", 500, 0, 2.5);
	EffEtaQ2 = new TH1F("EffEtaQ2","Efficiency v #eta for Quality 2 Tracks", 500, 0, 2.5);
	EffEtaQ1 = new TH1F("EffEtaQ1","Efficiency v #eta for Quality 1 Tracks", 500, 0, 2.5);	
	EtaQ3 = new TH1F("EtaQ3","Simulated Track #eta, Quality 3 Tracks",500,0,2.5);
	EtaQ2 = new TH1F("EtaQ2","Simulated Track #eta, Quality 2 Tracks",500,0,2.5);
	EtaQ1 = new TH1F("EtaQ1","Simulated Track #eta, Quality 1 Tracks",500,0,2.5);
	
	Radius = new TH1F("Radius","Angle between tf and sim track", 30, 0, 0.6);
	LostEta = new TH1F("LostEta","Sim #eta Value for Lost Tracks",500, 0, 2.5);
	LostPhi = new TH1F("LostPhi","Sim #phi Value for Lost Tracks",128, 0, 6.4);
	
	etaResolution = new TH1F("etaResolution","#eta Resolution", 1000, -1, 1);
	phiResolution = new TH1F("phiResolution","#phi Resolution", 1000, -1, 1);
	ptResolution = new TH1F("ptResolution","Pt Resolution for Quality > 1 Tracks", 300, -1.5, 1.5);
	ptResolutionQ3 = new TH1F("ptResolutionQ3","Pt Resolution for Quality > 2 Tracks", 300, -1.5, 1.5);
	ptResolutionEtaLow = new TH1F("ptResolutionEtaLow", "Pt Resolution for 1.2 < #eta < 1.6", 300, -1.5, 1.5);
	ptResolutionEtaHigh = new TH1F("ptResolutionEtaHigh","Pt Resolution for 1.6 < #eta < 2.1", 300, -1.5, 1.5);
	PhiResVPt = new TH2F("phiResVPt","#phi Resolution v Pt", 1000, -1, 1, 101,-0.5,100.5);
	PtResVPt = new TH2F("PtResVPt","Pt Resolution v Pt", 300, -1.5, 1.5, 101, -0.5,100.5);
	PtResVEta = new TH2F("PtResVEta", "PtResolution v #eta", 300, -1.5, 1.5, 500, 0.0, 2.5);
	
	ghostEta = new TH1F("ghostEta","Sim #eta Value for Fake Tracks",500, 0, 2.5);
	ghostPhi = new TH1F("ghostPhi","Sim #phi Value for Fake Tracks",128, 0, 6.4);
	ghostPt = new TH1F("ghostPt", "Sim Pt Value for Fake Tracks",301,-0.5,300.5);
	ghostRadius = new TH1F("ghostRadius", "Radius Value for Fake Tracks", 30, 0, 0.6);
	numEScat = new TH1F("numEScat", "Number of Scattered electrons for Ghost Tracks", 51, -0.5, 50.5);
	ghostDelPhi = new TH1F("ghostDelPhi", "Del #phi for Fake Tracks", 101, 0, 2.02); //0.0218
	ghostDelEta = new TH1F("ghostDelEta", "Del #eta for Fake Tracks", 200, 0, 1);
	ghostTrackRad = new TH1F("ghostTrackRad", "Angle between Tf and ghost Track", 30, 0, 0.6);	
	ghostselectPtRes = new TH1F("ghostselectPtRes","Pt Resolution for Smaller R TF on ghost event", 4000, -10, 30);
	ghostdropPtRes = new TH1F("ghostdropPtRes","Pt Resolution for Greater R TF on ghost event", 4000, -10, 30);
	
	simHaloPipeOff = new TH1F("simHaloPipeOff", "Simulated Value for muon offset", 100, 0, 100);
	simHaloPipeOff2 = new TH1F("simHaloPipeOff2", "Simulated Value for muon offset 2", 100, 0, 100);
	trackedHaloPipeOff = new TH1F("trkHaloPipeOff", "Simulated Value for Halo triggered muon offset", 100, 0, 100);
	EffHaloPipeOff = new TH1F("EffHaloPipeOff", "Efficiency Vs halo track radius", 100, 0, 100);
	LostHaloPipeOff = new TH1F("LostHaloPipeOff", "Simulated Value for muon offset for lost tracks", 100, 0, 100);
	simHaloPosition = new TH2F("simHaloPosition", "X-Y Distribution of Beam Halo Muons", 30, -30, 30, 30, -30, 30);
	trackHaloPosition = new TH2F("trackHaloPosition", "X-Y Distribution of Tf Beam Halo Muons", 30, -30, 30, 30, -30, 30);
	lostHaloPosition = new TH2F("lostHaloPosition", "X-Y Distribution of Lost Beam Halo Muons", 30, -30, 30, 30, -30, 30);
	
	dtStubBx = new TH1F("dtStubBx", "Stub timing from DT", 13, -6.5, 6.5);
	
	overDeleta12 = new TH1F("overDeleta12", "#Delta #eta_{12} lost Tracks overlap", 128, 0, 128);
	overDelphi12 = new TH1F("overDelphi12", "#Delta #phi_{12} lost Tracks overlap", 4096, 0, 4096);
	overDeleta25 = new TH1F("overDeleta25", "#Delta #eta_{25} lost Tracks overlap", 128, 0, 128);
	overDelphi25 = new TH1F("overDelphi25", "#Delta #phi_{25} lost Tracks overlap", 4096, 0, 4096);
	overDeleta15 = new TH1F("overDeleta15", "#Delta #eta_{15} lost Tracks overlap", 128, 0, 128);
	overDelphi15 = new TH1F("overDelphi15", "#Delta #phi_{15} lost Tracks overlap", 4096, 0, 4096);
	
	ghosts = 0;
	haloGhosts = 0;
	lostTracks = 0;
	haloTrigger = 0;
	DebugCounter = 0; //Counter to be moved randomly for debug purposes
}

void CSCTFEfficiencies::endJob()
{
	/////////////////////////
	//// Set Axis Labels ////
	/////////////////////////
	EffEn->GetXaxis()->SetTitle("Simulated Track Energy (GeV)");
	EffEn->GetYaxis()->SetTitle("Efficiency");
	EffEn->SetLineColor(3);
	EffEn->SetFillColor(9);
	EffHaloPipeOff->GetXaxis()->SetTitle("Radius");
	EffHaloPipeOff->GetYaxis()->SetTitle("Efficiency");
	EffHaloPipeOff->SetLineColor(3);
	EffHaloPipeOff->SetFillColor(9);
	EffPhi->GetXaxis()->SetTitle("#phi Sim (Rad)");
	EffPhi->GetYaxis()->SetTitle("Efficiency");
	PtResVPt->GetXaxis()->SetTitle("Pt_{Sim} / Pt_{Tf} - 1");
	PtResVPt->GetYaxis()->SetTitle("Pt_{Sim} (GeV/c)");
	PtResVEta->GetXaxis()->SetTitle("Pt_{Sim} / Pt_{Tf} - 1");
	PtResVEta->GetYaxis()->SetTitle("#eta_{Sim}");
	etaResolution->GetXaxis()->SetTitle("#eta_{Sim} - #eta_{Tf}");
	etaResolution->GetYaxis()->SetTitle("Count");
	phiResolution->GetXaxis()->SetTitle("#phi_{Sim} - #phi_{Tf} (Rad)");
	phiResolution->GetYaxis()->SetTitle("Count");
	ptResolution->GetXaxis()->SetTitle("Pt_{Sim} / Pt_{Tf} - 1");
	ptResolution->GetYaxis()->SetTitle("Count");
	Radius->GetXaxis()->SetTitle("Radius value");
	Radius->GetYaxis()->SetTitle("Count");
	ghostEta->GetXaxis()->SetTitle("#eta_{Sim}");
	ghostEta->GetYaxis()->SetTitle("Count");
	ghostPhi->GetXaxis()->SetTitle("#phi_{Sim}");
	ghostPhi->GetYaxis()->SetTitle("Count");
	ghostPt->GetXaxis()->SetTitle("Pt_{Sim} (GeV/c)");
	ghostPt->GetYaxis()->SetTitle("Count");
	simHaloPosition->GetXaxis()->SetTitle("X position");
	simHaloPosition->GetYaxis()->SetTitle("Y position");
	simHaloPosition->SetMarkerStyle(3);
	trackHaloPosition->SetMarkerStyle(3);
	lostHaloPosition->SetMarkerStyle(2);
	ptResolutionQ3->GetXaxis()->SetTitle("Pt_{Sim} / Pt_{Tf} - 1");
	ptResolutionQ3->GetYaxis()->SetTitle("Count");
	ptResolutionEtaLow->GetXaxis()->SetTitle("Pt_{Sim} / Pt_{Tf} - 1");
	ptResolutionEtaLow->GetYaxis()->SetTitle("Count");
	ptResolutionEtaHigh->GetXaxis()->SetTitle("Pt_{Sim} / Pt_{Tf} - 1");
	ptResolutionEtaHigh->GetYaxis()->SetTitle("Count");
	simEHalo->GetXaxis()->SetTitle("Simulated Muon Energy");
	simEHalo->GetYaxis()->SetTitle("Count");
	simPt->GetXaxis()->SetTitle("Pt_{Sim}");
	simPt->GetYaxis()->SetTitle("Count");
	simP->GetXaxis()->SetTitle("P_{Sim}");
	simP->GetYaxis()->SetTitle("Count");
	simPz->GetXaxis()->SetTitle("Pz_{Sim}");
	simPz->GetYaxis()->SetTitle("Count");
	
	/////////////////////
	//// Efficiency /////
	/////////////////////
	
	EffPt10->Divide(matchedPt10, fidPtDen);
	EffPt20->Divide(matchedPt20, fidPtDen);
	EffPt40->Divide(matchedPt40, fidPtDen);
	EffPt60->Divide(matchedPt60, fidPtDen);
	EffPt->Divide(matchedPt, fidPtDen);
	EffEtaQ3->Divide(EtaQ3, simEta);
	EffEtaQ2->Divide(EtaQ2, simEta);
	EffEtaQ1->Divide(EtaQ1, simEta);
	EffPhi->Divide(matchedPhi, simPhi);
		
	TCanvas* PtEffAll = new TCanvas("PtEffAll");
	TF1* fitThresh = new TF1("fitThresh", csctf_analysis::thresh, 0, 100, 4);
	fitThresh->SetParNames("Pt40","Resol","Constant","Slope");
	EffPt->GetXaxis()->SetTitle("Pt Sim (GeV/c)");
	EffPt->GetYaxis()->SetTitle("Efficiency");
	EffPt->SetMaximum(1.1);
	EffPt->SetTitle("Efficience for tf pt >0, >10, >20, >40, >60");
	
	EffPt->SetFillColor(1);
	//EffPt->Fit("fitThresh");
	//EffPt->SetMarkerColor(6);
	//EffPt->SetMarkerStyle(24);
	
	EffPt10->SetFillColor(2);
	//EffPt10->Fit("fitThresh");
	//EffPt10->SetMarkerColor(7);
	//EffPt10->SetMarkerStyle(23);
	
	EffPt20->SetFillColor(3);
	//EffPt20->Fit("fitThresh");
	//EffPt20->SetMarkerColor(8);
	//EffPt20->SetMarkerStyle(22);
	
	EffPt40->SetFillColor(4);
	//EffPt40->Fit("fitThresh");
	//EffPt40->SetMarkerColor(9);
	//EffPt40->SetMarkerStyle(21);
	
	EffPt60->SetFillColor(5);
	//EffPt60->Fit("fitThresh");
	//EffPt60->SetMarkerColor(2);
	//EffPt60->SetMarkerStyle(25);
	
	EffPt->Draw();
	EffPt10->Draw("same");
	EffPt20->Draw("same");
	EffPt40->Draw("same");
	EffPt60->Draw("same");
	TrackerLeg1 = new TLegend(0.4,0.6,0.8,0.8);
	TrackerLeg1->AddEntry(EffPt,"All Tracks","f");
	TrackerLeg1->AddEntry(EffPt10,"Pt_{TF} > 10","f");
	TrackerLeg1->AddEntry(EffPt20,"Pt_{TF} > 20","f");
	TrackerLeg1->AddEntry(EffPt40,"Pt_{TF} > 40","f");
	TrackerLeg1->AddEntry(EffPt60,"Pt_{TF} > 60","f");
	TrackerLeg1->Draw();
	PtEffAll->Print("EffPt");
	
	
	TCanvas* EtaEff = new TCanvas("EtaEff");
	EffEtaQ1->GetXaxis()->SetTitle("Eta Sim");
	EffEtaQ1->GetYaxis()->SetTitle("Efficiency");
	EffEtaQ1->SetMaximum(1.1);
	EffEtaQ1->SetTitle("Efficiency for Quality 1, 2, and 3 Tracks");
	EffEtaQ1->SetFillColor(1);
	EffEtaQ2->SetFillColor(4);
	EffEtaQ3->SetFillColor(3);
	EffEtaQ1->Draw();
	EffEtaQ2->Draw("same");
	EffEtaQ3->Draw("same");
	TrackerLeg2 = new TLegend(0.4,0.6,0.8,0.8);
	TrackerLeg2->AddEntry(EffEtaQ1,"All Tracks","f");
	TrackerLeg2->AddEntry(EffEtaQ2,"Quality > 1","f");
	TrackerLeg2->AddEntry(EffEtaQ3,"Quality > 2","f");
	TrackerLeg2->Draw();
	EtaEff->Print("EffEta");
	
	/////////////////////////////////
	//// Ghost & Lost Track Info ////
	/////////////////////////////////
	LostPhi->Add(simPhi,matchedPhi,1,-1);
	LostEta->Add(simEta,matchedEta,1,-1);
	lostHaloPosition->Add(simHaloPosition,trackHaloPosition,1,-1);
	LostHaloPipeOff->Add(trackedHaloPipeOff,simHaloPipeOff, -1,1);
	std::cout << "Ghosts: " << ghosts << std::endl << "Lost Tracks: " << lostTracks << std::endl;
	std::cout << "Halo bit set: " << haloTrigger << "x times." << std::endl;
	std::cout << "Halo Ghosts: " << haloGhosts << std::endl << std::endl;
	
	//////////////////////
	//// Print Histos ////
	//////////////////////
	fAnalysis = new TFile(outFile.c_str(), "RECREATE");//c
	TObjArray Hlist(0);
	Hlist.Add(modeOcc);
	Hlist.Add(dtStubBx);
	Hlist.Add(EffPt10);
	Hlist.Add(EffPt20);
	Hlist.Add(EffPt40);
	Hlist.Add(EffPt60);
	Hlist.Add(PtEffAll);
	Hlist.Add(simPz);
	Hlist.Add(EtaEff);
	Hlist.Add(trackedEHalo);
	Hlist.Add(trackedEta);
	Hlist.Add(trackedPhi);
	Hlist.Add(trackedPtHalo);
	Hlist.Add(HaloPRes);
	Hlist.Add(ghostDelPhi);
	Hlist.Add(ghostDelEta);
	Hlist.Add(ghostTrackRad);
	Hlist.Add(ghostselectPtRes);
	Hlist.Add(ghostdropPtRes);
	Hlist.Add(EffPt10);
	Hlist.Add(EffPt20);
	Hlist.Add(EffPt40);
	Hlist.Add(EffPt60);
	Hlist.Add(ghostPhi);
	Hlist.Add(ghostEta);
	Hlist.Add(ghostRadius);
	Hlist.Add(ghostPt);
	Hlist.Add(trackedBx);
	Hlist.Add(numEScat);
	Hlist.Add(EffPhi);
	Hlist.Add(EffPt);
	Hlist.Add(EffEn);
	Hlist.Add(simEHalo);
	Hlist.Add(simEta);
	Hlist.Add(simPhi);
	Hlist.Add(simPt);
	Hlist.Add(simP);
	Hlist.Add(Radius);
	Hlist.Add(LostPhi);
	Hlist.Add(LostEta);
	Hlist.Add(etaResolution);
	Hlist.Add(phiResolution);
	Hlist.Add(ptResolution);
	Hlist.Add(PhiResVPt); 
	Hlist.Add(PtResVPt);
	Hlist.Add(PtResVEta);
	Hlist.Add(ptResolutionQ3);
	Hlist.Add(ptResolutionEtaLow);
	Hlist.Add(ptResolutionEtaHigh);
	Hlist.Add(overDeleta12);
	Hlist.Add(overDelphi12);
	Hlist.Add(overDeleta25);
	Hlist.Add(overDelphi25);
	Hlist.Add(overDeleta15);
	Hlist.Add(overDelphi15);
	Hlist.Write();
	delete fAnalysis;
}

void CSCTFEfficiencies::analyze(edm::Event const& e, edm::EventSetup const& es)
{
	
  const float ptscale[33] = { 
  	-1.,   0.0,   1.5,   2.0,   2.5,   3.0,   3.5,   4.0,
    4.5,   5.0,   6.0,   7.0,   8.0,  10.0,  12.0,  14.0,  
    16.0,  18.0,  20.0,  25.0,  30.0,  35.0,  40.0,  45.0, 
    50.0,  60.0,  70.0,  80.0,  90.0, 100.0, 120.0, 140.0, 1.E6 };


	// get sim track and found track data here... then run analysis functions
	edm::Handle<edm::SimTrackContainer> simTracks;
	edm::Handle<std::vector<L1MuRegionalCand> > tfTracks;
	edm::Handle<std::vector<SimVertex> > simVertexs;
	e.getByLabel("joeOut","CSC", tfTracks);
	e.getByLabel("g4SimHits",simTracks);
	e.getByLabel("g4SimHits",simVertexs);
					
	//std::vector<L1MuRegionalCand>::const_iterator tfTrk;
	//for(tfTrk = tfTracks->begin(); tfTrk != tfTracks->end(); tfTrk++) 	
		
	// declare variables which must be defined outside of sim & tf loop for verbose track matching
	bool DEBUG = false;
	double R1 = 0.5, R2 = 0.5, R3 = 0.5, R4 = 0.5;
	double R1Old, R2Old, R3Old, R4Old;
	int tfLoop = 0, tfLoop2 = 0, simCounter = 0, firstMuon = 0;
	bool link1 = false, link2 = false, link3 = false,link4 = false;
	bool fid1 = false, fid2 = false, fid3 = false, fid4 = false, fid1Old = false, fid2Old = false, fid3Old = false, fid4Old = false; 
	bool pt10bit1 = false, pt20bit1 = false, pt40bit1 = false, pt60bit1 = false;
	bool pt10bit2 = false, pt20bit2 = false, pt40bit2 = false, pt60bit2 = false;
	bool pt10bit3 = false, pt20bit3 = false, pt40bit3 = false, pt60bit3 = false;
	bool pt10bit4 = false, pt20bit4 = false, pt40bit4 = false, pt60bit4 = false;
	bool LowResbit1 = false, LowResbit2 = false, LowResbit3 = false, LowResbit4 = false;
	bool pt10bit1Old = false, pt20bit1Old = false, pt40bit1Old = false, pt60bit1Old = false;
	bool pt10bit2Old = false, pt20bit2Old = false, pt40bit2Old = false, pt60bit2Old = false;
	bool pt10bit3Old = false, pt20bit3Old = false, pt40bit3Old = false, pt60bit3Old = false;
	bool pt10bit4Old = false, pt20bit4Old = false, pt40bit4Old = false, pt60bit4Old = false;
	double pairPtOld1, pairPt10Old1, pairPt20Old1, pairPt40Old1, pairPt60Old1, pairPhiOld1, pairEtaOld1;
	double pairPtOld2, pairPt10Old2, pairPt20Old2, pairPt40Old2, pairPt60Old2, pairPhiOld2, pairEtaOld2;
	double pairPtOld3, pairPt10Old3, pairPt20Old3, pairPt40Old3, pairPt60Old3, pairPhiOld3, pairEtaOld3;
	double pairPtOld4, pairPt10Old4, pairPt20Old4, pairPt40Old4, pairPt60Old4, pairPhiOld4, pairEtaOld4;
	double PtRes1OldHigh, PtRes2OldHigh, PtRes3OldHigh, PtRes4OldHigh, PtRes1OldLow, PtRes2OldLow, PtRes3OldLow, PtRes4OldLow;
	double T1Phi, T1Eta, T1Pt, T2Phi, T2Eta, T2Pt, T3Phi, T3Eta, T3Pt, T4Phi, T4Eta, T4Pt, gDelPhi, gDelEta, gRad;
	double EtaRes1, EtaRes2, EtaRes3, EtaRes4, PhiRes1, PhiRes2, PhiRes3, PhiRes4, PtRes1, PtRes2, PtRes3, PtRes4;
	double EtaRes1Old, EtaRes2Old, EtaRes3Old, EtaRes4Old, PhiRes1Old, PhiRes2Old, PhiRes3Old, PhiRes4Old, PtRes1Old, PtRes2Old, PtRes3Old, PtRes4Old;
	int mecBx, T1Bx, T2Bx, T3Bx, T4Bx, T1BxOld, T2BxOld, T3BxOld, T4BxOld, pair1Bx, pair2Bx, pair3Bx, pair4Bx;
	int T1Q, T2Q, T3Q, T4Q;
	int pairQ1, pairQ2, pairQ3, pairQ4;
	int simTrkCount = 0;
	double genEta =0;
	
	//Declare iterators for sim & tf loops
	edm::SimTrackContainer::const_iterator simTrk = simTracks->begin();
	std::vector<L1MuRegionalCand>::const_iterator tfTrk;
	std::vector<SimVertex>::const_iterator simVtx;
	for(; simTrk != simTracks->end(); simTrk++)  // Loop over all Simulated tracks for an Event
	{
		simTrkCount++;
		TLorentzVector mom;
		mom.SetPxPyPzE(simTrk->momentum().x(), simTrk->momentum().y(), simTrk->momentum().z(), simTrk->momentum().t());
		
		double genPhi = (mom.Phi() > 0) ? mom.Phi() : mom.Phi() + 2*M_PI;
		double genP = mom.P();
		
		if((genP >= 2) &&  (fabs(simTrk->type()) == 13) ) // Disclude electrons from scattering
		{
			///////////////////////////////
			//// Sim Track Information ////
			///////////////////////////////
			firstMuon++;	
			genEta = mom.PseudoRapidity();
			double genE = mom.E();
			double genPz = mom.Z();
			double genPt = mom.Pt();
			
			if( (genPt>2.0) && (genEta >= 0.9 ) )
			{
				simCounter++;
				simEta->Fill( fabs(genEta) );
				simPhi->Fill( genPhi );
				simPt->Fill( fabs(genPt) );
				simEHalo->Fill( genE );
				simPz->Fill( fabs(genPz) );
				simP->Fill( fabs(genP) );
				//simHaloPipeOff->Fill( haloPipeOffset );
				//simHaloPosition->Fill( genXpos, genYpos);
								
				
				tfLoop = 0;
				int firstMuonHalo = 0;
				
				int noShower = 0;

				
				if( ( genEta >= 1.2 ) && ( genEta <= 2.1 ) )
				{
					fidPtDen->Fill(genPt);
				}
				
				for(tfTrk = tfTracks->begin(); tfTrk != tfTracks->end(); tfTrk++)  // Loop over all Found tracks for an event
				{
					tfLoop++;
					
					///////////////////////////////////
					//// Machine Track Information ////
					///////////////////////////////////
					double mecPhi = (2.5*( tfTrk->phi_packed() ))*(M_PI)/180 + 0.0218;
					double mecEta = 0.9 + 0.05*( tfTrk->eta_packed() ) +0.025;
					int mecPtPacked = tfTrk->pt_packed();
					float mecPt = ptscale[mecPtPacked];
					int mecQuality = tfTrk->quality_packed();
					int mecChargePacked = tfTrk->charge_packed();
					int mecBx = tfTrk->bx();
					int mecCharge;
					int mecHalo = tfTrk->finehalo_packed();
					
					if( ( mecHalo != 0 ) && ( tfLoop == 1 ) )
					{
						//std::cout << std::endl << "Halo Trigger" << std::endl;
						//std::cout << std::endl << "Halo Val: " << mecHalo << "." << std::endl; 
						if( firstMuonHalo == 0)
						{
							haloTrigger++;
							trackedEHalo->Fill( genE );
							trackedPtHalo->Fill( mecPt );
							double ERes = genP/mecPt - 1;
							HaloPRes->Fill(ERes);
							firstMuonHalo = 1;
						}
					}
					
					if( ( tfLoop != 1 ) && ( firstMuonHalo == 1 ) ) haloGhosts++;
					
					if( mecChargePacked == 1) // Packed charge = 1 for -1 and 0 for +1
					{
						mecCharge = -1;
					}
					else
					{
						mecCharge = 1;
					}
					
					if( firstMuon == 1 )
					{
						trackedPhi->Fill(mecPhi);
						trackedPt->Fill(mecPt);
						trackedEta->Fill(mecEta);
						trackedBx->Fill(mecBx);
					}// Fill mec iformation first time a sim muon loop is done (Insures no double counting of found tracks).
					
					////////////////////////
					//// Track Matching ////
					////////////////////////
					if( tfLoop == 1 )
					{
						R1 = sqrt( ( mecPhi - genPhi )*( mecPhi - genPhi) + ( mecEta - genEta )*( mecEta - genEta ) );
						T1Phi = genPhi;
						T1Eta = genEta;
						T1Pt = genPt;
						T1Q = mecQuality;
						T1Bx = mecBx;
						
						EtaRes1 = genEta - mecEta;
						PhiRes1 = genPhi - mecPhi;
 
						if( (genEta >= 1.2) && (genEta <= 2.1) && ( mecQuality > 1 ) )
						{
							fid1 = true;

							if( mecPt >= 10) pt10bit1 = true;
							if( mecPt >= 20) pt20bit1 = true;
							if( mecPt >= 40) pt40bit1 = true;
							if( mecPt >= 60) pt60bit1 = true;
							
							PtRes1 = (genPt)/(mecPt) - 1;
							
						}  //pt Resolution very bad for Quality 1 tracks, so they are discluded.  
					} //first tf muon
					
					if( tfLoop == 2 )
					{
						R2 = sqrt( ( mecPhi - genPhi )*( mecPhi - genPhi) + ( mecEta - genEta )*( mecEta - genEta ) );
						T2Phi = genPhi;
						T2Eta = genEta;		
						T2Pt = genPt;
						T2Q = mecQuality;
						T2Bx = mecBx;
						
						EtaRes2 = genEta - mecEta;
						PhiRes2 = genPhi - mecPhi;
						
						if( (genEta >= 1.2) && (genEta <= 2.1) && ( mecQuality > 1 ) )
						{
							fid2 = true;
														
							if( mecPt >= 10) pt10bit2 = true;
							if( mecPt >= 20) pt20bit2 = true;
							if( mecPt >= 40) pt40bit2 = true;
							if( mecPt >= 60) pt60bit2 = true;
							
							PtRes2 = (genPt)/(mecPt) - 1;
						}
					} //second tf muon
					
					if( tfLoop == 3 )
					{
						R3 = sqrt( ( mecPhi - genPhi )*( mecPhi - genPhi) + ( mecEta - genEta )*( mecEta - genEta ) );
						T3Phi = genPhi;
						T3Eta = genEta;	
						T3Pt = genPt;
						T3Q = mecQuality;
						T3Bx = mecBx;
							
						EtaRes3 = genEta - mecEta;
						PhiRes3 = genPhi - mecPhi;
							
						if( (genEta >= 1.2) && (genEta <= 2.1) && ( mecQuality > 1 ) )
						{
							fid3 = true;
														
							if( mecPt >= 10) pt10bit3 = true;
							if( mecPt >= 20) pt20bit3 = true;
							if( mecPt >= 40) pt40bit3 = true;
							if( mecPt >= 60) pt60bit3 = true;

							PtRes3 = (genPt)/(mecPt) - 1;						
						}
					} //third tf muon
					
					if( tfLoop == 4 )
					{
						R4 = sqrt( ( mecPhi - genPhi )*( mecPhi - genPhi) + ( mecEta - genEta )*( mecEta - genEta ) );
						T4Phi = genPhi;
						T4Eta = genEta;	
						T4Pt = genPt;
						T4Q = mecQuality;	
						T4Bx = mecBx;
							
						EtaRes4 = genEta - mecEta;
						PhiRes4 = genPhi - mecPhi;	
						
						if( (genEta >= 1.2) && (genEta <= 2.1) && ( mecQuality > 1 ) )
						{
							fid4 = true;
							
							if( mecPt >= 10) pt10bit4 = true;
							if( mecPt >= 20) pt20bit4 = true;
							if( mecPt >= 40) pt40bit4 = true;
							if( mecPt >= 60) pt60bit4 = true;
						
							PtRes4 = (genPt)/(mecPt) - 1;
						}	
					}  //fourth tf muon

					if(DEBUG == true)
					{
						std::cout << std::endl << "Found Track Info"<< std::endl;
						std::cout << "Quality, Charge, pt, phi, eta:"<< std::endl; 
						std::cout << mecQuality << ", " << mecCharge  << ", " << mecPt << ", "<< mecPhi << ", " << mecEta << ", " << std::flush << std::endl;
						std::cout << "R1: " << R1 << ".  R2: " << R2 << ".  R3: " << R3 << ". R4: " << R4 << "."<< std::endl;
					}
					
					
				}//tfTrack loop
				
				if (tfLoop == 0) std::cout << "Sim info for lost track (Pt, eta, phi): " << genPt << ", " << genEta << ", " << genPhi << "." << std::endl;
				
				////////////////////////////
				//// Matched Track Info ////
				////////////////////////////

				if( (R1 < R2) && (R1 < R3) && (R1 < R4) )
				{
					if( (link1 == true) && ( R1 < R1Old) )
					{	
						R1Old = R1;
						pairPhiOld1 = T1Phi;
						pairEtaOld1 = T1Eta;
						pairQ1 = T1Q;
						pair1Bx = T1Bx;
						
						EtaRes1Old = EtaRes1;
						PhiRes1Old = PhiRes1;
						
						fid1Old = false;
						
						if (fid1 == true)
						{
							pairPtOld1 = T1Pt;
							PtRes1Old = PtRes1;
							
							if( T1Eta < 1.6 )
							{
								PtRes1OldLow = PtRes1;
								LowResbit1 = true;
							} else {
								PtRes1OldHigh = PtRes1;
							}
							

							if( pt10bit1 == true ) 
							{
								pairPt10Old1 = T1Pt;
								pt10bit1Old = true;
							}
							
							if( pt20bit1 == true ) 
							{
								pairPt20Old1 = T1Pt;
								pt20bit1Old = true;
							}
							
							if( pt40bit1 == true ) 
							{
								pairPt40Old1 = T1Pt;
								pt40bit1Old = true;
							}
							
							if( pt60bit1 == true ) 
							{
								pairPt60Old1 = T1Pt;
								pt60bit1Old = true;
							}
							
							fid1Old = true;
						}
					}
					
					if(link1 == false)
					{
						R1Old = R1;
						pairPhiOld1 = T1Phi;
						pairEtaOld1 = T1Eta;
						pairQ1 = T1Q;
						pair1Bx = T1Bx;
						EtaRes1Old = EtaRes1;
						PhiRes1Old = PhiRes1;
						
						if (fid1 == true)
						{
							PtRes1Old = PtRes1;
							pairPtOld1 = T1Pt;
							
							if( T1Eta < 1.6 )
							{
								PtRes1OldLow = PtRes1;
								LowResbit1 = true;
							} else {
								PtRes1OldHigh = PtRes1;
							}
							
							if( pt10bit1 == true ) 
							{
								pairPt10Old1 = T1Pt;
								pt10bit1Old = true;
							}
							
							if( pt20bit1 == true ) 
							{
								pairPt20Old1 = T1Pt;
								pt20bit1Old = true;
							}
							
							if( pt40bit1 == true ) 
							{
								pairPt40Old1 = T1Pt;
								pt40bit1Old = true;
							}
							
							if( pt60bit1 == true ) 
							{
								pairPt60Old1 = T1Pt;
								pt60bit1Old = true;
							}
							
							fid1Old = true;
						}
					}
					
					link1 = true;
				}//R1 Smallest
				
				if( (R2 < R1) && (R2 < R3) && (R2 < R4) )
				{	
					if( (link2 == true) && ( R2 < R2Old) )
					{	
						R2Old = R2;
						pairPhiOld2 = T2Phi;
						pairEtaOld2 = T2Eta;
						pairPtOld2 = T2Pt;
						pairQ2 = T2Q;
						pair2Bx = T2Bx;
						
						EtaRes2Old = EtaRes2;
						PhiRes2Old = PhiRes2;
						fid2Old = false;
						
						if (fid2 == true)
						{
							pairPtOld2 = T2Pt;
							PtRes2Old = PtRes2;
							
							if( T2Eta < 1.6 )
							{
								PtRes2OldLow = PtRes2;
								LowResbit2 = true;
							} else {
								PtRes2OldHigh = PtRes2;
							}
							
							if( pt10bit2 == true ) 
							{
								pairPt10Old2 = T2Pt;
								pt10bit2Old = true;
							}
							
							if( pt20bit2 == true )
							{
								pairPt20Old2 = T2Pt;
								pt20bit2Old = true;
							}
							
							if( pt40bit2 == true )
							{
								pairPt40Old2 = T2Pt;
								pt40bit2Old = true;
							}
							
							if( pt60bit2 == true ) 
							{
								pairPt60Old2 = T2Pt;
								pt60bit2Old = true;
							}
														
							fid2Old = true;
						}
					}
					
					if(link2 == false)
					{
						R2Old = R2;
						pairPhiOld2 = T2Phi;
						pairEtaOld2 = T2Eta;
						pairPtOld2 = T2Pt;
						pairQ2 = T2Q;
						pair2Bx = T2Bx;
						
						EtaRes2Old = EtaRes2;
						PhiRes2Old = PhiRes2;
						
						if (fid2 == true)
						{
							pairPtOld2 = T2Pt;
							PtRes2Old = PtRes2;
							
							if( T2Eta < 1.6 )
							{
								PtRes2OldLow = PtRes2;
								LowResbit2 = true;
							} else {
								PtRes2OldHigh = PtRes2;
							}
							
							if( pt10bit2 == true ) 
							{
								pairPt10Old2 = T2Pt;
								pt10bit2Old = true;
							}
							
							if( pt20bit2 == true ) 
							{
								pairPt20Old2 = T2Pt;
								pt20bit2Old = true;
							}
							
							if( pt40bit2 == true ) 
							{
								pairPt40Old2 = T2Pt;
								pt40bit2Old = true;
							}
							
							if( pt60bit2 == true ) 
							{
								pairPt60Old2 = T2Pt;
								pt60bit2Old = true;
							}
								
							
							fid2Old = true;
						}
					}
					
					link2 = true;
				}//R2 Smallest
				
				if( (R3 < R1) && (R3 < R2) && (R3 < R4) )
				{			
					if( (link3 == true) && ( R3 < R3Old) )
					{	
						R3Old = R3;
						pairPhiOld3 = T3Phi;
						pairEtaOld3 = T3Eta;
						pairPtOld3 = T3Pt;
						pairQ3 = T3Q;
						pair3Bx = T3Bx;
						
						EtaRes3Old = EtaRes3;
						PhiRes3Old = PhiRes3;

						fid3Old = false;
						
						if (fid3 == true)
						{
							pairPtOld3 = T3Pt;
							PtRes3Old = PtRes3;
							
							if( T3Eta < 1.6 )
							{
								PtRes3OldLow = PtRes3;
								LowResbit3 = true;
							} else {
								PtRes3OldHigh = PtRes3;
							}							
							
							if( pt10bit3 == true ) 
							{
								pairPt10Old3 = T3Pt;
								pt10bit3Old = true;
							}
							
							if( pt20bit3 == true ) 
							{
								pairPt20Old3 = T3Pt;
								pt20bit3Old = true;
							}
							
							if( pt40bit3 == true ) 
							{
								pairPt40Old3 = T3Pt;
								pt40bit3Old = true;
							}
							
							if( pt60bit3 == true ) 
							{
								pairPt60Old3 = T3Pt;
								pt60bit3Old = true;
							}
							
							fid3Old = true;
						}
					}
					
					if(link3 == false)
					{
						R3Old = R3;
						pairPhiOld3 = T3Phi;
						pairEtaOld3 = T3Eta;
						pairPtOld3 = T3Pt;
						pairQ3 = T3Q;
						pair3Bx = T3Bx;
						
						EtaRes3Old = EtaRes3;
						PhiRes3Old = PhiRes3;
						
						if (fid3 == true)
						{
							pairPtOld3 = T3Pt;
							PtRes3Old = PtRes3;
							
							if( T3Eta < 1.6 )
							{
								PtRes3OldLow = PtRes3;
								LowResbit3 = true;
							} else {
								PtRes3OldHigh = PtRes3;
							}
							
							if( pt10bit3 == true ) 
							{
								pairPt10Old3 = T3Pt;
								pt10bit3Old = true;
							}
							
							if( pt20bit3 == true ) 
							{
								pairPt20Old3 = T3Pt;
								pt20bit3Old = true;
							}
							
							if( pt40bit3 == true ) 
							{
								pairPt40Old3 = T3Pt;
								pt40bit3Old = true;
							}
							
							if( pt60bit3 == true )
							{
								pairPt60Old3 = T3Pt;
								pt60bit3Old = true;
							}
							
							fid3Old = true;
						}
					}
					
					link3 = true;
				}//R3 Smallest
				
				if( (R4 < R1) && (R4 < R2) && (R4 < R3) )
				{
					if( (link4 == true) && ( R4 < R4Old) )
					{	
						R4Old = R4;
						pairPhiOld4 = T4Phi;
						pairEtaOld4 = T4Eta;
						pairPtOld4 = T4Pt;
						pairQ4 = T4Q;
						pair4Bx = T4Bx;
						
						EtaRes4Old = EtaRes4;
						PhiRes4Old = PhiRes4;
						fid4Old = false;
						
						if (fid4 == true)
						{
							pairPtOld4 = T4Pt;
							PtRes4Old = PtRes4;
							
							if( T4Eta < 1.6 )
							{
								PtRes4OldLow = PtRes4;
								LowResbit4 = true;
							} else {
								PtRes4OldHigh = PtRes4;
							}
							
							if( pt10bit4 == true ) 
							{
								pairPt10Old4 = T4Pt;
								pt10bit4Old = true;
							}
							
							if( pt20bit4 == true ) 
							{
								pairPt20Old4 = T4Pt;
								pt20bit4Old = true;
							}
							
							if( pt40bit4 == true ) 
							{
								pairPt40Old4 = T4Pt;
								pt40bit4Old = true;
							}
							
							if( pt60bit4 == true ) 
							{
								pairPt60Old4 = T4Pt;
								pt60bit4Old = true;
							}
							
							fid4Old = true;
						}
					}
					
					if(link4 == false)
					{
						R4Old = R4;
						pairPhiOld4 = T4Phi;
						pairEtaOld4 = T4Eta;
						pairPtOld4 = T4Pt;
						pairQ4 = T4Q;
						pair4Bx = T4Bx;
						
						EtaRes4Old = EtaRes4;
						PhiRes4Old = PhiRes4;
						
						if (fid4 == true)
						{
							pairPtOld4 = T4Pt;
							PtRes4Old = PtRes4;
							
							if( T4Eta < 1.6 )
							{
								PtRes4OldLow = PtRes4;
								LowResbit4 = true;
							} else {
								PtRes4OldHigh = PtRes4;
							}							
							
							if( pt10bit4 == true ) 
							{
								pairPt10Old4 = T4Pt;
								pt10bit4Old = true;
							}
							
							if( pt20bit4 == true ) 
							{
								pairPt20Old4 = T4Pt;
								pt20bit4Old = true;
							}
							
							if( pt40bit4 == true ) 
							{
								pairPt40Old4 = T4Pt;
								pt40bit4Old = true;
							}
							
							if( pt60bit4 == true ) 
							{
								pairPt60Old4 = T4Pt;
								pt60bit4Old = true;
							}
							
							fid4Old = true;
						}
					}
					
					link4 = true;
				}//R4 Smallest
							
			R1 = 0.5, R2 = 0.5, R3 = 0.5, R4 = 0.5;
			fid1 = false, fid2 = false, fid3 = false, fid4 = false;
			pt10bit1 = false, pt10bit2 = false, pt10bit3 = false, pt10bit4 = false;
			pt20bit1 = false, pt20bit2 = false, pt20bit3 = false, pt20bit4 = false;
			pt40bit1 = false, pt40bit2 = false, pt40bit3 = false, pt40bit4 = false;
			pt60bit1 = false, pt60bit2 = false, pt60bit3 = false, pt60bit4 = false;
			
			}//sim track momentum >= 10
	    
			if(DEBUG == true)
			{
				std::cout << "Link1: " << link1 << ".  Link2: " << link2 << ".  Link3: " << link3 << ".  Link4: " << link4 << "." << std::endl;
				std::cout << std::endl << "####### End Event########" << std::endl;
			}
		
		} // |partid| == 13
    } // simTrk loop
	
	/////////////////////////////
	//// Fill Matched Histos ////
	/////////////////////////////
	if( link1 == true )
	{
		if(DEBUG == true)
		{
			std::cout << "---------- Link 1 Fill -------------" << std::endl; 
		}
		
		matchedPhi->Fill(pairPhiOld1);
		matchedEta->Fill(pairEtaOld1);

		if(fid1Old == true)
		{
			matchedPt->Fill(pairPtOld1);
			ptResolution->Fill(PtRes1Old);
			PtResVPt->Fill(PtRes1Old,pairPtOld1);
			PtResVEta->Fill(PtRes1Old,pairEtaOld1);
		
			if( pt10bit1Old == true ) matchedPt10->Fill(pairPt10Old1);
			if( pt20bit1Old == true ) matchedPt20->Fill(pairPt20Old1);
			if( pt40bit1Old == true ) matchedPt40->Fill(pairPt40Old1);
			if( pt60bit1Old == true ) matchedPt60->Fill(pairPt60Old1);
			
			if( LowResbit1 == true )
			{
				ptResolutionEtaLow->Fill(PtRes1OldLow);
			} else {
				ptResolutionEtaHigh->Fill(PtRes1OldHigh);
			}
			
		}
		
		Radius->Fill(R1Old);
		etaResolution->Fill(EtaRes1Old);
		phiResolution->Fill(PhiRes1Old);
		PhiResVPt->Fill(PhiRes1Old,pairPtOld1);

		
		if( pairQ1 >= 1)
		{
			EtaQ1->Fill(pairEtaOld1);
		}
		
		if( pairQ1 >= 2)
		{	
			EtaQ2->Fill(pairEtaOld1);
		}
		if( pairQ1 >= 3)
		{
			EtaQ3->Fill(pairEtaOld1);
			if(fid1Old == true) ptResolutionQ3->Fill(PtRes1Old);
		}
		
	}//link1 == true
	
	if( link2 == true )
	{				
		if(DEBUG == true)
		{
			std::cout << "---------- Link 2 Fill -------------" << std::endl; 
		}
		
		matchedPhi->Fill(pairPhiOld2);
		matchedEta->Fill(pairEtaOld2);
		if(fid2Old == true)
		{
			matchedPt->Fill(pairPtOld2);
			ptResolution->Fill(PtRes2Old);
			PtResVPt->Fill(PtRes2Old,pairPtOld2);
			PtResVEta->Fill(PtRes2Old,pairEtaOld2);
			
			if( pt10bit2Old == true ) matchedPt10->Fill(pairPt10Old2);
			if( pt20bit2Old == true ) matchedPt20->Fill(pairPt20Old2);
			if( pt40bit2Old == true ) matchedPt40->Fill(pairPt40Old2);
			if( pt60bit2Old == true ) matchedPt60->Fill(pairPt60Old2);
			
			if( LowResbit2 == true )
			{
				ptResolutionEtaLow->Fill(PtRes2OldLow);
			} else {
				ptResolutionEtaHigh->Fill(PtRes2OldHigh);
			}
		}
		Radius->Fill(R2Old);
		etaResolution->Fill(EtaRes2Old);
		phiResolution->Fill(PhiRes2Old);
		PhiResVPt->Fill(PhiRes2Old,pairPtOld2);
						
		if( pairQ2 >= 1)
		{
			EtaQ1->Fill(pairEtaOld2);
		}
		
		if( pairQ2 >= 2)
		{	
			EtaQ2->Fill(pairEtaOld2);
		}
		if( pairQ2 >= 3)
		{
			EtaQ3->Fill(pairEtaOld2);
			if(fid2Old == true) ptResolutionQ3->Fill(PtRes2Old);
		}
		
	}//link2 == true
	
	if( link3 == true )
	{
		if(DEBUG == true)
		{
			std::cout << "---------- Link 3 Fill -------------" << std::endl; 
		}
		
	
		matchedPhi->Fill(pairPhiOld3);
		matchedEta->Fill(pairEtaOld3);
		if(fid3Old == true)
		{
			matchedPt->Fill(pairPtOld3);
			ptResolution->Fill(PtRes3Old);
			PtResVPt->Fill(PtRes3Old,pairPtOld3);
			PtResVEta->Fill(PtRes3Old,pairEtaOld3);
			
			if( pt10bit3Old == true ) matchedPt10->Fill(pairPt10Old3);
			if( pt20bit3Old == true ) matchedPt20->Fill(pairPt20Old3);
			if( pt40bit3Old == true ) matchedPt40->Fill(pairPt40Old3);
			if( pt60bit3Old == true ) matchedPt60->Fill(pairPt60Old3);
			
			if( LowResbit3 == true )
			{
				ptResolutionEtaLow->Fill(PtRes3OldLow);
			} else {
				ptResolutionEtaHigh->Fill(PtRes3OldHigh);
			}
		}
		Radius->Fill(R3Old);
		etaResolution->Fill(EtaRes3Old);
		phiResolution->Fill(PhiRes3Old);
		PhiResVPt->Fill(PhiRes3Old,pairPtOld3);
		
		if( pairQ3 >= 1)
		{
			EtaQ1->Fill(pairEtaOld3);
		}
		
		if( pairQ3 >= 2)
		{	
			EtaQ2->Fill(pairEtaOld3);
		}
		if( pairQ3 >= 3)
		{
			EtaQ3->Fill(pairEtaOld3);
			if(fid3Old == true) ptResolutionQ3->Fill(PtRes3Old);
		}
	}// link3 == true
	
	if( link4 == true )
	{
		if(DEBUG == true)
		{
			std::cout << "---------- Link 4 Fill -------------" << std::endl; 
		}
	
		matchedPhi->Fill(pairPhiOld4);
		matchedEta->Fill(pairEtaOld4);
		
		if(fid4Old == true)
		{
			matchedPt->Fill(pairPtOld4);
			ptResolution->Fill(PtRes4Old);
			PtResVPt->Fill(PtRes4Old,pairPtOld4);
			PtResVEta->Fill(PtRes4Old,pairEtaOld4);
			
			if( pt10bit4Old == true ) matchedPt10->Fill(pairPt10Old4);
			if( pt20bit4Old == true ) matchedPt20->Fill(pairPt20Old4);
			if( pt40bit4Old == true ) matchedPt40->Fill(pairPt40Old4);
			if( pt60bit4Old == true ) matchedPt60->Fill(pairPt60Old4);
			
			if( LowResbit4 == true )
			{
				ptResolutionEtaLow->Fill(PtRes4OldLow);
			} else {
				ptResolutionEtaHigh->Fill(PtRes4OldHigh);
			}
		}
		
		Radius->Fill(R4Old);
		etaResolution->Fill(EtaRes4Old);
		phiResolution->Fill(PhiRes4Old);
		PhiResVPt->Fill(PhiRes4Old,pairPtOld4);

		
		if( pairQ4 >= 1)
		{
			EtaQ1->Fill(pairEtaOld4);
		}
		
		if( pairQ4 >= 2)
		{	
			EtaQ2->Fill(pairEtaOld4);
		}
		if( pairQ4 >= 3)
		{
			EtaQ3->Fill(pairEtaOld4);
			if(fid4Old == true) ptResolutionQ3->Fill(PtRes4Old);
		}
	}// link4 == true
	
	////////////////////////////
	//// Lost & Fake Tracks ////
	////////////////////////////
	
	if( tfLoop >= (simCounter +1))
	{
		ghosts++;
		int bunches1, bunches2;
		
		/////////////////////////////
		//// Fake Track Plotting ////
		/////////////////////////////
		
		//Reset iterators for tf loops
		
		int electronCount;
		double ghostPhi1, ghostPhi2, ghostEta1, ghostEta2;
		double ghostPtRes1, ghostPtRes2;
		int ghostQ1, ghostQ2;
		
		tfTrk = tfTracks->begin();
		for(; tfTrk != tfTracks->end(); tfTrk++)  // Loop over all Found tracks for an event
		{
			tfLoop2++;
			double mecPhi = (2.5*( tfTrk->phi_packed() ))*(M_PI)/180 + 0.0218;
			double mecEta = 0.9 + 0.05*( tfTrk->eta_packed() ) +0.025;
			int mecPtPacked = tfTrk->pt_packed();
			float mecPt = ptscale[mecPtPacked];
			int mecQuality = tfTrk->quality_packed();
			int mecChargePacked = tfTrk->charge_packed();
			int mecBx = tfTrk->bx();
			int mecCharge;
			electronCount = 0;
		
			if( tfLoop2 == 1 )
			{
				ghostPhi1 = mecPhi;
				ghostEta1 = mecEta;
			}
			
			if( tfLoop2 == 2 )
			{
				ghostPhi2 = mecPhi;
				ghostEta2 = mecEta;
			}
		
			simTrk = simTracks->begin();
			for(; simTrk != simTracks->end(); simTrk++)  // Loop over all Simulated tracks for an Event
			{
				TLorentzVector mom;
				mom.SetPxPyPzE(simTrk->momentum().x(), simTrk->momentum().y(), simTrk->momentum().z(), simTrk->momentum().t());
				
				if( fabs(simTrk->type()) != 13 )
				{
					electronCount++;
				}
			
				if( fabs(simTrk->type()) == 13 ) // Disclude electrons from scattering
				{
					double genPhi = (mom.Phi() > 0) ? mom.Phi() : mom.Phi() + 2*M_PI;
					double genEta = mom.PseudoRapidity();
					double genPt = mom.Pt();
					if( tfLoop2 == 1 )
					{
						R1 = sqrt( ( mecPhi - genPhi )*( mecPhi - genPhi) + ( mecEta - genEta )*( mecEta - genEta ) );
						T1Phi = genPhi;
						T1Eta = genEta;
						T1Pt = genPt;
						bunches1 = mecBx;
						ghostPtRes1 = ( 1/( mecPt ) - 1/( genPt ) )/(1/( genPt )) ;
						ghostQ1 = mecQuality;
					}
					
					if( tfLoop2 == 2 )
					{
						R2 = sqrt( ( mecPhi - genPhi )*( mecPhi - genPhi) + ( mecEta - genEta )*( mecEta - genEta ) );
						T2Phi = genPhi;
						T2Eta = genEta;
						T2Pt = genPt;
						bunches2 = mecBx;
						ghostPtRes1 = ( 1/( mecPt ) - 1/( genPt ) )/(1/( genPt )) ;
						ghostQ2 = mecQuality;
					}
				}//muon Cut
			}//sim Loop
		}//tf Loop

		gDelPhi = fabs(ghostPhi1 - ghostPhi2);
		gDelEta = fabs(ghostEta1 - ghostEta2);
		gRad = sqrt( ( gDelPhi )*( gDelPhi ) + ( gDelEta )*( gDelEta ) );

		
		
		ghostDelPhi->Fill(gDelPhi);
		ghostDelEta->Fill(gDelEta);
		ghostTrackRad->Fill(gRad);
		numEScat->Fill(electronCount);
		
		if( R1 < R2 )
		{
			ghostEta->Fill(T2Eta);
			ghostPhi->Fill(T2Phi);
			ghostPt->Fill(T2Pt);
			ghostRadius->Fill(R2);
		}
		
		if( ghostQ1 <= ghostQ2)
		{
			ghostselectPtRes->Fill(ghostPtRes2);
			ghostdropPtRes->Fill(ghostPtRes1);
		
		}
		
		if( R2 < R1 )
		{
			ghostEta->Fill(T1Eta);
			ghostPhi->Fill(T1Phi);
			ghostPt->Fill(T1Pt);
			ghostRadius->Fill(R1);
		}
		
		if( ghostQ2 < ghostQ1)
		{
			ghostselectPtRes->Fill(ghostPtRes1);
			ghostdropPtRes->Fill(ghostPtRes2);
		
		}
		
		if( R2 == R1)
		{
			std::cout << "R1: " << R1 << ".  Bx1: " << bunches1 << ".  Bx2: " << bunches2 << "." << std::endl;
			
		}
		if(DEBUG == true)
		{
			std::cout << std::endl << "---------- Fake Track ------------" << std::flush << std::endl;
		}
	}
	
	edm::Handle<L1CSCTrackCollection> tracks;
  e.getByLabel("joeTrackOut",tracks);
  for(L1CSCTrackCollection::const_iterator trk=tracks->begin(); trk<tracks->end(); trk++)
	{
		long LUTAdd = trk->first.ptLUTAddress();
		int trigMode = ( (LUTAdd)&0xf0000 ) >> 16;
		modeOcc->Fill(trigMode);
	}
}
