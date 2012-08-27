#include "L1Trigger/CSCTrackFinder/test/src/EffHistogramList.h"
#include "L1Trigger/CSCTrackFinder/test/src/TrackHistogramList.h"
#include <TMath.h>
#include <iostream>

namespace csctf_analysis
{

EffHistogramList::EffHistogramList(const std::string dirname, const edm::ParameterSet* parameters)
{
	TFileDirectory dir = fs->mkdir(dirname);
	TFileDirectory ptSubdir = dir.mkdir("Pt_Efficiencies");
	TFileDirectory ptSubdirOverall = ptSubdir.mkdir("Overall");
	TFileDirectory ptSubdirCSCOnly = ptSubdir.mkdir("CSCOnly");
	TFileDirectory ptSubdirCSCRestricted = ptSubdir.mkdir("CSCRestricted");
	TFileDirectory ptSubdirDTOnly = ptSubdir.mkdir("DTOnly");
	TFileDirectory ptSubdirOverlap = ptSubdir.mkdir("Overlap");
	TFileDirectory ptSubdirHighEta = ptSubdir.mkdir("HighEta");
	
	TFileDirectory etaSubdir = dir.mkdir("Eta_Efficiency");
	TFileDirectory phiSubdir = dir.mkdir("Phi_Efficiency");
    	
	double maxpt=parameters->getUntrackedParameter<double>("MaxPtHist");
	double minpt=parameters->getUntrackedParameter<double>("MinPtHist");
	int ptbins=parameters->getUntrackedParameter<double>("BinsPtHist");
	PtEffStatsFilename=parameters->getUntrackedParameter<std::string>("PtEffStatsFilename");
	
	std::string histoDescription = parameters->getUntrackedParameter<std::string>("HistoDescription");
	latexDescription = new TLatex(0.121,0.883,histoDescription.c_str());

	latexDescription->SetTextAlign(13);
	latexDescription->SetNDC();

    	EffPhi = phiSubdir.make<TH1F>("EffPhi","Efficiency v #phi",144,0,6.283);
    	EffPhi_mod_10_Q2_endcap1 = phiSubdir.make<TH1F>("EffPhi_mod_10_Q2_endcap1","Efficiency v #phi mod 10,Q>=2, Endcap 1",140,-2,12);
    	EffPhi_mod_10_Q3_endcap1 = phiSubdir.make<TH1F>("EffPhi_mod_10_Q3_endcap1","Efficiency v #phi mod 10,Q>=3, Endcap 1",140,-2,12);
    	EffPhi_mod_10_Q2_endcap2 = phiSubdir.make<TH1F>("EffPhi_mod_10_Q2_endcap2","Efficiency v #phi mod 10,Q>=2, Endcap 2",140,-2,12);
    	EffPhi_mod_10_Q3_endcap2 = phiSubdir.make<TH1F>("EffPhi_mod_10_Q3_endcap2","Efficiency v #phi mod 10,Q>=3, Endcap 2",140,-2,12);
    	//EffPt = dir.make<TH1F>("EffPt","Efficiency v Pt; 1.2 <= #eta <= 2.1",ptbins, minpt, maxpt);
    	EffPtDTOnly = ptSubdirDTOnly.make<TH1F>("EffPtDTOnly","Efficiency v Pt; 0<= #eta <=0.9",ptbins, minpt, maxpt);
    	EffPtCSCOnly = ptSubdirCSCOnly.make<TH1F>("EffPtCSCOnly","Efficiency v Pt; 1.2<= #eta <=2.4",ptbins, minpt, maxpt);
    	EffPtCSCRestricted = ptSubdirCSCRestricted.make<TH1F>("EffPtCSCRestricted","Efficiency v Pt; 1.2<= #eta <=2.1",ptbins, minpt, maxpt);
	EffPtOverall = ptSubdirOverall.make<TH1F>("EffPtOverall","Efficiency v Overall Pt",ptbins, minpt, maxpt);
    	EffPtOverlap = ptSubdirOverlap.make<TH1F>("EffPtOverlap","Efficiency v Pt; 0.9<= #eta <=1.2",ptbins, minpt, maxpt);
    	EffPtHighEta = ptSubdirHighEta.make<TH1F>("EffPtHighEta","Efficiency v Pt; 2.1<= #eta",ptbins, minpt, maxpt);
    	EffTFPt10Overall = ptSubdirOverall.make<TH1F>("EffTFPt10Overall","Efficiency v Overall Pt Tf > 10",ptbins, minpt, maxpt);
	EffTFPt12Overall = ptSubdirOverall.make<TH1F>("EffTFPt12Overall","Efficiency v Overall Pt Tf > 12",ptbins, minpt, maxpt);
    	EffTFPt16Overall = ptSubdirOverall.make<TH1F>("EffTFPt16Overall","Efficiency v Overall Pt Tf > 16",ptbins, minpt, maxpt);
	EffTFPt20Overall = ptSubdirOverall.make<TH1F>("EffTFPt20Overall","Efficiency v Overall Pt Tf > 20",ptbins, minpt, maxpt);
    	EffTFPt40Overall = ptSubdirOverall.make<TH1F>("EffTFPt40Overall","Efficiency v Overall Pt Tf > 40",ptbins, minpt, maxpt);
    	EffTFPt60Overall = ptSubdirOverall.make<TH1F>("EffTFPt60Overall","Efficiency v Overall Pt Tf > 60",ptbins, minpt, maxpt);
    	EffTFPt10CSCOnly = ptSubdirCSCOnly.make<TH1F>("EffTFPt10CSCOnly","Efficiency v Pt Tf > 10; 1.2<= #eta <=2.4",ptbins, minpt, maxpt);
	EffTFPt12CSCOnly = ptSubdirCSCOnly.make<TH1F>("EffTFPt12CSCOnly","Efficiency v Pt Tf > 12; 1.2<= #eta <=2.4",ptbins, minpt, maxpt);
	EffTFPt16CSCOnly = ptSubdirCSCOnly.make<TH1F>("EffTFPt16CSCOnly","Efficiency v Pt Tf > 16; 1.2<= #eta <=2.4",ptbins, minpt, maxpt);
    	EffTFPt20CSCOnly = ptSubdirCSCOnly.make<TH1F>("EffTFPt20CSCOnly","Efficiency v Pt Tf > 20; 1.2<= #eta <=2.4",ptbins, minpt, maxpt);
    	EffTFPt40CSCOnly = ptSubdirCSCOnly.make<TH1F>("EffTFPt40CSCOnly","Efficiency v Pt Tf > 40; 1.2<= #eta <=2.4",ptbins, minpt, maxpt);
    	EffTFPt60CSCOnly = ptSubdirCSCOnly.make<TH1F>("EffTFPt60CSCOnly","Efficiency v Pt Tf > 60; 1.2<= #eta <=2.4",ptbins, minpt, maxpt);
    	EffTFPt10CSCRestricted = ptSubdirCSCRestricted.make<TH1F>("EffTFPt10CSCRestricted","Efficiency v Pt Tf > 10; 1.2<= #eta <=2.1",ptbins, minpt, maxpt);
	EffTFPt12CSCRestricted = ptSubdirCSCRestricted.make<TH1F>("EffTFPt12CSCRestricted","Efficiency v Pt Tf > 12; 1.2<= #eta <=2.1",ptbins, minpt, maxpt);
	EffTFPt16CSCRestricted = ptSubdirCSCRestricted.make<TH1F>("EffTFPt16CSCRestricted","Efficiency v Pt Tf > 16; 1.2<= #eta <=2.1",ptbins, minpt, maxpt);
    	EffTFPt20CSCRestricted = ptSubdirCSCRestricted.make<TH1F>("EffTFPt20CSCRestricted","Efficiency v Pt Tf > 20; 1.2<= #eta <=2.1",ptbins, minpt, maxpt);
    	EffTFPt40CSCRestricted = ptSubdirCSCRestricted.make<TH1F>("EffTFPt40CSCRestricted","Efficiency v Pt Tf > 40; 1.2<= #eta <=2.1",ptbins, minpt, maxpt);
    	EffTFPt60CSCRestricted = ptSubdirCSCRestricted.make<TH1F>("EffTFPt60CSCRestricted","Efficiency v Pt Tf > 60; 1.2<= #eta <=2.1",ptbins, minpt, maxpt);
    	EffTFPt10DTOnly = ptSubdirDTOnly.make<TH1F>("EffTFPt10DTOnly","Efficiency v Pt Tf > 10; 0<= #eta <=0.9",ptbins, minpt, maxpt);
	EffTFPt12DTOnly = ptSubdirDTOnly.make<TH1F>("EffTFPt12DTOnly","Efficiency v Pt Tf > 12; 0<= #eta <=0.9",ptbins, minpt, maxpt);
	EffTFPt16DTOnly = ptSubdirDTOnly.make<TH1F>("EffTFPt16DTOnly","Efficiency v Pt Tf > 16; 0<= #eta <=0.9",ptbins, minpt, maxpt);
    	EffTFPt20DTOnly = ptSubdirDTOnly.make<TH1F>("EffTFPt20DTOnly","Efficiency v Pt Tf > 20; 0<= #eta <=0.9",ptbins, minpt, maxpt);
    	EffTFPt40DTOnly = ptSubdirDTOnly.make<TH1F>("EffTFPt40DTOnly","Efficiency v Pt Tf > 40; 0<= #eta <=0.9",ptbins, minpt, maxpt);
    	EffTFPt60DTOnly = ptSubdirDTOnly.make<TH1F>("EffTFPt60DTOnly","Efficiency v Pt Tf > 60; 0<= #eta <=0.9",ptbins, minpt, maxpt);
    	EffTFPt10Overlap = ptSubdirOverlap.make<TH1F>("EffTFPt10Overlap","Efficiency v Pt Tf > 10; 0.9<= #eta <=1.2",ptbins, minpt, maxpt);
	EffTFPt12Overlap = ptSubdirOverlap.make<TH1F>("EffTFPt12Overlap","Efficiency v Pt Tf > 12; 0.9<= #eta <=1.2",ptbins, minpt, maxpt);
	EffTFPt16Overlap = ptSubdirOverlap.make<TH1F>("EffTFPt16Overlap","Efficiency v Pt Tf > 16; 0.9<= #eta <=1.2",ptbins, minpt, maxpt);
    	EffTFPt20Overlap = ptSubdirOverlap.make<TH1F>("EffTFPt20Overlap","Efficiency v Pt Tf > 20; 0.9<= #eta <=1.2",ptbins, minpt, maxpt);
    	EffTFPt40Overlap = ptSubdirOverlap.make<TH1F>("EffTFPt40Overlap","Efficiency v Pt Tf > 40; 0.9<= #eta <=1.2",ptbins, minpt, maxpt);
    	EffTFPt60Overlap = ptSubdirOverlap.make<TH1F>("EffTFPt60Overlap","Efficiency v Pt Tf > 60; 0.9<= #eta <=1.2",ptbins, minpt, maxpt);


    	EffTFPt10HighEta = ptSubdirHighEta.make<TH1F>("EffTFPt10HighEta","Efficiency v Pt Tf > 10; 2.1<= #eta",ptbins, minpt, maxpt);
	EffTFPt12HighEta = ptSubdirHighEta.make<TH1F>("EffTFPt12HighEta","Efficiency v Pt Tf > 12; 2.1<= #eta",ptbins, minpt, maxpt);
	EffTFPt16HighEta = ptSubdirHighEta.make<TH1F>("EffTFPt16HighEta","Efficiency v Pt Tf > 16; 2.1<= #eta",ptbins, minpt, maxpt);
    	EffTFPt20HighEta = ptSubdirHighEta.make<TH1F>("EffTFPt20HighEta","Efficiency v Pt Tf > 20; 2.1<= #eta",ptbins, minpt, maxpt);
    	EffTFPt40HighEta = ptSubdirHighEta.make<TH1F>("EffTFPt40HighEta","Efficiency v Pt Tf > 40; 2.1<= #eta",ptbins, minpt, maxpt);
    	EffTFPt60HighEta = ptSubdirHighEta.make<TH1F>("EffTFPt60HighEta","Efficiency v Pt Tf > 60; 2.1<= #eta",ptbins, minpt, maxpt);


    	EffEtaAll = etaSubdir.make<TH1F>("EffEtaAll","Efficiency v eta for all Tracks", 50, 0, 2.5);
    	EffEtaQ3 = etaSubdir.make<TH1F>("EffEtaQ3","Efficiency v #eta for Quality >= 3 Tracks", 50, 0, 2.5);
    	EffEtaQ2 = etaSubdir.make<TH1F>("EffEtaQ2","Efficiency v #eta for Quality >= 2 Tracks", 50, 0, 2.5);
    	EffEtaQ1 = etaSubdir.make<TH1F>("EffEtaQ1","Efficiency v #eta for Quality >= 1 Tracks", 50, 0, 2.5);	
    	EffSignedEtaAll = etaSubdir.make<TH1F>("EffSignedEtaAll","Efficiency v eta for all Tracks", 100, -2.5, 2.5);
    	EffSignedEtaQ3 = etaSubdir.make<TH1F>("EffSignedEtaQ3","Efficiency v #eta for Quality >= 3 Tracks", 100, -2.5, 2.5);
    	EffSignedEtaQ2 = etaSubdir.make<TH1F>("EffSignedEtaQ2","Efficiency v #eta for Quality >= 2 Tracks", 100, -2.5, 2.5);
    	EffSignedEtaQ1 = etaSubdir.make<TH1F>("EffSignedEtaQ1","Efficiency v #eta for Quality >= 1 Tracks", 100, -2.5, 2.5);	
    	EffPhiQ3 = phiSubdir.make<TH1F>("EffPhiQ3","Efficiency v #phi for Quality >= 3 Tracks",144,0,6.283);
    	EffPhiQ2 = phiSubdir.make<TH1F>("EffPhiQ2","Efficiency v #phi for Quality >= 2 Tracks",144,0,6.283);
    	EffPhiQ1 = phiSubdir.make<TH1F>("EffPhiQ1","Efficiency v #phi for Quality >= 1 Tracks",144,0,6.283);
}


void EffHistogramList::ComputeEff(TrackHistogramList* refHists)
{
    
    divideHistograms(refHists);
	
////////////////////
//// Pt Eff ///////
//////////////////
   
    //Putting Histograms in vectors for use in later functions
    std::vector<TH1F*> Overallhists;
    Overallhists.push_back(EffPtOverall);
    Overallhists.push_back(EffTFPt12Overall);
    Overallhists.push_back(EffTFPt20Overall);
    Overallhists.push_back(EffTFPt40Overall);
    Overallhists.push_back(EffTFPt60Overall);
    
    std::vector<TH1F*> CSCOnlyhists;
    CSCOnlyhists.push_back(EffPtCSCOnly);
    CSCOnlyhists.push_back(EffTFPt12CSCOnly);
    CSCOnlyhists.push_back(EffTFPt20CSCOnly);
    CSCOnlyhists.push_back(EffTFPt40CSCOnly);
    CSCOnlyhists.push_back(EffTFPt60CSCOnly);

    std::vector<TH1F*> CSCRestrictedhists;
    CSCRestrictedhists.push_back(EffPtCSCRestricted);
    CSCRestrictedhists.push_back(EffTFPt12CSCRestricted);
    CSCRestrictedhists.push_back(EffTFPt20CSCRestricted);
    CSCRestrictedhists.push_back(EffTFPt40CSCRestricted);
    CSCRestrictedhists.push_back(EffTFPt60CSCRestricted);
    
    std::vector<TH1F*> DTOnlyhists;
    DTOnlyhists.push_back(EffPtDTOnly);
    DTOnlyhists.push_back(EffTFPt12DTOnly);
    DTOnlyhists.push_back(EffTFPt20DTOnly);
    DTOnlyhists.push_back(EffTFPt40DTOnly);
    DTOnlyhists.push_back(EffTFPt60DTOnly);
    
    std::vector<TH1F*> Overlaphists;
    Overlaphists.push_back(EffPtOverlap);
    Overlaphists.push_back(EffTFPt12Overlap);
    Overlaphists.push_back(EffTFPt20Overlap);
    Overlaphists.push_back(EffTFPt40Overlap);
    Overlaphists.push_back(EffTFPt60Overlap);

    std::vector<TH1F*> HighEtahists;
    HighEtahists.push_back(EffPtHighEta);
    HighEtahists.push_back(EffTFPt12HighEta);
    HighEtahists.push_back(EffTFPt20HighEta);
    HighEtahists.push_back(EffTFPt40HighEta);
    HighEtahists.push_back(EffTFPt60HighEta);
    
    //must match the threshold order of the histograms pushed into
    //the vectors directly above
    std::vector<std::string> thresholds;
    thresholds.push_back("");
    thresholds.push_back("12");
    thresholds.push_back("20");
    thresholds.push_back("40");
    thresholds.push_back("60");
    
    
    //Here you define where your Pt Histograms "Plateau"
    //the indices should line up with the OverallHists, 
    //CSCOnlyhists, DTOnlyhists, etc. above.
    std::vector<double> PlateauDefinitions;
    PlateauDefinitions.push_back(80);
    PlateauDefinitions.push_back(24);
    PlateauDefinitions.push_back(30);
    PlateauDefinitions.push_back(60);
    PlateauDefinitions.push_back(80);
    
    //Computing the plateau Efficiencies of Pt Histograms and writing them to a file
    std::ofstream* PtStats=new std::ofstream(PtEffStatsFilename.c_str());
    (*PtStats)<<"Pt Plateau Efficiencies for Overall region (|eta|<=2.4)";
    computePtPlateauEff(PtStats, PlateauDefinitions,thresholds,Overallhists);
    (*PtStats)<<"\n\nPt Plateau Efficiencies for CSC Only region (1.2<=|eta|<=2.4)";
    computePtPlateauEff(PtStats, PlateauDefinitions,thresholds,CSCOnlyhists);
    (*PtStats)<<"\n\nPt Plateau Efficiencies for CSC Restricted region (1.2<=|eta|<=2.1)";
    computePtPlateauEff(PtStats, PlateauDefinitions,thresholds,CSCRestrictedhists);
    (*PtStats)<<"\n\nPt Plateau Efficiencies for DT Only region (|eta|<=0.9)";
    computePtPlateauEff(PtStats, PlateauDefinitions,thresholds,DTOnlyhists);
    (*PtStats)<<"\n\nPt Plateau Efficiencies for Overlap region (1.2<=|eta|<=0.9)";
    computePtPlateauEff(PtStats, PlateauDefinitions,thresholds,Overlaphists);
    (*PtStats)<<"\n\nPt Plateau Efficiencies for HighEta region (2.1<=|eta|)";
    computePtPlateauEff(PtStats, PlateauDefinitions,thresholds,HighEtahists);
    PtStats->close();
    
    //Drawing Pt Histograms
    DrawPtEffHists("Overall",PtEffAllOverall,fitThreshOverall,TrackerLeg1Overall,thresholds,Overallhists);
    DrawPtEffHists("CSCOnly",PtEffAllCSCOnly,fitThreshCSCOnly,TrackerLeg1CSCOnly,thresholds,CSCOnlyhists);
    DrawPtEffHists("CSCRestricted",PtEffAllCSCRestricted,fitThreshCSCRestricted,TrackerLeg1CSCRestricted,thresholds,CSCRestrictedhists);
    DrawPtEffHists("DTOnly",PtEffAllDTOnly,fitThreshDTOnly,TrackerLeg1DTOnly,thresholds,DTOnlyhists);
    DrawPtEffHists("Overlap",PtEffAllOverlap,fitThreshOverlap,TrackerLeg1Overlap,thresholds,Overlaphists);
    DrawPtEffHists("HighEta",PtEffAllHighEta,fitThreshHighEta,TrackerLeg1HighEta,thresholds,HighEtahists);
   
    ///////////////////
    //Overall Eta Eff//
    //////////////////
    EtaEff = fs->make<TCanvas>("EtaEff");
    EffEtaQ1->GetXaxis()->SetTitle("Eta Sim");
    EffEtaQ1->GetYaxis()->SetTitle("Efficiency");
    EffEtaQ2->GetXaxis()->SetTitle("Eta Sim");
    EffEtaQ2->GetYaxis()->SetTitle("Efficiency");
    EffEtaQ3->GetXaxis()->SetTitle("Eta Sim");
    EffEtaQ3->GetYaxis()->SetTitle("Efficiency");
    EffEtaAll->GetYaxis()->SetRangeUser(0.0,1.1);
    EffEtaQ1->GetYaxis()->SetRangeUser(0.0,1.1);
    EffEtaQ2->GetYaxis()->SetRangeUser(0.0,1.1);
    EffEtaQ3->GetYaxis()->SetRangeUser(0.0,1.1);
    EffEtaQ1->SetTitle("Efficiency for Quality 1, 2, and 3 Tracks");
    EffEtaQ1->SetFillColor(1);
    EffEtaQ2->SetFillColor(4);
    EffEtaQ3->SetFillColor(3);

    EffEtaQ1->Draw();
    EffEtaQ2->Draw("same");
    EffEtaQ3->Draw("same");
    TrackerLeg2 = new TLegend(0.12,0.12,0.32,0.27);
    TrackerLeg2->AddEntry(EffEtaQ1,"All Tracks","f");
    TrackerLeg2->AddEntry(EffEtaQ2,"Quality > 1","f");
    TrackerLeg2->AddEntry(EffEtaQ3,"Quality > 2","f");
    TrackerLeg2->Draw();
    latexDescription->Draw();
    gPad->SetTicks(1,0);
    //EtaEff->Print("EffEta.png","png");
    
    
    //////////////////////////
    //Overall Signed Eta Eff// 
    /////////////////////////
    SignedEtaEff = fs->make<TCanvas>("SignedEtaEff");
    EffSignedEtaQ1->GetXaxis()->SetTitle("Eta Sim");
    EffSignedEtaQ1->GetYaxis()->SetTitle("Efficiency");
    EffSignedEtaQ2->GetXaxis()->SetTitle("Eta Sim");
    EffSignedEtaQ2->GetYaxis()->SetTitle("Efficiency");
    EffSignedEtaQ3->GetXaxis()->SetTitle("Eta Sim");
    EffSignedEtaQ3->GetYaxis()->SetTitle("Efficiency");
    EffSignedEtaAll->GetYaxis()->SetRangeUser(0.0,1.1);
    EffSignedEtaQ1->GetYaxis()->SetRangeUser(0.0,1.1);
    EffSignedEtaQ2->GetYaxis()->SetRangeUser(0.0,1.1);
    EffSignedEtaQ3->GetYaxis()->SetRangeUser(0.0,1.1);
    EffSignedEtaQ1->SetTitle("Efficiency for Quality 1, 2, and 3 Tracks");
    EffSignedEtaQ1->SetFillColor(1);
    EffSignedEtaQ2->SetFillColor(4);
    EffSignedEtaQ3->SetFillColor(3);
    EffSignedEtaQ1->Draw();
    EffSignedEtaQ2->Draw("same");
    EffSignedEtaQ3->Draw("same");
    TrackerLeg3 = new TLegend(0.12,0.12,0.32,0.27);
    TrackerLeg3->AddEntry(EffSignedEtaQ1,"All Tracks","f");
    TrackerLeg3->AddEntry(EffSignedEtaQ2,"Quality > 1","f");
    TrackerLeg3->AddEntry(EffSignedEtaQ3,"Quality > 2","f");
    TrackerLeg3->Draw();
    latexDescription->Draw();
    gPad->SetTicks(1,0);
    //EtaEff->Print("EffEta.png","png");
    
    //////////////////////
    //// Overall Phi Eff//
    //////////////////////
    PhiEff = fs->make<TCanvas>("PhiEff");
    EffPhiQ1->GetXaxis()->SetTitle("Phi Sim");
    EffPhiQ1->GetYaxis()->SetTitle("Efficiency");
    EffPhiQ2->GetXaxis()->SetTitle("Phi Sim");
    EffPhiQ2->GetYaxis()->SetTitle("Efficiency");
    EffPhiQ3->GetXaxis()->SetTitle("Phi Sim");
    EffPhiQ3->GetYaxis()->SetTitle("Efficiency");
    EffPhi->GetYaxis()->SetRangeUser(0.0,1.1);
    EffPhiQ1->GetYaxis()->SetRangeUser(0.0,1.1);
    EffPhiQ2->GetYaxis()->SetRangeUser(0.0,1.1);
    EffPhiQ3->GetYaxis()->SetRangeUser(0.0,1.1);
    EffPhiQ1->SetTitle("Efficiency for Quality 1, 2, and 3 Tracks");
    EffPhiQ1->SetFillColor(1);
    EffPhiQ2->SetFillColor(4);
    EffPhiQ3->SetFillColor(3);
    EffPhiQ1->Draw(); 
    EffPhiQ2->Draw("same");
    EffPhiQ3->Draw("same");
    TrackerLeg2 = new TLegend(0.4,0.12,0.6,0.27);
    TrackerLeg2->AddEntry(EffPhiQ1,"All Tracks","f");
    TrackerLeg2->AddEntry(EffPhiQ2,"Quality > 1","f"); 
    TrackerLeg2->AddEntry(EffPhiQ3,"Quality > 2","f");
    TrackerLeg2->Draw();
    latexDescription->Draw();
    gPad->SetTicks(1,0);
    //PhiEff->Print("EffPhi.png","png");
    
    EffPhi_mod_10_Q2_endcap1->GetXaxis()->SetTitle("Phi%10 (deg)");
    EffPhi_mod_10_Q3_endcap1->GetXaxis()->SetTitle("Phi%10 (deg)");
    EffPhi_mod_10_Q2_endcap2->GetXaxis()->SetTitle("Phi%10 (deg)");
    EffPhi_mod_10_Q3_endcap2->GetXaxis()->SetTitle("Phi%10 (deg)");
    EffPhi_mod_10_Q2_endcap1->GetYaxis()->SetTitle("Efficiency");
    EffPhi_mod_10_Q3_endcap1->GetYaxis()->SetTitle("Efficiency");
    EffPhi_mod_10_Q2_endcap2->GetYaxis()->SetTitle("Efficiency");
    EffPhi_mod_10_Q3_endcap2->GetYaxis()->SetTitle("Efficiency");

}
void EffHistogramList::Print()
{
    PtEffAllOverall->Print("EffPtOverall.png","png");
    PtEffAllOverlap->Print("EffPtOverlap.png","png");
    PtEffAllHighEta->Print("EffPtHighEta.png","png");
    PtEffAllCSCOnly->Print("EffPtCSCOnly.png","png");
    EtaEff->Print("EffEta.png","png");
    SignedEtaEff->Print("EffSignedEta.png","png");
    PhiEff->Print("EffPhi.png","png");
}
  
  
  
void EffHistogramList::DrawPtEffHists(std::string region, TCanvas* canvas, TF1* fit, TLegend* legend, std::vector<std::string> thresholds, std::vector<TH1F*> PtEffHists)
{
    std::string tmp;
    
    tmp="PtEffAll"+region;
    canvas = fs->make<TCanvas>(tmp.c_str());
    tmp="fitThresh"+region;
    fit = new TF1(tmp.c_str(), csctf_analysis::thresh, 0, 100, 4);
    legend = new TLegend(0.7,0.15,0.85,0.35);
	
    std::vector<TH1F*>::iterator iHist;
    std::vector<std::string>::iterator iThreshold;

    int i=0;
    for(iHist=PtEffHists.begin();iHist!=PtEffHists.end();iHist++)
    {
	PtEffHists[i]->GetXaxis()->SetTitle("Pt Sim (GeV/c)");
	PtEffHists[i]->GetYaxis()->SetTitle("Efficiency");
	PtEffHists[i]->GetYaxis()->SetRangeUser(0.0,1.1);
	tmp=region+" Pt Efficiency";
	PtEffHists[i]->SetTitle(tmp.c_str());
	PtEffHists[i]->SetFillColor(7-i);
	tmp="Pt"+thresholds[i];
	fit->SetParNames(tmp.c_str(),"Resol","Constant","Slope");
	
	tmp="fitThresh"+region;	
	PtEffHists[i]->Fit(tmp.c_str());
	
	if(i==0) tmp="All Tracks";
	else tmp="Pt_{TF} > "+thresholds[i];
	
	legend->AddEntry(PtEffHists[i],tmp.c_str(),"f");
	
	i++;
    }
    
    i=0;
    PtEffHists[0]->Draw("Hist");
    for(iHist=PtEffHists.begin();iHist!=PtEffHists.end();iHist++){PtEffHists[i]->Draw("Hist Same"); i++;}
 
    legend->Draw("same");
    latexDescription->Draw();
    gPad->SetTicks(1,0);
}


void EffHistogramList::computeErrors(TrackHistogramList* refHists)
{
	refHists->matchTFPt10CSCRestricted->Sumw2();	refHists->matchTFPt12CSCRestricted->Sumw2();	refHists->matchTFPt16CSCRestricted->Sumw2();
	refHists->matchTFPt20CSCRestricted->Sumw2();	refHists->matchTFPt40CSCRestricted->Sumw2();	refHists->matchTFPt60CSCRestricted->Sumw2();
	
	refHists->matchTFPt10Overall->Sumw2();	refHists->matchTFPt12Overall->Sumw2();	refHists->matchTFPt16Overall->Sumw2();
	refHists->matchTFPt20Overall->Sumw2();	refHists->matchTFPt40Overall->Sumw2();	refHists->matchTFPt60Overall->Sumw2();
	
	refHists->matchTFPt10CSCOnly->Sumw2();	refHists->matchTFPt12CSCOnly->Sumw2();	refHists->matchTFPt16CSCOnly->Sumw2();
	refHists->matchTFPt20CSCOnly->Sumw2();	refHists->matchTFPt40CSCOnly->Sumw2();	refHists->matchTFPt60CSCOnly->Sumw2();

	refHists->matchTFPt10DTOnly->Sumw2();	refHists->matchTFPt12DTOnly->Sumw2();	refHists->matchTFPt16DTOnly->Sumw2();
	refHists->matchTFPt20DTOnly->Sumw2();	refHists->matchTFPt40DTOnly->Sumw2();	refHists->matchTFPt60DTOnly->Sumw2();

	refHists->matchTFPt10Overlap->Sumw2();	refHists->matchTFPt12Overlap->Sumw2();	refHists->matchTFPt16Overlap->Sumw2();
	refHists->matchTFPt20Overlap->Sumw2();	refHists->matchTFPt40Overlap->Sumw2();	refHists->matchTFPt60Overlap->Sumw2();
	refHists->matchTFPt10HighEta->Sumw2();	refHists->matchTFPt12HighEta->Sumw2();	refHists->matchTFPt16HighEta->Sumw2();
	refHists->matchTFPt20HighEta->Sumw2();	refHists->matchTFPt40HighEta->Sumw2();	refHists->matchTFPt60HighEta->Sumw2();

	refHists->matchPtOverall->Sumw2();	refHists->matchPtCSCOnly->Sumw2();	refHists->matchPtDTOnly->Sumw2();
	refHists->matchPtHighEta->Sumw2();	refHists->ptDenHighEta->Sumw2();	refHists->ptDenCSCOnly->Sumw2();

	refHists->ptDenDTOnly->Sumw2();	refHists->ptDenCSCRestricted->Sumw2();	refHists->ptDenOverall->Sumw2();
}

void EffHistogramList::divideHistograms(TrackHistogramList* refHists)
{
	computeErrors(refHists);


	EffTFPt10Overall->Divide(refHists->matchTFPt10Overall, refHists->ptDenOverall);
	EffTFPt12Overall->Divide(refHists->matchTFPt12Overall, refHists->ptDenOverall);
	EffTFPt16Overall->Divide(refHists->matchTFPt16Overall, refHists->ptDenOverall);
    	EffTFPt20Overall->Divide(refHists->matchTFPt20Overall, refHists->ptDenOverall);
    	EffTFPt40Overall->Divide(refHists->matchTFPt40Overall, refHists->ptDenOverall);
    	EffTFPt60Overall->Divide(refHists->matchTFPt60Overall, refHists->ptDenOverall);
    	EffTFPt10CSCOnly->Divide(refHists->matchTFPt10CSCOnly, refHists->ptDenCSCOnly);
	EffTFPt12CSCOnly->Divide(refHists->matchTFPt12CSCOnly, refHists->ptDenCSCOnly);
	EffTFPt16CSCOnly->Divide(refHists->matchTFPt16CSCOnly, refHists->ptDenCSCOnly);
    	EffTFPt20CSCOnly->Divide(refHists->matchTFPt20CSCOnly, refHists->ptDenCSCOnly);
    	EffTFPt40CSCOnly->Divide(refHists->matchTFPt40CSCOnly, refHists->ptDenCSCOnly);
    	EffTFPt60CSCOnly->Divide(refHists->matchTFPt60CSCOnly, refHists->ptDenCSCOnly);
    	EffTFPt10CSCRestricted->Divide(refHists->matchTFPt10CSCRestricted, refHists->ptDenCSCRestricted);
	EffTFPt12CSCRestricted->Divide(refHists->matchTFPt12CSCRestricted, refHists->ptDenCSCRestricted);
	EffTFPt16CSCRestricted->Divide(refHists->matchTFPt16CSCRestricted, refHists->ptDenCSCRestricted);
    	EffTFPt20CSCRestricted->Divide(refHists->matchTFPt20CSCRestricted, refHists->ptDenCSCRestricted);
    	EffTFPt40CSCRestricted->Divide(refHists->matchTFPt40CSCRestricted, refHists->ptDenCSCRestricted);
    	EffTFPt60CSCRestricted->Divide(refHists->matchTFPt60CSCRestricted, refHists->ptDenCSCRestricted);    
    	EffTFPt10DTOnly->Divide(refHists->matchTFPt10DTOnly, refHists->ptDenDTOnly);
	EffTFPt12DTOnly->Divide(refHists->matchTFPt12DTOnly, refHists->ptDenDTOnly);
	EffTFPt16DTOnly->Divide(refHists->matchTFPt16DTOnly, refHists->ptDenDTOnly);
    	EffTFPt20DTOnly->Divide(refHists->matchTFPt20DTOnly, refHists->ptDenDTOnly);
    	EffTFPt40DTOnly->Divide(refHists->matchTFPt40DTOnly, refHists->ptDenDTOnly);
    	EffTFPt60DTOnly->Divide(refHists->matchTFPt60DTOnly, refHists->ptDenDTOnly);
    	EffTFPt10Overlap->Divide(refHists->matchTFPt10Overlap, refHists->ptDenOverlap);
	EffTFPt12Overlap->Divide(refHists->matchTFPt12Overlap, refHists->ptDenOverlap);
    	EffTFPt16Overlap->Divide(refHists->matchTFPt16Overlap, refHists->ptDenOverlap);
	EffTFPt20Overlap->Divide(refHists->matchTFPt20Overlap, refHists->ptDenOverlap);
    	EffTFPt40Overlap->Divide(refHists->matchTFPt40Overlap, refHists->ptDenOverlap);
    	EffTFPt60Overlap->Divide(refHists->matchTFPt60Overlap, refHists->ptDenOverlap);
    	EffTFPt10HighEta->Divide(refHists->matchTFPt10HighEta, refHists->ptDenHighEta);
	EffTFPt12HighEta->Divide(refHists->matchTFPt12HighEta, refHists->ptDenHighEta);
    	EffTFPt16HighEta->Divide(refHists->matchTFPt16HighEta, refHists->ptDenHighEta);
	EffTFPt20HighEta->Divide(refHists->matchTFPt20HighEta, refHists->ptDenHighEta);
    	EffTFPt40HighEta->Divide(refHists->matchTFPt40HighEta, refHists->ptDenHighEta);
    	EffTFPt60HighEta->Divide(refHists->matchTFPt60HighEta, refHists->ptDenHighEta);


    	//EffPt->Divide(refHists->matchPt, refHists->fidPtDen);
	EffPtOverall->Divide(refHists->matchPtOverall, refHists->ptDenOverall);
    	EffPtCSCOnly->Divide(refHists->matchPtCSCOnly, refHists->ptDenCSCOnly);
	EffPtCSCRestricted->Divide(refHists->matchPtCSCRestricted, refHists->ptDenCSCRestricted);
    	EffPtDTOnly->Divide(refHists->matchPtDTOnly, refHists->ptDenDTOnly); 
    	EffPtOverlap->Divide(refHists->matchPtOverlap, refHists->ptDenOverlap);
    	EffPtHighEta->Divide(refHists->matchPtHighEta, refHists->ptDenHighEta);
	
    	EffEtaAll->Divide(refHists->matchEta, refHists->Eta);
	EffEtaQ3->Divide(refHists->EtaQ3, refHists->Eta);
    	EffEtaQ2->Divide(refHists->EtaQ2, refHists->Eta);
    	EffEtaQ1->Divide(refHists->EtaQ1, refHists->Eta);
	EffSignedEtaAll->Divide(refHists->signedMatchEta, refHists->signedEta);
    	EffSignedEtaQ3->Divide(refHists->signedEtaQ3, refHists->signedEta);
    	EffSignedEtaQ2->Divide(refHists->signedEtaQ2, refHists->signedEta);
    	EffSignedEtaQ1->Divide(refHists->signedEtaQ1, refHists->signedEta);

    	//EffPhi->Divide(refHists->matchPhi, refHists->Phi);
    	EffPhi_mod_10_Q2_endcap1->Divide(refHists->matchPhi_mod_10_Q2_endcap1,refHists->Phi_mod_10_endcap1);
    	EffPhi_mod_10_Q3_endcap1->Divide(refHists->matchPhi_mod_10_Q3_endcap1,refHists->Phi_mod_10_endcap1);
    	EffPhi_mod_10_Q2_endcap2->Divide(refHists->matchPhi_mod_10_Q2_endcap2,refHists->Phi_mod_10_endcap2);
    	EffPhi_mod_10_Q3_endcap2->Divide(refHists->matchPhi_mod_10_Q3_endcap2,refHists->Phi_mod_10_endcap2);

	EffPhi->Divide(refHists->matchPhi, refHists->Phi);
    	EffPhiQ3->Divide(refHists->PhiQ3, refHists->Phi);
    	EffPhiQ2->Divide(refHists->PhiQ2, refHists->Phi);
    	EffPhiQ1->Divide(refHists->PhiQ1, refHists->Phi);

}

void EffHistogramList::computePtPlateauEff(std::ofstream* PtStats, std::vector<double> PlateauDefinitions, std::vector<std::string> thresholds, std::vector<TH1F*> PtEffHists)
{
	std::vector<TH1F*>::iterator iHist;
	std::vector<std::string>::iterator iThreshold;
	
	TF1* constFit = new TF1("constFit", "[0]", 2., 140.);
	
	Double_t xmin;
	int i=0;
	for(iHist=PtEffHists.begin();iHist!=PtEffHists.end();iHist++)
	{
		xmin = PlateauDefinitions[i]; //define start of plateau
		PtEffHists[i] -> Fit(constFit,"R","", xmin, 140.); //do fit
		if(i==0) (*PtStats)<<"\nDefault Pt thresh   Efficiency:  "<<constFit->GetParameter(0)<<"   "; // efficiency of plateau
		else (*PtStats)<<"\nPt>"<<thresholds[i].c_str()<<"               Efficiency:  "<<constFit->GetParameter(0)<<"   "; // efficiency of plateau
		
		(*PtStats)<<"Error:   "<<constFit->GetParError(0); // error on the plateau efficiency
		i++;
	}
}
  
  
  
  
  
  
Double_t thresh(Double_t* pt, Double_t* par)
  {
    Double_t fitval = (0.5*TMath::Erf((pt[0]/par[0] + 1.0)/(TMath::Sqrt(2.0)*par[1])) + 0.5*TMath::Erf((pt[0]/par[0] - 1.0)/(TMath::Sqrt(2.0)*par[1])) )*(par[2] + par[3]*pt[0]);
    return fitval;
  }
}
