#include <algorithm>
#include <fstream>
#include <TH2D.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TString.h>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

void SavePlotWithName(TH2D* h2, TString s);
double findXMax(TF1* fitFunc,double xMin, double xLimit, double dist_fraction, double epsilon);

void ViewPlots(Int_t RunNumber, bool noVideoOutput = false){
	bool verbose = false;
	bool additionalPlots = true;
	map<pair<int,int>,double> dist_fraction = {
		{pair<int,int>(0,0),0.4},
		{pair<int,int>(0,2),0.4},
		{pair<int,int>(1,0),0.4},
		{pair<int,int>(1,2),0.4}
	};

	gStyle->SetPalette(1);
	gStyle->SetOptStat(0);

	Int_t  screen_x=0, screen_y=0;
	UInt_t screen_w=1920, screen_h=1080;

	// cout<<screen_x<<" "<<screen_y<<" "<<screen_w<<" "<<screen_h<<endl;
	
	// !!! ATTENTION !!!
	// Ballpark figures, just to make the code work. To be corrected
	Int_t firstRunOfTheYear = 314247;
	Int_t lastRunPreTs1     = 317696;
	Int_t lastRunPreTs2     = 322633;
	Int_t lastRunOfTheYear  = 324897;

	Int_t yAlignmentRun 	= 315512;

	bool Pre_TS1 = kFALSE;
	bool TS1_TS2 = kFALSE;
	bool Post_TS2 = kFALSE;
	map<int,double> shift = {{0,0},{2,0}};


	if (RunNumber < firstRunOfTheYear){
		cout << "This run doesn't belong to 2018 data taking!" << endl;
		// return;
	}
	if(RunNumber > firstRunOfTheYear && RunNumber <= lastRunPreTs1){
		cout << "This run belongs to Pre-TS1 data taking" << endl;
		Pre_TS1 = kTRUE;
	}
	if(RunNumber > lastRunPreTs1 && RunNumber <= lastRunPreTs2){
		cout << "This run belongs to the period of data taking beween TS1 and TS2" << endl;
		TS1_TS2 = kTRUE;
	}
	if(RunNumber > lastRunPreTs2 && RunNumber <= lastRunOfTheYear){
		cout << "This run belongs to Post-TS2 data taking" << endl;
		Post_TS2 = kTRUE;
	}
	if(RunNumber > lastRunOfTheYear){
		cout << "This run doesn't belong to 2018 data taking!" << endl;
		return;
	}

	if(TS1_TS2){ 
		shift = {{0,5*TMath::Cos(8./180.)},{2,-5}};
		// shift = {{0,0},{2,0}};
		yAlignmentRun = 315512;
	}
	if(Post_TS2){ 
		shift = {{0,10*TMath::Cos(8./180.)},{2,-10}};
		// shift = {{0,0},{2,0}};
		yAlignmentRun = 315512;
	}

	// This part defines the area in which the track efficiency is going to be averaged. For every RP it will be computed on a rectangular region going from
	// (xbin, ybin-hbin) to (xbin+wbin, ybin+hbin).
	// vector<Int_t> areaForAvgEfficiency_armX_stX{xbin,ybin,hbin,wbin};
	
	// std::map <std::pair<int,int>,std::vector<int> > areaForAvgEfficiency;
	std::map <std::pair<int,int>,std::vector<float> > areaForControlEfficiency;

	// vector<int> areaForAvgEfficiency_arm0_st0{29,160,2,10};
	// vector<int> areaForAvgEfficiency_arm0_st2{22,157,2,10};
	// vector<int> areaForAvgEfficiency_arm1_st0{23,166,4,10};
	// vector<int> areaForAvgEfficiency_arm1_st2{24,155,4,10};

	vector<float> areaForControlEfficiency_arm0_st0{15.,-2.5,3.5};
	vector<float> areaForControlEfficiency_arm0_st2{15.,-2.,4.};
	vector<float> areaForControlEfficiency_arm1_st0{15.,-2.5,3.5};
	vector<float> areaForControlEfficiency_arm1_st2{15.,-2.,4.};

	// areaForAvgEfficiency_arm0_st0[1] += shift[0];
	// areaForAvgEfficiency_arm1_st0[1] += shift[0];
	// areaForAvgEfficiency_arm0_st2[1] += shift[2];
	// areaForAvgEfficiency_arm1_st2[1] += shift[2];


	// areaForAvgEfficiency[std::pair<Int_t,Int_t>(0,0)] = areaForAvgEfficiency_arm0_st0;
	// areaForAvgEfficiency[std::pair<Int_t,Int_t>(0,2)] = areaForAvgEfficiency_arm0_st2;
	// areaForAvgEfficiency[std::pair<Int_t,Int_t>(1,0)] = areaForAvgEfficiency_arm1_st0;
	// areaForAvgEfficiency[std::pair<Int_t,Int_t>(1,2)] = areaForAvgEfficiency_arm1_st2;

	areaForControlEfficiency[std::pair<Int_t,Int_t>(0,0)] = areaForControlEfficiency_arm0_st0;
	areaForControlEfficiency[std::pair<Int_t,Int_t>(0,2)] = areaForControlEfficiency_arm0_st2;
	areaForControlEfficiency[std::pair<Int_t,Int_t>(1,0)] = areaForControlEfficiency_arm1_st0;
	areaForControlEfficiency[std::pair<Int_t,Int_t>(1,2)] = areaForControlEfficiency_arm1_st2;

	// Extremes for fits map< std::pair<arm,station>, {minAmplFit,maxAmplFit,minMeanFit,maxMeanFit,minSigmaFit,maxSigmaFit} >
	map< std::pair<int,int>, vector<double> > fitRanges;
	fitRanges[std::pair<int,int>(0,0)] = {2.9,11,3,6,3,10};
	fitRanges[std::pair<int,int>(0,2)] = {2.28,11,3,6,3,10};
	fitRanges[std::pair<int,int>(1,0)] = {3.8,11,3,6,3,10};
	fitRanges[std::pair<int,int>(1,2)] = {2.6,11,3,6,3,10};

	// Sensor edge x position
	map< std::pair<int,int>, double> sensorEdgeX;
	sensorEdgeX[std::pair<int,int>(0,0)] = 2.85;
	sensorEdgeX[std::pair<int,int>(0,2)] = 2.28;
	sensorEdgeX[std::pair<int,int>(1,0)] = 3.28;
	sensorEdgeX[std::pair<int,int>(1,2)] = 2.42;

	gROOT->SetBatch(kTRUE);
	TCanvas *uselessCanvas = new TCanvas("prova","prova",screen_x,screen_y,screen_w,screen_h);
	// uselessCanvas->cd();
	// TF1 *prova = new TF1("gaus","gaus(0)",-5.,5.);
	// prova->Draw();
	if(!noVideoOutput) gROOT->SetBatch(kFALSE);
	TCanvas *cPixelHitmap = new TCanvas("cPixelHitmap","PixelHitmap",screen_x,screen_y,screen_w,screen_h);
	TCanvas *cPixelTrackEfficiency = new TCanvas("cPixelTrackEfficiency","PixelTrackEfficiency",screen_x,screen_y,screen_w,screen_h);
	// TCanvas *cPixelInterPotEfficiency = new TCanvas("cPixelInterPotEfficiency","PixelInterPotEfficiency",screen_x,screen_y,screen_w,screen_h);
	TCanvas *cPlaneEfficiency = new TCanvas("cPlaneEfficiency","PlaneEfficiency",screen_x,screen_y,screen_w,screen_h);
	// TCanvas *cCorrelation = new TCanvas("cCorrelation","Correlation",screen_x,screen_y,screen_w,screen_h);

	TFile* inputFile = new TFile(Form("OutputFiles/Run%i.root",RunNumber), "READ");
	TFile* inputFile_refinedEfficiency = new TFile(Form("OutputFiles/Run%i_refinedEfficiency.root",RunNumber), "READ");
	if (inputFile->IsZombie()){
		cout << "Inputfile not found. Aborting." << endl;
		return;
	}

	// Using run 315512 as fit area definition
	TFile* fitFile = new TFile(Form("OutputFiles/Run%i.root",315512), "READ");
	// TFile* fitFile = inputFile;
	if (fitFile->IsZombie()){
		cout << "Fit file not found. Aborting." << endl;
		return;
	}

	// Using first run after TS for Y alignment
	// TFile* yAlignmentFile = new TFile(Form("OutputFiles/Run%i.root",yAlignmentRun), "READ");
	TFile* yAlignmentFile = fitFile;
	if (yAlignmentFile->IsZombie()){
		cout << "Y alignment file not found. Aborting." << endl;
		return;
	}

	cPixelHitmap->Divide(2,2);
	cPixelTrackEfficiency->Divide(2,2);
	// cPixelInterPotEfficiency->Divide(2,2);
	// cCorrelation->Divide(2,2);
	cPlaneEfficiency->Divide(6,4);
	// cAvgEfficiencyFitted->Divide(2,2);

	vector<Int_t> planes{0,1,2,3,4,5};
	vector<Int_t> arms{0,1};
	vector<Int_t> stations{0,2};

	ofstream avgEfficiencyOutputFile;
	avgEfficiencyOutputFile.open(Form("OutputFiles/avgEfficiency_Run%i.dat",RunNumber));

	Int_t mapPlotsPadNumber = 0;
	// Int_t correlationPlotsPadNumber = 0;
	Int_t planePlotsPadNumber = 0;

	for (const auto & arm : arms){
		for (const auto & station : stations){
			++mapPlotsPadNumber;
			cPixelHitmap->cd(mapPlotsPadNumber);

			TH2D* h2PixelHitmap = (TH2D*)inputFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2TrackHitDistribution_arm%i_st%i_rp3",arm,station));
			h2PixelHitmap->SetNameTitle(Form("TrackHitDistribution_arm%i_st%i",arm,station),Form("TrackHitDistribution_arm%i_st%i",arm,station));
			h2PixelHitmap->DrawCopy("colz");
			// SavePlotWithName(h2PixelHitmap,Form("OutputFiles/PlotsRun%i/Run%i_PixelHitmap_arm%i_st%i.png",RunNumber,RunNumber,arm,station));

			// cPixelInterPotEfficiency->cd(mapPlotsPadNumber);
			// TH2D* h2InterPotEfficiency = (TH2D*)inputFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2BetterInterPotEfficiency_arm%i_st%i_rp3",arm,station));
			// h2InterPotEfficiency->SetNameTitle(Form("InterPotEfficiency_arm%i_st%i",arm,station),Form("InterPotEfficiency_arm%i_st%i",arm,station));
			// h2InterPotEfficiency->DrawCopy("colz");
			// SavePlotWithName(h2InterPotEfficiency,Form("OutputFiles/PlotsRun%i/Run%i_InterPotEfficiency_arm%i_st%i.png",RunNumber,RunNumber,arm,station));
			if(noVideoOutput) gROOT->SetBatch(kTRUE);

			cPixelTrackEfficiency->cd(mapPlotsPadNumber);
			TH2D* h2TrackEfficiencyMap = (TH2D*)inputFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2TrackEfficiencyMap_arm%i_st%i_rp3",arm,station));
			h2TrackEfficiencyMap->SetNameTitle(Form("TrackEfficiencyMap_arm%i_st%i",arm,station),Form("TrackEfficiencyMap_arm%i_st%i",arm,station));
			h2TrackEfficiencyMap->DrawCopy("colz");
			SavePlotWithName(h2TrackEfficiencyMap,Form("OutputFiles/PlotsRun%i/Run%i_TrackEfficiency_arm%i_st%i.png",RunNumber,RunNumber,arm,station));
			if(noVideoOutput) gROOT->SetBatch(kTRUE);

			for (const auto & plane : planes){
				planePlotsPadNumber++;
				cPlaneEfficiency->cd(planePlotsPadNumber);
				TH2D* h2PlaneEfficiency = (TH2D*)inputFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->GetDirectory(Form("Arm%i_st%i_rp3_pl%i",arm,station,plane))->Get(Form("h2EfficiencyMap_arm%i_st%i_rp3_pl%i",arm,station,plane));
				h2PlaneEfficiency->SetNameTitle(Form("EfficiencyMap_arm%i_st%i_pl%i",arm,station,plane),Form("EfficiencyMap_arm%i_st%i_pl%i",arm,station,plane));
				h2PlaneEfficiency->DrawCopy("colz");
			}
		}
		// correlationPlotsPadNumber++;
		// cCorrelation->cd(correlationPlotsPadNumber);
		// TH2D* h2X0Correlation = (TH2D*)inputFile->GetDirectory(Form("Arm%i_st0_rp3",arm))->Get(Form("h2X0Correlation_arm%i_st0_rp3",arm));
		// h2X0Correlation->SetNameTitle(Form("XCorrelation_arm%i",arm),Form("XCorrelation_arm%i",arm));
		// h2X0Correlation->DrawCopy("colz");
		// correlationPlotsPadNumber++;
		// cCorrelation->cd(correlationPlotsPadNumber);
		// TH2D* h2Y0Correlation = (TH2D*)inputFile->GetDirectory(Form("Arm%i_st0_rp3",arm))->Get(Form("h2Y0Correlation_arm%i_st0_rp3",arm));
		// h2Y0Correlation->SetNameTitle(Form("YCorrelation_arm%i",arm),Form("YCorrelation_arm%i",arm));
		// h2Y0Correlation->DrawCopy("colz");

		gROOT->SetBatch(kTRUE);
		// TCanvas* cCorrelationToSave = new TCanvas("cCorrelationToSave","Correlation Plots",screen_x,screen_y,screen_w,screen_h);
		// cCorrelationToSave->Divide(2);
		// cCorrelationToSave->cd(1);
		// h2X0Correlation->DrawCopy("colz");
		// cCorrelationToSave->cd(2);
		// h2Y0Correlation->DrawCopy("colz");
		// cCorrelationToSave->SaveAs(Form("OutputFiles/PlotsRun%i/Run%i_Correlation_arm%i.png",RunNumber,RunNumber,arm));
		// delete cCorrelationToSave;
		if(noVideoOutput) gROOT->SetBatch(kFALSE);
	}

	int avgEfficiencyPlotsPadNumber = 0;

	avgEfficiencyOutputFile << "# arm station avgEfficiency avgEfficiencyFitted xMaxFitted pixelsUsedForAvgEfficiencyFitted xMaxFix yParam[0] yParam[1] yFix sigmaParam[0] sigmaParam[1] controlEfficiency"<<endl;

	TFile* areaFile;
	if(additionalPlots) areaFile = new TFile(Form("OutputFiles/avgEfficiencyAreaRun%i.root",RunNumber), "RECREATE");

	for(auto & arm : arms){
		for(auto & station : stations){
			if(verbose) cout << "Analyzing Arm" << arm << "_Station" << station << endl;

			// TF1* amplitudeFitFunction = new TF1("amplitudeFitFunction","expo",0,100);
			TF1* amplitudeFitFunction = new TF1("amplitudeFitFunction","[0]/(x-[1])+[2]",0,100);
			TF1* meanFitFunction = new TF1("meanFitFunction","pol1",0,100);
			TF1* sigmaFitFunction = new TF1("sigmaFitFunction","pol1",0,100);

			TH2D* h2TrackEfficiencyMap;
			TH2D* h2TrackHitDistribution;
			TH1D* h1AmplitudeHist;
			TH1D* h1MeanHist;
			TH1D* h1SigmaHist;

			if(station == 0){
				h2TrackEfficiencyMap = (TH2D*)inputFile_refinedEfficiency->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2RefinedTrackEfficiency_rotated_arm%i_st%i_rp3",arm,station));
				h2TrackHitDistribution = (TH2D*)inputFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp3",arm,station));
				h1AmplitudeHist = (TH1D*)fitFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp3_0",arm,station));
				h1MeanHist = (TH1D*)yAlignmentFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp3_1",arm,station));
				// h1MeanHist = (TH1D*)fitFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp3_1",arm,station));
				h1SigmaHist = (TH1D*)fitFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp3_2",arm,station));
			}
			else{
				h2TrackEfficiencyMap = (TH2D*)inputFile_refinedEfficiency->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2RefinedTrackEfficiency_arm%i_st%i_rp3",arm,station));
				h2TrackHitDistribution = (TH2D*)inputFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2TrackHitDistribution_arm%i_st%i_rp3",arm,station));
				h1AmplitudeHist = (TH1D*)fitFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2TrackHitDistribution_arm%i_st%i_rp3_0",arm,station));
				h1MeanHist = (TH1D*)yAlignmentFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2TrackHitDistribution_arm%i_st%i_rp3_1",arm,station));
				// h1MeanHist = (TH1D*)fitFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2TrackHitDistribution_arm%i_st%i_rp3_1",arm,station));
				h1SigmaHist = (TH1D*)fitFile->GetDirectory(Form("Arm%i_st%i_rp3",arm,station))->Get(Form("h2TrackHitDistribution_arm%i_st%i_rp3_2",arm,station));
			}

			TH2D* h2AvgEfficiencyArea = new TH2D(*h2TrackEfficiencyMap);
			TH2D* h2AreaForAvgEfficiency = new TH2D(*h2TrackHitDistribution);
			h2AreaForAvgEfficiency->Reset();
			h2AreaForAvgEfficiency->SetNameTitle(Form("areaUsedForAvgEfficiency_arm%i_st%i",arm,station),Form("areaUsedForAvgEfficiency_arm%i_st%i",arm,station));

			h1AmplitudeHist->Fit("amplitudeFitFunction","Q","",fitRanges[std::pair<int,int>(arm,station)].at(0),fitRanges[std::pair<int,int>(arm,station)].at(1));

			h1AmplitudeHist->GetXaxis()->SetRangeUser(0,fitRanges[std::pair<int,int>(arm,station)].at(1));
			double xPeak = h1AmplitudeHist->GetBinCenter(h1AmplitudeHist->GetMaximumBin());
			double xMin = sensorEdgeX[std::pair<int,int>(arm,station)];
			double xMax = findXMax(amplitudeFitFunction,xPeak,fitRanges[std::pair<int,int>(arm,station)].at(1),dist_fraction[pair<int,int>(arm,station)],0.001);
			double dist_frac = (double)amplitudeFitFunction->Integral(xPeak,xMax)/amplitudeFitFunction->Integral(xPeak,fitRanges[std::pair<int,int>(arm,station)].at(1));
			
			// double xMax = TMath::Log((67./100.)*(TMath::Exp(amplitudeFitFunction->GetParameter(1)*xMin)))/(amplitudeFitFunction->GetParameter(1));
			
			std::cout << "Fitting arm "<< arm << " station " << station << ": xMin = " << xMin << ", xMax = " << xMax << ", fraction = " << dist_frac << std::endl;

			h1MeanHist->Fit("meanFitFunction","WQ","",fitRanges[std::pair<int,int>(arm,station)].at(2),fitRanges[std::pair<int,int>(arm,station)].at(3));
				// fitRanges[std::pair<int,int>(arm,station)].at(3));
			h1SigmaHist->Fit("sigmaFitFunction","WQ","",fitRanges[std::pair<int,int>(arm,station)].at(4),
				// 2*xMax-1*fitRanges[std::pair<int,int>(arm,station)].at(4));
				fitRanges[std::pair<int,int>(arm,station)].at(5));
			if(verbose) cout << "Selecting area between x values: " << xMin << " and " << xMax << endl;

			Double_t avgEfficiency = 0;
			Int_t pixelsUsedForAvgEfficiency = 0;
			double avgEfficiencyFitted = 0;
			double controlEfficiency = 0;
			Int_t pixelsUsedForControlEfficiency = 0;
			Int_t pixelsUsedForAvgEfficiencyFitted = 0;

			for(int xbin = 1; xbin <= h2TrackEfficiencyMap->GetXaxis()->GetNbins();xbin++){
				double xbinCenter = h2TrackEfficiencyMap->GetXaxis()->GetBinCenter(xbin);
				if(xMin < xbinCenter && xbinCenter < xMax){
					for(int ybin = 1; ybin <= h2TrackEfficiencyMap->GetYaxis()->GetNbins();ybin++){
						double ybinCenter = h2TrackEfficiencyMap->GetYaxis()->GetBinCenter(ybin);
						double yMin = meanFitFunction->Eval(xbinCenter) - 1.5*sigmaFitFunction->Eval(xbinCenter)/2.;// + shift[station]*0.1;
						double yMax = meanFitFunction->Eval(xbinCenter) + 1.5*sigmaFitFunction->Eval(xbinCenter)/2.;// + shift[station]*0.1;
						if(yMin < ybinCenter && ybinCenter < yMax){

							if(verbose) cout << "meanFitFunction->Eval(xbinCenter) " << meanFitFunction->Eval(xbinCenter) << endl;
							if(verbose) cout << "sigmaFitFunction->Eval(xbinCenter) " << sigmaFitFunction->Eval(xbinCenter) << endl;
							if(verbose) cout << "Selecting area between y values: " << yMin << " and " << yMax << endl;
							if(verbose) cout << "Selecting pixel: " << xbinCenter << ", " << ybinCenter << endl;
							if(verbose) cout << "It has: " << h2TrackEfficiencyMap->GetBinContent(xbin,ybin) << " efficiency" << endl;

							++pixelsUsedForAvgEfficiencyFitted;
							h2AreaForAvgEfficiency->SetBinContent(xbin,ybin,1.);
							avgEfficiencyFitted += h2TrackEfficiencyMap->GetBinContent(xbin,ybin);
						}
						else{
							if(areaForControlEfficiency[std::pair<int,int>(arm,station)][1] < ybinCenter && ybinCenter < areaForControlEfficiency[std::pair<int,int>(arm,station)][2]){
								++pixelsUsedForControlEfficiency;
								controlEfficiency += h2TrackEfficiencyMap->GetBinContent(xbin,ybin);
							}
						}
					}
				}
				else{
					if(xMin < xbinCenter && xbinCenter < areaForControlEfficiency[std::pair<int,int>(arm,station)][0]){
						for(int ybin = 1; ybin <= h2TrackEfficiencyMap->GetYaxis()->GetNbins();ybin++){
							double ybinCenter = h2TrackEfficiencyMap->GetYaxis()->GetBinCenter(ybin);
							double controlMinY = areaForControlEfficiency[std::pair<int,int>(arm,station)][1] + shift[station]*0.1;
							double controlMaxY = areaForControlEfficiency[std::pair<int,int>(arm,station)][2] + shift[station]*0.1;
							if( controlMinY < ybinCenter && ybinCenter < controlMaxY){
								++pixelsUsedForControlEfficiency;
								controlEfficiency += h2TrackEfficiencyMap->GetBinContent(xbin,ybin);
							}
						}
					}
				}
			}
			if(additionalPlots){
				areaFile->mkdir(Form("Arm%i_St%i",arm,station));
				areaFile->cd(Form("Arm%i_St%i",arm,station));
				h2TrackHitDistribution->Write();
				h2AreaForAvgEfficiency->Write();
				h2TrackEfficiencyMap->Write();
			}
			delete h2AreaForAvgEfficiency;

			if(verbose) cout << "Pixels used for EfficiencyFitted " << pixelsUsedForAvgEfficiencyFitted << endl;
			avgEfficiencyFitted = avgEfficiencyFitted/((double)pixelsUsedForAvgEfficiencyFitted);
			controlEfficiency = controlEfficiency/((double)pixelsUsedForControlEfficiency);

			cout << "The average fitted efficiency on arm" << arm << "_station" << station << " is: " << avgEfficiencyFitted << endl;
			cout << "The average control efficiency on arm" << arm << "_station" << station << " is: " << controlEfficiency << endl;

			// avgEfficiencyPlotsPadNumber++;
			// cAvgEfficiencyFitted->cd(avgEfficiencyPlotsPadNumber);
			// h2AvgEfficiencyArea->Draw("colz");
			// delete h2AvgEfficiencyArea;

			// int xbin =  areaForAvgEfficiency[std::pair<int,int>(arm,station)].at(0);
			// int ybin =  areaForAvgEfficiency[std::pair<int,int>(arm,station)].at(1);
			// int hbin =  areaForAvgEfficiency[std::pair<int,int>(arm,station)].at(2);
			// int wbin =  areaForAvgEfficiency[std::pair<int,int>(arm,station)].at(3);

			// for(Int_t xPixel = xbin; xPixel <= xbin+wbin; xPixel++){
				// for(Int_t yPixel = ybin-hbin; yPixel<=ybin+hbin; yPixel++){
					// ++pixelsUsedForAvgEfficiency;
					// avgEfficiency += h2TrackEfficiencyMap->GetBinContent(xPixel,yPixel);
				// }
			// }

			// avgEfficiency = avgEfficiency/((double)pixelsUsedForAvgEfficiency);
			// cout << "The average efficiency on arm" << arm << "_station" << station << " is:" << " " << avgEfficiency << endl;
			avgEfficiencyOutputFile << arm << " " << station << " " << avgEfficiencyFitted << " " << pixelsUsedForAvgEfficiencyFitted
			<< " " << meanFitFunction->Eval(sensorEdgeX[std::pair<int,int>(arm,station)]) << " " << controlEfficiency << "\n";

			delete amplitudeFitFunction;
			delete meanFitFunction;
			delete sigmaFitFunction;
		}
	}
	avgEfficiencyOutputFile.close();
	cout << "Average efficiency data saved in " << Form("OutputFiles/avgEfficiency_Run%i.dat",RunNumber) << endl;
	inputFile->Close();
	fitFile->Close();
	if(additionalPlots){
		areaFile->Close();
		std::cout << "Saving average efficiency area plots in: " << areaFile->GetName() << std::endl;
		delete areaFile;
	}
	// delete inputFile;
	// delete fitFile;
}

void SavePlotWithName(TH2D* h2, TString s){
	Int_t  screen_x=0, screen_y=0;
	UInt_t screen_w=1920, screen_h=1080;
	gROOT->SetBatch(kTRUE);
	TCanvas* c = new TCanvas("","",screen_x,screen_y,screen_w,screen_h);
	h2->DrawCopy("colz");
	c->SaveAs(s);
	delete c;
}

double findXMax(TF1* fitFunc,double xMin, double xLimit, double dist_fraction, double epsilon){
		double xMax = xLimit;
		double xMaxLow = xMin;
		double error = fitFunc->Integral(xMin,xMax)/fitFunc->Integral(xMin,xLimit) - dist_fraction;
		int cycle = 1;
		while(TMath::Abs(error)>epsilon){
			if(error > 0){
				xMax -= (xMax - xMaxLow)/2.;
			}
			else{
				double xMaxLow_tmp = xMaxLow;
				xMaxLow = xMax;
				xMax += (xMax - xMaxLow_tmp)/2.;
			}
			error = fitFunc->Integral(xMin,xMax)/fitFunc->Integral(xMin,xLimit) - dist_fraction;
			if(cycle > 1000){
				std::cout << "WARNING: No xMax found!" <<std::endl;
				return xLimit;
			}
			cycle++;
		}
		std::cout << "xMax found after " << cycle << " cycles" << std::endl;
		return xMax;
}