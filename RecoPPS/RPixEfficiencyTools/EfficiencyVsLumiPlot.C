//include <string>
using namespace std;


void LineParser(ifstream&, vector<string>&);						//Function to turn a multiple lines file into a vector of strings 
void Parser(ifstream& file_to_parse, vector<vector<string> >& );	//Function to turn a vector of lines coming from a text in a vector
							 										//of vectors containing the fields. spreadsheet should be empty

void EfficiencyVsLumiPlot(){
	vector<int> runs = {	
			// Add Runs here
			315512, 
			315713,
			315840,
			316114,
			316199,
			316240,
			316505,
			316666,
			316758,
			// 316985,
			317182,
			317320,
			317435,
			317527,
			317641,
			317696,
			// TS1
			319337, 
			319450,
			319579,
			319756,
			319991,
			320038,
			320804,
			320917,
			321051,
			321149,
			321233,
			321396,
			321457,
			321755,
			321831,
			321909,
			321988,
			322106,
			322252,
			322356,
			322431,
			322625,
			// TS2
			323487,
			323702,
			323790,
			323940,
			324077,
			324293,
			324747,
			324791,
			324841
	};
	vector<int> arms = {0, 1};
	vector<int> stations = {0, 2};
	
	TFile* runLumiFile = new TFile("data/RunLumiHist2018.root","READ");
	TFile *outputFile = new TFile("EfficiencyVsLumiPlot.root","RECREATE");
	TH1D* h1RunLumiHist = (TH1D*)runLumiFile->Get("h1RunLumiDel");

	map<int,bool> missingRuns;
	map<int,double> runLumiMap; // map<run,integratedLuminosity>
	map< pair<int,int>, map<int,double> > avgEfficiencyPeakFitted; // map< pair<arm,station>, map<run,avgEfficiencyFittedValues> >
	map< pair<int,int>, map<int,double> > controlAreaEfficiency; // map< pair<arm,station>, map<run,controlAreaEfficiency> >
	map< pair<int,int>, map<int,int> > pixelsUsedForAvgEfficiencyPeakFitted; // map< pair<arm,station>, map<run,pixelsUsedForAvgEfficiencyFitted> >
	map< pair<int,int>, map<int,double> > yAlignment; // map< pair<arm,station>, map<run,yAlignment> >

	int palette[4] = {8,kRed,kMagenta,kBlue};
	gStyle->SetPalette(4,palette);
	int firstRun = 0;

	for(auto & run : runs){
		cout << "Reading data from run " << run << endl;
		// Fill Run - Integrated Lumi (delivered) lookup table
		runLumiMap[run] = h1RunLumiHist->GetBinContent(h1RunLumiHist->GetXaxis()->FindBin(run));

		// Read all average efficiency measurements
		string avgEfficiencyFileName = Form("OutputFiles/avgEfficiency_Run%i.dat",run);
		ifstream avgEfficiencyFile(avgEfficiencyFileName);
		if(!avgEfficiencyFile){
			cout << "Average efficiency file for run " << run << " is missing. Skipping..." << endl;
			missingRuns[run] = true;
			continue;	
		}

		vector< vector<string> > avgEfficiencyFileContent;
		Parser(avgEfficiencyFile,avgEfficiencyFileContent);
		for(auto & line : avgEfficiencyFileContent){ //loop on every line of the file
			avgEfficiencyPeakFitted[pair<int,int>(stoi(line.at(0)),stoi(line.at(1)))][run] = stod(line.at(2));
			pixelsUsedForAvgEfficiencyPeakFitted[pair<int,int>(stoi(line.at(0)),stoi(line.at(1)))][run] = stod(line.at(3));
			yAlignment[pair<int,int>(stoi(line.at(0)),stoi(line.at(1)))][run] = stod(line.at(4));
			controlAreaEfficiency[pair<int,int>(stoi(line.at(0)),stoi(line.at(1)))][run] = stod(line.at(5));
		}
	}

	// Book and Fill graphs

	TMultiGraph *mgAvgEfficiencyPeakFitted = new TMultiGraph("mgAvgEfficiencyPeakFitted","Irradiation Peak Average Efficiency (Fit Method); Integrated Luminosity (fb^{-1}); Average Efficiency");
	TMultiGraph *mgRelativeAvgEfficiencyPeakFitted = new TMultiGraph("mgRelativeAvgEfficiencyPeakFitted","Irradiation Peak Relative Average Efficiency (Fit Method); Integrated Luminosity (fb^{-1}); RelativeAverage Efficiency");
	map< pair<int,int>, TGraph* > gAvgEfficiencyPeakFitted; 
	map< pair<int,int>, TGraph* > gRelativeAvgEfficiencyPeakFitted; 
	mgAvgEfficiencyPeakFitted->SetMinimum(0.3);
	mgRelativeAvgEfficiencyPeakFitted->SetMinimum(0.3);
	mgAvgEfficiencyPeakFitted->SetMaximum(1.1);
	mgRelativeAvgEfficiencyPeakFitted->SetMaximum(1.1);

	TMultiGraph *mgControlAreaEfficiency = new TMultiGraph("mgControlAreaEfficiency","ControlArea Area Average Efficiency (Fit Method); Integrated Luminosity (fb^{-1}); Average Efficiency");
	TMultiGraph *mgRelativeControlAreaEfficiency = new TMultiGraph("mgRelativeControlAreaEfficiency","ControlArea Area Relative Average Efficiency (Fit Method); Integrated Luminosity (fb^{-1}); RelativeAverage Efficiency");
	map< pair<int,int>, TGraph* > gControlAreaEfficiency; 
	map< pair<int,int>, TGraph* > gRelativeControlAreaEfficiency; 
	mgControlAreaEfficiency->SetMinimum(0.3);
	mgRelativeControlAreaEfficiency->SetMinimum(0.3);
	mgControlAreaEfficiency->SetMaximum(1.1);
	mgRelativeControlAreaEfficiency->SetMaximum(1.1);

	TMultiGraph *mgPixelsUsedForAvgEfficiencyPeakFitted = new TMultiGraph("mgPixelsUsedForAvgEfficiencyPeakFitted","Pixels used for Avg Efficiency; Integrated Luminosity (fb^{-1}); Pixels Used");
	map< pair<int,int>, TGraph* > gPixelsUsedForAvgEfficiencyPeakFitted; 

	TMultiGraph *mgYAlignmentSensorEdge = new TMultiGraph("mgYAlignmentSensorEdge","Y Alignment @ Sensor Edge; Integrated Luminosity (fb^{-1}); mm");
	map< pair<int,int>, TGraph* > gYAlignmentSensorEdge; 
	
	for(auto & arm : arms ){
		for(auto & station : stations){
			int sector = 0;
			int station_id = 0;
			if (arm == 0) sector = 45;
			if (arm == 1) sector = 56;
			if (station == 0) station_id = 210;
			if (station == 2) station_id = 220;

			gAvgEfficiencyPeakFitted[pair<int,int>(arm,station)] = new TGraph();
			gAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetNameTitle(Form("gAvgEfficiencyPeakFitted_arm%i_st%i",arm,station), Form("LHC Sector %i %i FAR",sector,station_id));
			gRelativeAvgEfficiencyPeakFitted[pair<int,int>(arm,station)] = new TGraph();
			gRelativeAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetNameTitle(Form("gAvgEfficiencyPeakFitted_arm%i_st%i",arm,station), Form("LHC Sector %i %i FAR",sector,station_id));
			
			gControlAreaEfficiency[pair<int,int>(arm,station)] = new TGraph();
			gControlAreaEfficiency[pair<int,int>(arm,station)]->SetNameTitle(Form("gControlAreaEfficiency_arm%i_st%i",arm,station), Form("LHC Sector %i %i FAR",sector,station_id));
			gRelativeControlAreaEfficiency[pair<int,int>(arm,station)] = new TGraph();
			gRelativeControlAreaEfficiency[pair<int,int>(arm,station)]->SetNameTitle(Form("gControlAreaEfficiency_arm%i_st%i",arm,station), Form("LHC Sector %i %i FAR",sector,station_id));
			
			gPixelsUsedForAvgEfficiencyPeakFitted[pair<int,int>(arm,station)] = new TGraph();
			gPixelsUsedForAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetNameTitle(Form("gPixelsUsedForAvgEfficiencyPeakFitted_arm%i_st%i",arm,station), Form("LHC Sector %i %i FAR",sector,station_id));
			gYAlignmentSensorEdge[pair<int,int>(arm,station)] = new TGraph();
			gYAlignmentSensorEdge[pair<int,int>(arm,station)]->SetNameTitle(Form("gYAlignmentSensorEdge%i_st%i",arm,station), Form("Y Alignment LHC Sector %i %i FAR",sector,station_id));

			for(auto & run : runs){
				if(firstRun == 0) firstRun = run;
				if(missingRuns[run]) continue;
				if(arm == 0 && station == 0 && run == 315840) continue;
				if(arm == 1 && station == 2 && run == 324747) continue;
				
				gAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetPoint(gAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->GetN(),runLumiMap[run],avgEfficiencyPeakFitted[pair<int,int>(arm,station)][run]);
				gRelativeAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetPoint(gRelativeAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->GetN(),runLumiMap[run],avgEfficiencyPeakFitted[pair<int,int>(arm,station)][run]/avgEfficiencyPeakFitted[pair<int,int>(arm,station)][firstRun]);
			
				gControlAreaEfficiency[pair<int,int>(arm,station)]->SetPoint(gControlAreaEfficiency[pair<int,int>(arm,station)]->GetN(),runLumiMap[run],controlAreaEfficiency[pair<int,int>(arm,station)][run]);
				gRelativeControlAreaEfficiency[pair<int,int>(arm,station)]->SetPoint(gRelativeControlAreaEfficiency[pair<int,int>(arm,station)]->GetN(),runLumiMap[run],controlAreaEfficiency[pair<int,int>(arm,station)][run]/controlAreaEfficiency[pair<int,int>(arm,station)][firstRun]);
			
				gPixelsUsedForAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetPoint(gPixelsUsedForAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->GetN(),runLumiMap[run],pixelsUsedForAvgEfficiencyPeakFitted[pair<int,int>(arm,station)][run]);
				gYAlignmentSensorEdge[pair<int,int>(arm,station)]->SetPoint(gYAlignmentSensorEdge[pair<int,int>(arm,station)]->GetN(),runLumiMap[run],yAlignment[pair<int,int>(arm,station)][run]);
			}

			//Graph beauty farm

			gAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetLineWidth(2);
			gRelativeAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetLineWidth(2);

			gControlAreaEfficiency[pair<int,int>(arm,station)]->SetLineWidth(2);
			gRelativeControlAreaEfficiency[pair<int,int>(arm,station)]->SetLineWidth(2);
			gPixelsUsedForAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetLineWidth(2);

			gYAlignmentSensorEdge[pair<int,int>(arm,station)]->SetLineWidth(2);
			if (station == 0){
				gAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetMarkerStyle(22);
				gRelativeAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetMarkerStyle(22);
				gControlAreaEfficiency[pair<int,int>(arm,station)]->SetMarkerStyle(22);
				gRelativeControlAreaEfficiency[pair<int,int>(arm,station)]->SetMarkerStyle(22);
				gPixelsUsedForAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetMarkerStyle(22);
				gYAlignmentSensorEdge[pair<int,int>(arm,station)]->SetMarkerStyle(22);	
				gAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetMarkerSize(1.3);
				gRelativeAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetMarkerSize(1.3);
				gControlAreaEfficiency[pair<int,int>(arm,station)]->SetMarkerSize(1.3);
				gRelativeControlAreaEfficiency[pair<int,int>(arm,station)]->SetMarkerSize(1.3);
				gPixelsUsedForAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetMarkerSize(1.3);
				gYAlignmentSensorEdge[pair<int,int>(arm,station)]->SetMarkerSize(1.3);				
			}else{
				gAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetMarkerStyle(kFullSquare);
				gRelativeAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetMarkerStyle(kFullSquare);
				gControlAreaEfficiency[pair<int,int>(arm,station)]->SetMarkerStyle(kFullSquare);
				gRelativeControlAreaEfficiency[pair<int,int>(arm,station)]->SetMarkerStyle(kFullSquare);
				gPixelsUsedForAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->SetMarkerStyle(kFullSquare);
				gYAlignmentSensorEdge[pair<int,int>(arm,station)]->SetMarkerStyle(kFullSquare);
			}
			// Saving Hists

			outputFile->cd();
			gAvgEfficiencyPeakFitted[pair<int,int>(arm,station)]->Write();

			// Adding to multigraph

			mgAvgEfficiencyPeakFitted->Add(gAvgEfficiencyPeakFitted[pair<int,int>(arm,station)],"PL");
			mgRelativeAvgEfficiencyPeakFitted->Add(gRelativeAvgEfficiencyPeakFitted[pair<int,int>(arm,station)],"PL");


			mgControlAreaEfficiency->Add(gControlAreaEfficiency[pair<int,int>(arm,station)],"PL");
			mgRelativeControlAreaEfficiency->Add(gRelativeControlAreaEfficiency[pair<int,int>(arm,station)],"PL");

			mgPixelsUsedForAvgEfficiencyPeakFitted->Add(gPixelsUsedForAvgEfficiencyPeakFitted[pair<int,int>(arm,station)],"PL");

			mgYAlignmentSensorEdge->Add(gYAlignmentSensorEdge[pair<int,int>(arm,station)],"PL");
		}
	}

	TCanvas *cAvgEfficiencyPeakFitted = new TCanvas("cAvgEfficiencyPeakFitted","AvgEfficiencyPeakFitted");
	mgAvgEfficiencyPeakFitted->Draw("A pmc plc");
	cAvgEfficiencyPeakFitted->BuildLegend(0.1,0.1,0.35,0.3);
	TCanvas *cRelativeAvgEfficiencyPeakFitted = new TCanvas("cRelativeAvgEfficiencyPeakFitted","RelativeAvgEfficiencyPeakFitted");
	mgRelativeAvgEfficiencyPeakFitted->Draw("A pmc plc");
	cRelativeAvgEfficiencyPeakFitted->BuildLegend(0.1,0.1,0.35,0.3);

	TCanvas *cControlAreaEfficiency = new TCanvas("cControlAreaEfficiency","ControlAreaEfficiency");
	mgControlAreaEfficiency->Draw("A pmc plc");
	cControlAreaEfficiency->BuildLegend(0.1,0.1,0.35,0.3);
	TCanvas *cRelativeControlAreaEfficiency = new TCanvas("cRelativeControlAreaEfficiency","RelativeControlAreaEfficiency");
	mgRelativeControlAreaEfficiency->Draw("A pmc plc");
	cRelativeControlAreaEfficiency->BuildLegend(0.1,0.1,0.35,0.3);

	TCanvas *cPixelsUsedForAvgEfficiencyPeakFitted = new TCanvas("cPixelsUsedForAvgEfficiencyPeakFitted","PixelsUsedForAvgEfficiencyPeakFitted");
	mgPixelsUsedForAvgEfficiencyPeakFitted->Draw("A pmc plc");
	cPixelsUsedForAvgEfficiencyPeakFitted->BuildLegend(0.1,0.1,0.35,0.3);

	TCanvas *cYAlignmentSensorEdge = new TCanvas("cYAlignmentSensorEdge","cYAlignmentSensorEdge");
	mgYAlignmentSensorEdge->Draw("A pmc plc");
	cYAlignmentSensorEdge->BuildLegend(0.1,0.1,0.35,0.3);

	outputFile->Close();
}

void LineParser(ifstream& file_to_parse, vector<string>& lines){
  if (!file_to_parse){
    printf("ERROR in LineParser > file not found.\n");
    exit(1);
  }
  while (true){
    if (file_to_parse.eof())break;
    string line;
    getline(file_to_parse,line);
    if (line[0] != '#' && line.size() > 0){
      lines.push_back(line);
    }
  }
}

void Parser(ifstream& file_to_parse,vector<vector<string> >& spreadsheet){
  vector<string> lines;
  LineParser(file_to_parse,lines);
  for (vector<string>::iterator line=lines.begin();line!=lines.end();++line){
    string field_content;
    istringstream linestream(*line);
    if (!linestream.eof()){
      spreadsheet.push_back(vector<string>());
      while(1){
	getline(linestream,field_content,' ');
	spreadsheet.back().push_back(field_content);
	if (linestream.eof()){break;}
      }
    }
  }
}