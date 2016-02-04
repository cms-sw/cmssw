// This macro can be used after the CheckAllIOVs.py to extract from the summaries
// the number of modules with LV/HV on or off as a function of the IOV.


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>

#include "TString.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TVectorD.h"
#include "TDatime.h"
#include "TFile.h"
#include "TStyle.h"
#include "TROOT.h"

using namespace std;

void Tokenize(const string& str,
	      vector<string>& tokens,
	      const string& delimiters = " ")
{
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  string::size_type pos     = str.find_first_of(delimiters, lastPos);
  
  while (string::npos != pos || string::npos != lastPos) {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}

// Function to tokenize a string
vector<string> tokenize(const string & line)
{
  stringstream ss(line);
  vector<string> tokenized;
  while(ss) {
    string tok;
    ss >> tok;
    tokenized.push_back(tok);
  }
  return tokenized;
}

double * duplicateForGraph(const unsigned int size, const Float_t * summedValues)
{
  double * summedValuesArray = new double[size*2-1];
  for( unsigned int i=0; i<size; ++i ) {
    summedValuesArray[2*i] = summedValues[i];
    // cout << "summedValuesArray["<<2*i<<"] = " << summedValuesArray[2*i] << endl;
    if( i != size-1 ) {
      summedValuesArray[2*i+1] = summedValues[i];
    }
  }
  return summedValuesArray;
}

struct Holder
{
  Holder() :
    layer( new vector<int>(20, 0) ),
    iov(0)
  {}
  // void add(const int layerNum, const int side, const int modulesOff, const double & inputIOV)
  void add(const int layerNum, const int side, const int modulesOff, const double & timeInSeconds)
  {
    (*layer)[layerNum+side*10] = modulesOff;
    iov = timeInSeconds;
  }
  int modules(const int layerNum, const int side)
  {
    if( layerNum+side*10 < int(layer->size()) ) {
      return (*layer)[layerNum+side*10];
    }
    else {
      cout << "ERROR: layerNum+side*10 = " << layerNum+side*10 << " bigger than number of layers = " << layer->size() << endl;
      cout << "Returning 0" << endl;
    }
    return 0;
  }
  vector<int> * layer;
  double iov;
};


struct HistoHolder
{
  HistoHolder(const TString & subDet, const int IOVs)
  {
    layer = new vector<TH1F*>;
    fillLayers(subDet, IOVs);
    fillLayers(subDet, IOVs, "stereo");
  }

  void fillLayers(const TString & subDet, const int IOVs, const TString & addToName = "")
  {
    for( int i=0; i<10; ++i ) {
      stringstream ss;
      ss << i;
      layer->push_back(new TH1F(subDet+ss.str()+addToName, subDet+ss.str()+addToName, IOVs, 0, IOVs));
      timeVector.push_back(new vector<double>);
      valueVector.push_back(new vector<double>);
    }
  }

  void SetBinContent(const int IOV, const int layerNum, const int side, const int modulesOff, const double & time)
  {
    // Skip the zero time case
    // cout << "IOV = " << IOV << ", layerNum = " << layerNum << ", side = " << side << ", modulesOff = " << modulesOff << ", time = " << time << endl;
    // if( time == 0 ) cout << "TIME == 0" << endl;

    (*layer)[layerNum+side*10]->SetBinContent(IOV, modulesOff);
    timeVector[layerNum+side*10]->push_back( time );
    valueVector[layerNum+side*10]->push_back( modulesOff );
  }

  TH1F* histo(const int layerNum, const int side)
  {
    return (*layer)[layerNum+side*10];
  }

  void removeZeros(const int layerNum, const int side)
  {
    // Remove all the times == 0
    vector<double>::iterator it = find(timeVector[layerNum+10*side]->begin(), timeVector[layerNum+10*side]->end(), 0);
    while( it != timeVector[layerNum+10*side]->end() ) {
      timeVector[layerNum+10*side]->erase(it);
      valueVector[layerNum+10*side]->erase(valueVector[layerNum+10*side]->begin()+distance(timeVector[layerNum+10*side]->begin(), it));
      it = find(timeVector[layerNum+10*side]->begin(), timeVector[layerNum+10*side]->end(), 0);
    }
  }

  double * time(const int layerNum, const int side)
  {
    // Take twice the values. We propagate the previous point to the next time,
    // so that the graph will display in a way similar to a TH1F (but with the correct spacing
    // between times).
    unsigned int size = timeVector[layerNum+side*10]->size();
    double * timeV = new double[2*size-1];
    for( unsigned int i=0; i<size; ++i ) {
      timeV[2*i] = (*(timeVector[layerNum+side*10]))[i];
      // Put the next time, which will correspond to the value of the current time
      if( i != size-1 ) {
	timeV[2*i+1] = (*(timeVector[layerNum+side*10]))[i+1];
      }
    }
    return timeV;
  }

  double * value(const int layerNum, const int side)
  {
    unsigned int size = valueVector[layerNum+side*10]->size();
    double * valueV = new double[2*size-1];
    for( unsigned int i=0; i<size; ++i ) {
      valueV[2*i] = (*(valueVector[layerNum+side*10]))[i];

      // Put the same value, which will correspond to the next time
      if( i != size-1 ) {
	valueV[2*i+1] = (*(valueVector[layerNum+side*10]))[i];
      }
    }
    return valueV;
  }

  unsigned int getSize(const int layerNum, const int side)
  {
    return 2*(timeVector[layerNum+side*10]->size())-1;
  }

  vector<TH1F*> * layer;
  vector<vector<double> *> timeVector;
  vector<vector<double> *> valueVector;
};

void drawHistoTracker(TH1F* histo, const TString option, const unsigned int color, vector<vector<HistoHolder> > & histos)
{
  // +1 because the array returned by the histogram starts from the underflow bin
  Float_t * summedValues = histo->GetArray()+1;
  unsigned int size = histo->GetNbinsX();
  double * summedValuesArray = duplicateForGraph(size, summedValues);
  TGraph * graph = new TGraph(histos[0][0].getSize(1, 0), histos[0][0].time(1, 0), summedValuesArray);
  graph->Draw(option);
  graph->SetLineColor(color);
  graph->GetXaxis()->SetTimeDisplay(1);
  graph->GetXaxis()->SetLabelOffset(0.02);
  graph->GetXaxis()->SetTimeFormat("#splitline{  %d}{%H:%M}");
  graph->GetXaxis()->SetTimeOffset(0,"gmt");
  graph->GetYaxis()->SetRangeUser(0,16000);
  graph->GetXaxis()->SetTitle("day/hour");
  graph->GetXaxis()->SetTitleSize(0.03);
  graph->GetXaxis()->SetTitleColor(kBlack);
  graph->GetXaxis()->SetTitleOffset(1.80);
  graph->GetYaxis()->SetTitle("number of modules off");
  graph->GetYaxis()->SetTitleSize(0.03);
  graph->GetYaxis()->SetTitleColor(kBlack);
  graph->GetYaxis()->SetTitleOffset(1.80);
  graph->SetTitle();
}

vector<vector<Holder> > extractFromFile( const string & fileName, const string & date )
{
  ifstream inputFile(fileName.c_str());

  vector<string> tokens;
  Tokenize(date, tokens, "_");

  unsigned int day = 0;
  stringstream sDay(tokens[2]); // day
  sDay >> day;
  unsigned int hour = 0;
  stringstream sHour(tokens[3]); // hour
  sHour >> hour;
  unsigned int minute = 0;
  stringstream sMinute(tokens[4]); // minute
  sMinute >> minute;
  unsigned int second = 0;
  stringstream sSecond(tokens[5]); // second
  sSecond >> second;
  unsigned int year = 0;
  stringstream sYear(tokens[6]); // year
  sYear >> year;

  // Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
  std::map<string, unsigned int> monthsToNumbers;
  monthsToNumbers.insert(make_pair("Jan", 1));
  monthsToNumbers["Feb"] = 2;
  monthsToNumbers["Mar"] = 3;
  monthsToNumbers["Apr"] = 4;
  monthsToNumbers["May"] = 5;
  monthsToNumbers["Jun"] = 6;
  monthsToNumbers["Jul"] = 7;
  monthsToNumbers["Aug"] = 8;
  monthsToNumbers["Sep"] = 9;
  monthsToNumbers["Oct"] = 10;
  monthsToNumbers["Nov"] = 11;
  monthsToNumbers["Dec"] = 12;

  std::map<string, unsigned int>::iterator month = monthsToNumbers.find(tokens[1]);
  TDatime date1(year, month->second, day, hour, minute, second);

  double timeInSeconds = date1.Convert();

  vector<vector<Holder> > holder;
  // HV/LV = 2 cases
  for( int i=0; i<2; ++i ) {
    holder.push_back(vector<Holder>());
    // 4 possible subdetectors
    for( int j=0; j<4; ++j ) {
      holder[i].push_back(Holder());
    }
  }

  int HVLV = 0;
  string line;

  bool start = false;

  string subDet = "";
  while( getline(inputFile, line) ) {

    // Skip empty lines
    if( line == "" ) continue;

    if( line.find("subDet") != string::npos ) {
      start = true;
    }
    // Skip the rest until you find the starting line
    if( start == false ) continue;

    if( line.find("Summary") != string::npos ) {
      // Skip also the next two lines
      getline(inputFile, line);
      getline(inputFile, line);
      ++HVLV;
    }
    else if( line.find("%MSG") != string::npos || line.find("DummyCondObjPrinter") != string::npos ) continue;

    vector<string> tokenized(tokenize(line));
    if( !tokenized.empty() && tokenized[0] != "" ) {
      int index = 0;
      if( tokenized.size() == 5 ) {
	subDet = tokenized[0];
	++index;
      }

      // Extract the relevant quantities
      stringstream ss1( tokenized[index] );
      int layerNum = 0;
      ss1 >> layerNum;
      stringstream ss2( tokenized[index+1] );
      int side = 0;
      ss2 >> side;
      stringstream ss3( tokenized[index+2] );
      int modulesOff = 0;
      ss3 >> modulesOff;

      if( subDet == "TIB" )      holder[HVLV][0].add( layerNum, side, modulesOff, timeInSeconds );
      else if( subDet == "TID" ) holder[HVLV][1].add( layerNum, side, modulesOff, timeInSeconds );
      else if( subDet == "TOB" ) holder[HVLV][2].add( layerNum, side, modulesOff, timeInSeconds );
      else if( subDet == "TEC" ) holder[HVLV][3].add( layerNum, side, modulesOff, timeInSeconds );

    }
  }

  return holder;
}

// Small function used to fill the histograms for the different layers of the different subDetectors
void fillHistos( vector<vector<Holder> > & it, vector<vector<HistoHolder> > & histos, const int firstLayer, const int totLayers,
		 const int doubleSidedLayers, const int HVLVid, const int subDetId, const int iov )
{
  for( int layerNum = firstLayer; layerNum <= totLayers; ++layerNum ) {
    histos[HVLVid][subDetId].SetBinContent( iov, layerNum, 0, it[HVLVid][subDetId].modules(layerNum, 0), it[HVLVid][subDetId].iov );
    if( layerNum <= doubleSidedLayers ) {
      histos[HVLVid][subDetId].SetBinContent( iov, layerNum, 1, it[HVLVid][subDetId].modules(layerNum, 1), it[HVLVid][subDetId].iov );
    }
  }
}

// Small function used to draw the histograms for the different layers of the different subDetectors
void drawHistos( TCanvas ** canvas, vector<vector<HistoHolder> > & histos, TH1F ** histoTracker, const int firstLayer, const int totLayers,
		 const int doubleSidedLayers, const int HVLVid, const int subDetId )
{
  TString option("AL");
  int lineColor = 2;
  if( HVLVid == 1 ) {
    // option = "SAME";
    // No SAME option for the TGraph (it contains the A, and it would mean a different thing)
    // They will always be drawn as if SAME is selected.
    option = "L";
    lineColor = 1;
  }
  for( int layerNum = firstLayer; layerNum <= totLayers; ++layerNum ) {
    // First of all remove all the times = 0
    histos[HVLVid][subDetId].removeZeros(layerNum, 0);
    canvas[subDetId]->cd(layerNum);
    TH1F * histo = histos[HVLVid][subDetId].histo( layerNum, 0 );

    // cout << "[0] = " << histo->GetArray()[0] << ", [1] = " << histo->GetArray()[1] << ",[2] = " << histo->GetArray()[2] << endl;

    TGraph * graph = new TGraph(histos[HVLVid][subDetId].getSize(layerNum, 0), histos[HVLVid][subDetId].time(layerNum, 0), histos[HVLVid][subDetId].value(layerNum, 0));
    graph->SetTitle(histo->GetTitle());
    graph->SetLineColor(lineColor);
    graph->Draw(option);
    graph->SetMarkerColor(lineColor);
    graph->GetXaxis()->SetTimeDisplay(1); 
    graph->GetXaxis()->SetTimeFormat("#splitline{  %d}{%H:%M}");
    graph->GetXaxis()->SetLabelOffset(0.02);
    graph->GetXaxis()->SetTimeOffset(0,"gmt");
    histoTracker[HVLVid]->Add( histo );

    if( layerNum <= doubleSidedLayers ) {
      histos[HVLVid][subDetId].removeZeros(layerNum, 1);
      canvas[subDetId]->cd(totLayers+layerNum);
      histo = histos[HVLVid][subDetId].histo( layerNum, 1 );
      TGraph * graphStereo = new TGraph(histos[HVLVid][subDetId].getSize(layerNum, 1), histos[HVLVid][subDetId].time(layerNum, 1), histos[HVLVid][subDetId].value(layerNum, 1));
      graphStereo->SetTitle(histo->GetTitle());
      graphStereo->SetLineColor(lineColor);
      graphStereo->SetMarkerColor(lineColor);
      graphStereo->Draw(option);

      histoTracker[HVLVid]->Add( histo );
    }
  }
}

void clearEmptyFiles(vector<vector<vector<Holder> > > & holderVsIOV)
{
  for( vector<vector<vector<Holder> > >::iterator it1 = holderVsIOV.begin(); it1 != holderVsIOV.end(); ++it1 ) {
    if( (*it1)[0][0].iov == 0 ) {
      cout << "Removing iov = 0" << endl;
      it1 = holderVsIOV.erase(it1);
    }
  }
}

void ExtractTrends()
{
  gROOT->SetStyle("Plain");
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetTitleFillColor(kWhite);
  gStyle->SetTitleColor(kWhite);

  TFile * outputFile = new TFile("trends.root", "RECREATE");

  ifstream listFile("list.txt");
  string fileName;

  vector<vector<vector<Holder> > > holderVsIOV;

  while( getline(listFile, fileName) ) {

    size_t first = fileName.find("__FROM");
    size_t last = fileName.find("_TO");
    string subString(fileName.substr(first+7, last-(first+7)));
    holderVsIOV.push_back(extractFromFile(fileName, subString));
  }

  // Clear the residuals from empty files
  clearEmptyFiles(holderVsIOV);

  // Create histograms for each subDet and layer and fill them
  vector<vector<HistoHolder> > histos;
  for( int i=0; i<2; ++i ) {
    histos.push_back(vector<HistoHolder>());
    string HVLVstring;
    if( i == 0 ) HVLVstring = "HV";
    else HVLVstring = "LV";
    histos[i].push_back(HistoHolder("TIB_"+HVLVstring, holderVsIOV.size()));
    histos[i].push_back(HistoHolder("TID_"+HVLVstring, holderVsIOV.size()));
    histos[i].push_back(HistoHolder("TOB_"+HVLVstring, holderVsIOV.size()));
    histos[i].push_back(HistoHolder("TEC_"+HVLVstring, holderVsIOV.size()));
  }

  vector<vector<vector<Holder> > >::iterator it = holderVsIOV.begin();
  int iov = 1;
  for( ; it != holderVsIOV.end(); ++it, ++iov ) {

    // HV status
    // ---------
    for( int HVLVid = 0; HVLVid < 2; ++HVLVid ) {

      // par:     holder, histos, firstLayer, totLayers, doubleSidedLayers, HVLVid, subDetId, iov
      fillHistos( *it,    histos,          1,         4,                 2, HVLVid,        0, iov ); // TIB
      fillHistos( *it,    histos,          1,         3,                 3, HVLVid,        1, iov ); // TID
      fillHistos( *it,    histos,          1,         6,                 2, HVLVid,        2, iov ); // TOB
      fillHistos( *it,    histos,          1,         9,                 9, HVLVid,        3, iov ); // TEC
    }
  }

  TCanvas *allCanvas[2];
  allCanvas[0] = new TCanvas("Tracker HV status", "HVstatus", 1000, 800);
  allCanvas[1] = new TCanvas("Tracker LV status", "LVstatus", 1000, 800);
  TH1F *histoTracker[2];
  histoTracker[0] = new TH1F("Tracker status HV", "TrackerHVstatus", holderVsIOV.size(), 0, holderVsIOV.size());
  histoTracker[1] = new TH1F("Tracker status LV", "TrackerLVstatus", holderVsIOV.size(), 0, holderVsIOV.size());

  // Loop again on the HVLV and draw the histograms

  TCanvas *canvas[4];
  canvas[0] = new TCanvas("TIB HV status", "HVstatus", 1000, 800);
  canvas[1] = new TCanvas("TID HV status", "HVstatus", 1000, 800);
  canvas[2] = new TCanvas("TOB HV status", "HVstatus", 1000, 800);
  canvas[3] = new TCanvas("TEC HV status", "HVstatus", 1000, 800);

  canvas[0]->Divide(4,2);
  canvas[1]->Divide(3,2);
  canvas[2]->Divide(6,2);
  canvas[3]->Divide(9,2);

  for( int HVLVid = 0; HVLVid < 2; ++HVLVid ) {

    // par:     canvas, histos, histoTracker, firstLayer, totLayers, doubleSidedLayers, HVLVid, subDetId, iov
    drawHistos( canvas, histos, histoTracker,          1,         4,                 2, HVLVid,        0 ); // TIB
    drawHistos( canvas, histos, histoTracker,          1,         3,                 3, HVLVid,        1 ); // TID
    drawHistos( canvas, histos, histoTracker,          1,         6,                 2, HVLVid,        2 ); // TOB
    drawHistos( canvas, histos, histoTracker,          1,         9,                 9, HVLVid,        3 ); // TEC

    outputFile->cd();
    canvas[0]->Draw();
    canvas[1]->Draw();
    canvas[2]->Draw();
    canvas[3]->Draw();
    canvas[0]->Write();
    canvas[1]->Write();
    canvas[2]->Write();
    canvas[3]->Write();
  }
  allCanvas[0]->cd();

  histoTracker[0]->SetLineColor(2);

  drawHistoTracker( histoTracker[1], "AL", 1, histos);
  drawHistoTracker( histoTracker[0], "L", 2, histos);

  TLegend * legend2 = new TLegend(0.715,0.87,0.98,1,NULL,"brNDC");
  legend2->SetTextSize(0.035);
  legend2->SetFillColor(0); // Have a white background
  legend2->AddEntry(histoTracker[0], "High Voltage off");
  legend2->AddEntry(histoTracker[1], "Low Voltage off");
  legend2->Draw("SAME");

  allCanvas[0]->Draw();
  allCanvas[0]->Write();

  outputFile->Write();
  outputFile->Close();
}
