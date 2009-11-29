// This macro can be used after the CheckAllIOVs.py to extract from the summaries
// the number of modules with LV/OV on or off as a function of the IOV.


#include <iostream>
#include <fstream>
#include <sstream>

#include "TString.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TLegend.h"

using namespace std;

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

struct Holder
{
  Holder()
  {
    layer = new vector<int>(20, 0);
  }
  void add(const int layerNum, const int side, const int modulesOff)
  {
    (*layer)[layerNum+side*10] = modulesOff;
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
    }
  }

  void SetBinContent(const int IOV, const int layerNum, const int side, const int modulesOff)
  {
    // cout << "Setting bin content for layer = "<< layerNum+side*10 << " for IOV = " << IOV << " to " << modulesOff << endl;
    // cout << "layer = " << layer << endl;
    // cout << "layer->size() = " << layer->size() << "(*layer)[layerNum+side*10] = " << (*layer)[layerNum+side*10] << endl;
    (*layer)[layerNum+side*10]->SetBinContent(IOV, modulesOff);
    // cout << "set" << endl;
  }

  TH1F* histo(const int layerNum, const int side) {
    // cout << "Retrieving histogram for layer = " << layerNum+side*10 << endl;
    return (*layer)[layerNum+side*10];
  }


  vector<TH1F*> * layer;
};

vector<vector<Holder> > extractFromFile( const string & fileName )
{
  ifstream inputFile(fileName.c_str());

  vector<vector<Holder> > holder;
  for( int i=0; i<2; ++i ) {
    holder.push_back(vector<Holder>());
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
    // End the loop if the lines are finished
    // else if( HVLV == 1 && subDet == "TID" && line == "" ) break;

    vector<string> tokenized(tokenize(line));
    if( !tokenized.empty() && tokenized[0] != "" ) {
      int index = 0;
      if( tokenized.size() == 5 ) {
	// cout << tokenized[0] << endl;
	subDet = tokenized[0];
	++index;
      }

      // cout << "line = " << line << endl;

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

      // cout << tokenized[index] << ", " << tokenized[index+1] << endl;

      if( subDet == "TIB" ) {
	cout << "Filling HVLV = " << HVLV << ", 0, layerNum = " << layerNum << ", side = " << side << ", modulesOff = " << modulesOff << endl;
	holder[HVLV][0].add( layerNum, side, modulesOff );
      }
      else if( subDet == "TID" ) holder[HVLV][1].add( layerNum, side, modulesOff );
      else if( subDet == "TOB" ) holder[HVLV][2].add( layerNum, side, modulesOff );
      else if( subDet == "TEC" ) holder[HVLV][3].add( layerNum, side, modulesOff );

    }
  }

//   cout << "TIB layer 1 side 0 modules with HV off = " << holder[0][0].modules(1, 0) << endl;
//   cout << "TIB layer 1 side 0 modules with LV off = " << holder[1][0].modules(1, 0) << endl;
  cout << "TOB layer 1 side 0 modules with HV off = " << holder[0][2].modules(1, 0) << endl;
  cout << "TOB layer 1 side 0 modules with LV off = " << holder[1][2].modules(1, 0) << endl;
  return holder;
}

// Small function used to fill the histograms for the different layers of the different subDetectors
void fillHistos( vector<vector<Holder> > & it, vector<vector<HistoHolder> > & histos, const int firstLayer, const int totLayers,
		 const int doubleSidedLayers, const int HVLVid, const int subDetId, const int iov )
{
  for( int layerNum = firstLayer; layerNum <= totLayers; ++layerNum ) {
    // cout << "filling histos for subDetId = " << subDetId << ", layer = " << layerNum << endl;
    histos[HVLVid][subDetId].SetBinContent( iov, layerNum, 0, it[HVLVid][subDetId].modules(layerNum, 0) );
    if( layerNum <= doubleSidedLayers ) {
      histos[HVLVid][subDetId].SetBinContent( iov, layerNum, 1, it[HVLVid][subDetId].modules(layerNum, 1) );
    }
  }
}

// Small function used to draw the histograms for the different layers of the different subDetectors
void drawHistos( TCanvas ** canvas, vector<vector<HistoHolder> > & histos, TH1F ** histoTracker, const int firstLayer, const int totLayers,
		 const int doubleSidedLayers, const int HVLVid, const int subDetId )
// 		 const int doubleSidedLayers, const int HVLVid, const int subDetId, TLegend * legend )
{
  int canvasCorrection = 0;
  TString option("");
  int lineColor = 2;
  // TString legendText("High voltage off");
  if( HVLVid == 1 ) {
    option = "SAME";
    lineColor = 1;
    // legendText = "Low voltage off";
  }
  if( firstLayer == 0 ) canvasCorrection = 1;
  for( int layerNum = firstLayer; layerNum <= totLayers; ++layerNum ) {
    canvas[subDetId]->cd(canvasCorrection+layerNum);
    TH1F * histo = histos[HVLVid][subDetId].histo( layerNum, 0 );
    histo->Draw(option);
    histo->SetLineColor(lineColor);
    // legend->AddEntry(histo, legendText);

    histoTracker[HVLVid]->Add( histo );
    if( layerNum <= doubleSidedLayers ) {
      canvas[subDetId]->cd(canvasCorrection+totLayers+layerNum);
      histo = histos[HVLVid][subDetId].histo( layerNum, 1 );
      histo->Draw(option);
      histo->SetLineColor(lineColor);
      // legend->AddEntry(histo, legendText);
      histoTracker[HVLVid]->Add( histo );
    }
  }
  // if( HVLVid == 1 ) legend->Draw("SAME");
}

void ExtracTrends()
{
  ifstream listFile("list.txt");
  string fileName;

  vector<vector<vector<Holder> > > holderVsIOV;

  while( getline(listFile, fileName) ) {

    size_t first = fileName.find("__FROM");
    size_t last = fileName.find("_TO");
    string subString(fileName.substr(first+7, last-(first+7)));
    cout << "substr = " << subString << endl;
    holderVsIOV.push_back(extractFromFile(fileName));
  }

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
      fillHistos( *it,    histos,          0,         2,                 1, HVLVid,        1, iov ); // TID (doubleSided = 1 because it starts from 1 and <= is used)
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
//   if( HVLVid == 0 ) {
  canvas[0] = new TCanvas("TIB HV status", "HVstatus", 1000, 800);
  canvas[1] = new TCanvas("TID HV status", "HVstatus", 1000, 800);
  canvas[2] = new TCanvas("TOB HV status", "HVstatus", 1000, 800);
  canvas[3] = new TCanvas("TEC HV status", "HVstatus", 1000, 800);

//   TLegend * legend = new TLegend(0.7,0.71,0.98,1.);
//   legend->SetTextSize(0.02);
  // legend->SetFillColor(0); // Have a white background

//   }
//     if( HVLVid == 1 ) {
//       canvas[0] = new TCanvas("TIB LV status", "LVstatus", 1000, 800);
//       canvas[1] = new TCanvas("TID LV status", "LVstatus", 1000, 800);
//       canvas[2] = new TCanvas("TOB LV status", "LVstatus", 1000, 800);
//       canvas[3] = new TCanvas("TEC LV status", "LVstatus", 1000, 800);
//     }
  canvas[0]->Divide(4,2);
  canvas[1]->Divide(2,2);
  canvas[2]->Divide(6,2);
  canvas[3]->Divide(9,2);

  for( int HVLVid = 0; HVLVid < 2; ++HVLVid ) {

    // par:     canvas, histos, histoTracker, firstLayer, totLayers, doubleSidedLayers, HVLVid, subDetId, iov
//     drawHistos( canvas, histos, histoTracker,          1,         4,                 2, HVLVid,        0, legend ); // TIB
//     drawHistos( canvas, histos, histoTracker,          0,         2,                 1, HVLVid,        1, legend ); // TID (doubleSided = 1 because it starts from 1 and <= is used)
//     drawHistos( canvas, histos, histoTracker,          1,         6,                 2, HVLVid,        2, legend ); // TOB
//     drawHistos( canvas, histos, histoTracker,          1,         9,                 9, HVLVid,        3, legend ); // TEC
    drawHistos( canvas, histos, histoTracker,          1,         4,                 2, HVLVid,        0 ); // TIB
    drawHistos( canvas, histos, histoTracker,          0,         2,                 1, HVLVid,        1 ); // TID (doubleSided = 1 because it starts from 1 and <= is used)
    drawHistos( canvas, histos, histoTracker,          1,         6,                 2, HVLVid,        2 ); // TOB
    drawHistos( canvas, histos, histoTracker,          1,         9,                 9, HVLVid,        3 ); // TEC

    canvas[0]->Draw();
    canvas[1]->Draw();
    canvas[2]->Draw();
    canvas[3]->Draw();
  }
  allCanvas[0]->cd();
  histoTracker[0]->Draw();
  histoTracker[0]->SetLineColor(2);
  // allCanvas[1]->cd();
  histoTracker[1]->Draw("SAME");
  // histoTracker[1]->SetLineColor(1);
  TLegend * legend2 = new TLegend(0.7,0.71,0.98,1.);
  legend2->SetTextSize(0.02);
  // legend2->SetFillColor(0); // Have a white background
  legend2->AddEntry(histoTracker[0], "High Voltage off");
  legend2->AddEntry(histoTracker[1], "Low Voltage off");
  legend2->Draw("SAME");

  allCanvas[0]->Draw();
  // allCanvas[1]->Draw();
}
