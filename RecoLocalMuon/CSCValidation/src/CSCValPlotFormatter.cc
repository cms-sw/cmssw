#include "RecoLocalMuon/CSCValidation/src/CSCValPlotFormatter.h"

  using namespace std;

  // constructor
  CSCValPlotFormatter::CSCValPlotFormatter(){

    cout << "CSCValPlotFormatter is making pretty pictures..." << endl;

  }

  // destructor
  CSCValPlotFormatter::~CSCValPlotFormatter(){

  }


  void CSCValPlotFormatter::makePlots(map<string,pair<TH1*,string> > tM){

    theMap = tM;

    string theName;
    string theTitle;
    string theFolder;
    string savename;
    TH1* theHisto;

    map<string,pair<TH1*,string> >::const_iterator mapit;
    for (mapit = theMap.begin(); mapit != theMap.end(); mapit++){
      theFolder = (*mapit).second.second;
      theName = (*mapit).first;
      theHisto = (*mapit).second.first;
      savename = theFolder + "_" + theName;
      if (theName == "hOWires" || theName == "hOStrips" || theName == "hORecHits" || theName == "hOSegments"){
         make2DTemperaturePlot(theHisto, theName);
      }
      else if ( theFolder == "Digis") makeGif(theHisto, savename, 1110);
      else if ( theFolder == "PedestalNoise") makeGif(theHisto, savename, 1110);
      else if ( theFolder == "recHits") makeGif(theHisto, savename, 1110);
      else if ( theFolder == "Resolution") makeGif(theHisto, savename, 1110);
      else if ( theFolder == "Segments") makeGif(theHisto, savename, 1110);
      //else if ( theFolder == "SignalProfile") makeGif(theHisto, savename, 0);
      //else if ( theFolder == "FirstTBADC") makeGif(theHisto, savename, 1110);
      else if ( theFolder == "Efficiency")  makeEffGif(theHisto, savename);
    }

    // Nikolai's plots
    nikolaiMacro(tM,1);
    nikolaiMacro(tM,2);
    nikolaiMacro(tM,3);
    nikolaiMacro(tM,4);

  }

  void CSCValPlotFormatter::makeComparisonPlots(map<string,pair<TH1*,string> > tM, string refFileName){

    theMap = tM;

    gStyle = getStyle();

    TFile *theFile = new TFile(refFileName.c_str(),"READ");

    string theName;
    string theTitle;
    string theFolder;
    string directory;
    TH1* histo1;
    TH1* histo2;


    map<string,pair<TH1*,string> >::const_iterator mapit;
    for (mapit = theMap.begin(); mapit != theMap.end(); mapit++){
      theFolder = (*mapit).second.second;
      theName = (*mapit).first;
      histo1 = (*mapit).second.first;
      TCanvas *c = new TCanvas("c","my canvas",1);
      directory = theFolder + "/" + theName;
      histo2 = (TH1*)theFile->Get(directory.c_str());
      gStyle->SetHistFillColor(92);
      histo1->UseCurrentStyle();
      histo1->GetXaxis()->SetLabelSize(0.06);
      histo1->GetYaxis()->SetLabelSize(0.06);
      histo1->GetXaxis()->SetTitleOffset(0.7);
      histo1->GetXaxis()->SetTitleSize(0.06);
      histo1->GetXaxis()->SetNdivisions(208,kTRUE);
      histo1->Draw();
      if (histo2){
        histo2->UseCurrentStyle();
        histo2->Draw("same e");
      }
      string savename = theFolder + "_" + theName + "_comparison" + ".gif";
      c->Update();
      c->Print(savename.c_str());
    }


  }

  void CSCValPlotFormatter::makeGlobalScatterPlots(TTree* t1, string type){

    gStyle = getStyle(0);
    TGraph *result;
    int np = 0;
    if (type == "rechits"){
      TCanvas *c = new TCanvas("c","my canvas",1);
      c->SetCanvasSize(700,700);
      //station 1 +side
      np = t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 1 && rHpos.endcap == 1","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("RecHit Global Position (Station +1)");
        result->Draw("AP");
        drawChamberLines(1,1);
        c->Update();
        c->Print("rHglobal_station_+1.gif");
      }
      //station 2 +side
      np = t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 2 && rHpos.endcap == 1","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("RecHit Global Position (Station +2)");
        result->Draw("AP");
        drawChamberLines(2,1);
        c->Update();
        c->Print("rHglobal_station_+2.gif");
      }
      //station 3 +side
      np = t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 3 && rHpos.endcap == 1","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("RecHit Global Position (Station +3)");
        result->Draw("AP");
        drawChamberLines(3,1);
        c->Update();
        c->Print("rHglobal_station_+3.gif");
      }
      //station 4 +side
      np = t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 4 && rHpos.endcap == 1","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("RecHit Global Position (Station +4)");
        result->Draw("AP");
        drawChamberLines(4,1);
        c->Update();
        c->Print("rHglobal_station_+4.gif");
      }

      //station 1 -side
      np = t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 1 && rHpos.endcap == 2","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("RecHit Global Position (Station -1)");
        result->Draw("AP");
        drawChamberLines(1,1);
        c->Update();
        c->Print("rHglobal_station_-1.gif");
      }
      //station 2 +side
      np = t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 2 && rHpos.endcap == 2","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("RecHit Global Position (Station -2)");
        result->Draw("AP");
        drawChamberLines(2,1);
        c->Update();
        c->Print("rHglobal_station_-2.gif");
      }
      //station 3 +side
      np = t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 3 && rHpos.endcap == 2","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("RecHit Global Position (Station -3)");
        result->Draw("AP");
        drawChamberLines(3,1);
        c->Update();
        c->Print("rHglobal_station_-3.gif");
      }
      //station 4 +side
      np = t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 4 && rHpos.endcap == 2","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("RecHit Global Position (Station -4)");
        result->Draw("AP");
        drawChamberLines(4,1);
        c->Update();
      c->Print("rHglobal_station_-4.gif");
      }
    }

    // SEGMENTS
    if (type == "segments"){
      TCanvas *c = new TCanvas("c","my canvas",1);
      c->SetCanvasSize(700,700);
      //station 1 +side
      np = t1->Draw("segpos.globaly:segpos.globalx","segpos.station == 1 && segpos.endcap == 1","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("Segment Global Position (Station +1)");
        result->Draw("AP");
        drawChamberLines(1,1);
        c->Update();
        c->Print("Sglobal_station_+1.gif");
      }
      //station 2 +side
      np = t1->Draw("segpos.globaly:segpos.globalx","segpos.station == 2 && segpos.endcap == 1","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("Segment Global Position (Station +2)");
        result->Draw("AP");
        drawChamberLines(2,1);
        c->Update();
        c->Print("Sglobal_station_+2.gif");
      }
      //station 3 +side
      np = t1->Draw("segpos.globaly:segpos.globalx","segpos.station == 3 && segpos.endcap == 1","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("Segment Global Position (Station +3)");
        result->Draw("AP");
        drawChamberLines(3,1);
        c->Update();
        c->Print("Sglobal_station_+3.gif");
      }
      //station 4 +side
      np = t1->Draw("segpos.globaly:segpos.globalx","segpos.station == 4 && segpos.endcap == 1","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("Segment Global Position (Station +4)");
        result->Draw("AP");
        drawChamberLines(4,1);
        c->Update();
        c->Print("Sglobal_station_+4.gif");
      }

      //station 1 -side
      np = t1->Draw("segpos.globaly:segpos.globalx","segpos.station == 1 && segpos.endcap == 2","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("Segment Global Position (Station -1)");
        result->Draw("AP");
        drawChamberLines(1,1);
        c->Update();
        c->Print("Sglobal_station_-1.gif");
      }
      //station 2 +side
      np = t1->Draw("segpos.globaly:segpos.globalx","segpos.station == 2 && segpos.endcap == 2","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("Segment Global Position (Station -2)");
        result->Draw("AP");
        drawChamberLines(2,1);
        c->Update();
        c->Print("Sglobal_station_-2.gif");
      }
      //station 3 +side
      np = t1->Draw("segpos.globaly:segpos.globalx","segpos.station == 3 && segpos.endcap == 2","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("Segment Global Position (Station -3)");
        result->Draw("AP");
        drawChamberLines(3,1);
        c->Update();
        c->Print("Sglobal_station_-3.gif");
      }
      //station 4 +side
      np = t1->Draw("segpos.globaly:segpos.globalx","segpos.station == 4 && segpos.endcap == 2","goff");
      if (np > 0){
        result = new TGraph(t1->GetSelectedRows(),t1->GetV2(),t1->GetV1());
        result->GetXaxis()->SetLimits(-720,720);
        result->GetYaxis()->SetLimits(-720,720);
        result->GetXaxis()->SetRangeUser(-720,720);
        result->GetYaxis()->SetRangeUser(-720,720);
        result->SetTitle("Segment Global Position (Station -4)");
        result->Draw("AP");
        drawChamberLines(4,1);
        c->Update();
      c->Print("Sglobal_station_-4.gif");
      }
    }

  }


  void CSCValPlotFormatter::make2DTemperaturePlot(TH1 *plot, string savename){

    gStyle->Reset();

    TCanvas *c = new TCanvas("c","my canvas",1);
    c->SetRightMargin(0.12);

    gStyle->SetPalette(1,0);

    plot->SetStats(false);

    plot->GetYaxis()->SetBinLabel(1,"ME- 4/2");
    plot->GetYaxis()->SetBinLabel(2,"ME- 4/1");
    plot->GetYaxis()->SetBinLabel(3,"ME- 3/2");
    plot->GetYaxis()->SetBinLabel(4,"ME- 3/1");
    plot->GetYaxis()->SetBinLabel(5,"ME- 2/2");
    plot->GetYaxis()->SetBinLabel(6,"ME- 2/1");
    plot->GetYaxis()->SetBinLabel(10,"ME- 1/1a");
    plot->GetYaxis()->SetBinLabel(7,"ME- 1/3");
    plot->GetYaxis()->SetBinLabel(8,"ME- 1/2");
    plot->GetYaxis()->SetBinLabel(9,"ME- 1/1b");
    plot->GetYaxis()->SetBinLabel(12,"ME+ 1/1b");
    plot->GetYaxis()->SetBinLabel(13,"ME+ 1/2");
    plot->GetYaxis()->SetBinLabel(14,"ME+ 1/3");
    plot->GetYaxis()->SetBinLabel(11,"ME+ 1/1a");
    plot->GetYaxis()->SetBinLabel(15,"ME+ 2/1");
    plot->GetYaxis()->SetBinLabel(16,"ME+ 2/2");
    plot->GetYaxis()->SetBinLabel(17,"ME+ 3/1");
    plot->GetYaxis()->SetBinLabel(18,"ME+ 3/2");
    plot->GetYaxis()->SetBinLabel(19,"ME+ 4/1");
    plot->GetYaxis()->SetBinLabel(20,"ME+ 4/2");

    plot->GetXaxis()->SetBinLabel(1,"1");
    plot->GetXaxis()->SetBinLabel(2,"2");
    plot->GetXaxis()->SetBinLabel(3,"3");
    plot->GetXaxis()->SetBinLabel(4,"4");
    plot->GetXaxis()->SetBinLabel(5,"5");
    plot->GetXaxis()->SetBinLabel(6,"6");
    plot->GetXaxis()->SetBinLabel(7,"7");
    plot->GetXaxis()->SetBinLabel(8,"8");
    plot->GetXaxis()->SetBinLabel(9,"9");
    plot->GetXaxis()->SetBinLabel(10,"10");
    plot->GetXaxis()->SetBinLabel(11,"11");
    plot->GetXaxis()->SetBinLabel(12,"12");
    plot->GetXaxis()->SetBinLabel(13,"13");
    plot->GetXaxis()->SetBinLabel(14,"14");
    plot->GetXaxis()->SetBinLabel(15,"15");
    plot->GetXaxis()->SetBinLabel(16,"16");
    plot->GetXaxis()->SetBinLabel(17,"17");
    plot->GetXaxis()->SetBinLabel(18,"18");
    plot->GetXaxis()->SetBinLabel(19,"19");
    plot->GetXaxis()->SetBinLabel(20,"20");
    plot->GetXaxis()->SetBinLabel(21,"21");
    plot->GetXaxis()->SetBinLabel(22,"22");
    plot->GetXaxis()->SetBinLabel(23,"23");
    plot->GetXaxis()->SetBinLabel(24,"24");
    plot->GetXaxis()->SetBinLabel(25,"25");
    plot->GetXaxis()->SetBinLabel(26,"26");
    plot->GetXaxis()->SetBinLabel(27,"27");
    plot->GetXaxis()->SetBinLabel(28,"28");
    plot->GetXaxis()->SetBinLabel(29,"29");
    plot->GetXaxis()->SetBinLabel(30,"30");
    plot->GetXaxis()->SetBinLabel(31,"31");
    plot->GetXaxis()->SetBinLabel(32,"32");
    plot->GetXaxis()->SetBinLabel(33,"33");
    plot->GetXaxis()->SetBinLabel(34,"34");
    plot->GetXaxis()->SetBinLabel(35,"35");
    plot->GetXaxis()->SetBinLabel(36,"36");

    plot->GetYaxis()->SetNdivisions(20,kFALSE);
    plot->GetXaxis()->SetNdivisions(36,kFALSE);

    plot->GetXaxis()->SetTitle("Chamber #");

    c->SetGrid();

    plot->Draw("COLZ");

    c->Update();

    savename = savename + ".gif";
    c->Print(savename.c_str());

    gStyle->Reset();

  }

  TStyle* CSCValPlotFormatter::getStyle(int stat){
    TStyle *theStyle;
    theStyle = new TStyle("myStyle", "myStyle");
    if (stat == 0) gStyle->SetOptStat(false);
    else gStyle->SetOptStat(stat);
    theStyle->SetPadBorderMode(0);
    theStyle->SetCanvasBorderMode(0);
    theStyle->SetPadColor(0);
    theStyle->SetStatColor(0);
    theStyle->SetCanvasColor(0);
    //theStyle->SetMarkerStyle(8);
    //theStyle->SetMarkerSize(0.7);
    //   theStyle->SetTextFont(132);
    //   theStyle->SetTitleFont(132);
    theStyle->SetTitleBorderSize(2);
    theStyle->SetTitleFillColor(0);
    theStyle->SetTitleW(0.7);
    theStyle->SetTitleH(0.07);
    theStyle->SetPalette(1);
    return theStyle;
  }


  void CSCValPlotFormatter::makeGif(TH1* theHisto, string savename, int stat){
    gStyle->Reset();
    if (stat == 0) gStyle->SetOptStat(false);
    else gStyle->SetOptStat(stat);
    gStyle->SetPadBorderMode(0);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadColor(0);
    gStyle->SetStatColor(0);
    gStyle->SetCanvasColor(0);
    //theStyle->SetMarkerStyle(8);
    //theStyle->SetMarkerSize(0.7);
    //   theStyle->SetTextFont(132);
    //   theStyle->SetTitleFont(132);
    gStyle->SetTitleBorderSize(2);
    gStyle->SetTitleFillColor(0);
    gStyle->SetTitleW(0.7);
    gStyle->SetTitleH(0.07);
    gStyle->SetPalette(1);
    gStyle->SetHistFillColor(92);
    TCanvas *c = new TCanvas("c","my canvas",1);
    theHisto->UseCurrentStyle();
    theHisto->GetXaxis()->SetLabelSize(0.04);
    theHisto->GetYaxis()->SetLabelSize(0.04);
    theHisto->GetXaxis()->SetTitleOffset(0.7);
    theHisto->GetXaxis()->SetTitleSize(0.04);
    theHisto->GetXaxis()->SetNdivisions(208,kTRUE);
    theHisto->SetMarkerStyle(6);
    theHisto->Draw();
    c->Update();
    savename = savename + ".gif";
    c->Print(savename.c_str());
    gStyle->Reset();
  }

  void CSCValPlotFormatter::makeEffGif(TH1* theHisto, string savename){
    gStyle->Reset();
    gStyle = getStyle(0);
    TCanvas *c = new TCanvas("c","my canvas",1);
    theHisto->UseCurrentStyle();
    theHisto->GetXaxis()->SetLabelSize(0.04);
    theHisto->GetYaxis()->SetLabelSize(0.04);
    theHisto->GetXaxis()->SetTitleOffset(0.7);
    theHisto->GetXaxis()->SetTitleSize(0.06);
    theHisto->GetXaxis()->SetNdivisions(208,kTRUE);
    theHisto->GetYaxis()->SetRangeUser(0.5,1.1);
    theHisto->SetMarkerStyle(6);
    theHisto->GetXaxis()->SetBinLabel(1,"ME +1/1b");
    theHisto->GetXaxis()->SetBinLabel(2,"ME +1/2");
    theHisto->GetXaxis()->SetBinLabel(3,"ME +1/3");
    theHisto->GetXaxis()->SetBinLabel(4,"ME +1/1a");
    theHisto->GetXaxis()->SetBinLabel(5,"ME +2/1");
    theHisto->GetXaxis()->SetBinLabel(6,"ME +2/2");
    theHisto->GetXaxis()->SetBinLabel(7,"ME +3/1");
    theHisto->GetXaxis()->SetBinLabel(8,"ME +3/2");
    theHisto->GetXaxis()->SetBinLabel(9,"ME +4/1");
    theHisto->GetXaxis()->SetBinLabel(10,"ME +4/2");
    theHisto->GetXaxis()->SetBinLabel(11,"ME -1/1b");
    theHisto->GetXaxis()->SetBinLabel(12,"ME -1/2");
    theHisto->GetXaxis()->SetBinLabel(13,"ME -1/3");
    theHisto->GetXaxis()->SetBinLabel(14,"ME -1/1a");
    theHisto->GetXaxis()->SetBinLabel(15,"ME -2/1");
    theHisto->GetXaxis()->SetBinLabel(16,"ME -2/2");
    theHisto->GetXaxis()->SetBinLabel(17,"ME -3/1");
    theHisto->GetXaxis()->SetBinLabel(18,"ME -3/2");
    theHisto->GetXaxis()->SetBinLabel(19,"ME -4/1");
    theHisto->GetXaxis()->SetBinLabel(20,"ME -4/2");
    theHisto->Draw();
    c->Update();
    savename = savename + ".gif";
    c->Print(savename.c_str());
    gStyle->Reset();
  }


  void CSCValPlotFormatter::drawChamberLines(int station, int lc1){

  gStyle->SetLineWidth(2);
  float pi = 3.14159;
  TVector3 x(0,0,1);
  int linecolor = 1;
  //for alternating colors, set 2 diff colors here
  int lc2 = lc1;

  if (station == 1){
    TVector3 p1(101,9.361,0);
    TVector3 p2(101,-9.361,0);
    TVector3 p3(260,-22.353,0);
    TVector3 p4(260,22.353,0);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    for (int i = 0; i < 36; i++){

      if (linecolor == lc1) linecolor = lc2;
      else linecolor = lc1;

      line1 = new TLine(p1(0),p1(1),p2(0),p2(1));
      line2 = new TLine(p2(0),p2(1),p3(0),p3(1));
      line3 = new TLine(p3(0),p3(1),p4(0),p4(1));
      line4 = new TLine(p4(0),p4(1),p1(0),p1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      p1.Rotate(pi/18,x);
      p2.Rotate(pi/18,x);
      p3.Rotate(pi/18,x);
      p4.Rotate(pi/18,x);

    }

    TVector3 q1(281.49,25.5,0);
    TVector3 q2(281.49,-25.5,0);
    TVector3 q3(455.99,-41.87,0);
    TVector3 q4(455.99,41.87,0);
    for (int i = 0; i < 36; i++){

      if (linecolor == lc2) linecolor = lc1;
      else linecolor = lc2;

      line1 = new TLine(q1(0),q1(1),q2(0),q2(1));
      line2 = new TLine(q2(0),q2(1),q3(0),q3(1));
      line3 = new TLine(q3(0),q3(1),q4(0),q4(1));
      line4 = new TLine(q4(0),q4(1),q1(0),q1(1));

      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      q1.Rotate(pi/18,x);
      q2.Rotate(pi/18,x);
      q3.Rotate(pi/18,x);
      q4.Rotate(pi/18,x);

    }

    TVector3 r1(511.99,31.7,0);
    TVector3 r2(511.99,-31.7,0);
    TVector3 r3(676.15,-46.05,0);
    TVector3 r4(676.15,46.05,0);

    for (int i = 0; i < 36; i++){

      if (linecolor == lc1) linecolor = lc2;
      else linecolor = lc1;

      line1 = new TLine(r1(0),r1(1),r2(0),r2(1));
      line2 = new TLine(r2(0),r2(1),r3(0),r3(1));
      line3 = new TLine(r3(0),r3(1),r4(0),r4(1));
      line4 = new TLine(r4(0),r4(1),r1(0),r1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();


      r1.Rotate(pi/18,x);
      r2.Rotate(pi/18,x);
      r3.Rotate(pi/18,x);
      r4.Rotate(pi/18,x);

    }

  }


  if (station == 2){

    TVector3 p1(146.9,27.0,0);
    TVector3 p2(146.9,-27.0,0);
    TVector3 p3(336.56,-62.855,0);
    TVector3 p4(336.56,62.855,0);

    p1.Rotate(pi/36,x);
    p2.Rotate(pi/36,x);
    p3.Rotate(pi/36,x);
    p4.Rotate(pi/36,x);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    for (int i = 0; i < 36; i++){

      if (linecolor == lc1) linecolor = lc2;
      else linecolor = lc1;

      line1 = new TLine(p1(0),p1(1),p2(0),p2(1));
      line2 = new TLine(p2(0),p2(1),p3(0),p3(1));
      line3 = new TLine(p3(0),p3(1),p4(0),p4(1));
      line4 = new TLine(p4(0),p4(1),p1(0),p1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      p1.Rotate(pi/9,x);
      p2.Rotate(pi/9,x);
      p3.Rotate(pi/9,x);
      p4.Rotate(pi/9,x);

    }

    TVector3 q1(364.02,33.23,0);
    TVector3 q2(364.02,-33.23,0);
    TVector3 q3(687.08,-63.575,0);
    TVector3 q4(687.08,63.575,0);

    for (int i = 0; i < 36; i++){

      if (linecolor == lc2) linecolor = lc1;
      else linecolor = lc2;

      line1 = new TLine(q1(0),q1(1),q2(0),q2(1));
      line2 = new TLine(q2(0),q2(1),q3(0),q3(1));
      line3 = new TLine(q3(0),q3(1),q4(0),q4(1));
      line4 = new TLine(q4(0),q4(1),q1(0),q1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      q1.Rotate(pi/18,x);
      q2.Rotate(pi/18,x);
      q3.Rotate(pi/18,x);
      q4.Rotate(pi/18,x);

    }

  }

  if (station == 3){

    TVector3 p1(166.89,30.7,0);
    TVector3 p2(166.89,-30.7,0);
    TVector3 p3(336.59,-62.855,0);
    TVector3 p4(336.59,62.855,0);

    p1.Rotate(pi/36,x);
    p2.Rotate(pi/36,x);
    p3.Rotate(pi/36,x);
    p4.Rotate(pi/36,x);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    for (int i = 0; i < 36; i++){

      if (linecolor == lc1) linecolor = lc2;
      else linecolor = lc1;

      line1 = new TLine(p1(0),p1(1),p2(0),p2(1));
      line2 = new TLine(p2(0),p2(1),p3(0),p3(1));
      line3 = new TLine(p3(0),p3(1),p4(0),p4(1));
      line4 = new TLine(p4(0),p4(1),p1(0),p1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      p1.Rotate(pi/9,x);
      p2.Rotate(pi/9,x);
      p3.Rotate(pi/9,x);
      p4.Rotate(pi/9,x);

    }

    TVector3 q1(364.02,33.23,0);
    TVector3 q2(364.02,-33.23,0);
    TVector3 q3(687.08,-63.575,0);
    TVector3 q4(687.08,63.575,0);

    for (int i = 0; i < 36; i++){

      if (linecolor == lc2) linecolor = lc1;
      else linecolor = lc2;

      line1 = new TLine(q1(0),q1(1),q2(0),q2(1));
      line2 = new TLine(q2(0),q2(1),q3(0),q3(1));
      line3 = new TLine(q3(0),q3(1),q4(0),q4(1));
      line4 = new TLine(q4(0),q4(1),q1(0),q1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      q1.Rotate(pi/18,x);
      q2.Rotate(pi/18,x);
      q3.Rotate(pi/18,x);
      q4.Rotate(pi/18,x);
    }

  }

  if (station == 4){

    TVector3 p1(186.99,34.505,0);
    TVector3 p2(186.99,-34.505,0);
    TVector3 p3(336.41,-62.825,0);
    TVector3 p4(336.41,62.825,0);

    p1.Rotate(pi/36,x);
    p2.Rotate(pi/36,x);
    p3.Rotate(pi/36,x);
    p4.Rotate(pi/36,x);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    for (int i = 0; i < 36; i++){

      if (linecolor == lc1) linecolor = lc2;
      else linecolor = lc1;

      line1 = new TLine(p1(0),p1(1),p2(0),p2(1));
      line2 = new TLine(p2(0),p2(1),p3(0),p3(1));
      line3 = new TLine(p3(0),p3(1),p4(0),p4(1));
      line4 = new TLine(p4(0),p4(1),p1(0),p1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      p1.Rotate(pi/9,x);
      p2.Rotate(pi/9,x);
      p3.Rotate(pi/9,x);
      p4.Rotate(pi/9,x);

    }
  }

  }


  int CSCValPlotFormatter::typeIndex(CSCDetId id){

    // linearlized index bases on endcap, station, and ring based on CSCDetId
    int i = 2 * id.station() + id.ring(); // i=2S+R ok for S=2, 3, 4
    if ( id.station() == 1 ) {
      --i;                       // ring 1R -> i=1+R (2S+R-1=1+R for S=1)
      if ( i > 4 ) i = 1;        // But ring 1A (R=4) -> i=1
    }
    if (id.endcap() == 1) i = i+10;
    if (id.endcap() == 2) i = 11-i;

    return i;

  }


  // this is TEMPORARY to take care of Nikolai's plots
  void CSCValPlotFormatter::nikolaiMacro(map<string,pair<TH1*,string> > theMap, int flag){
    gStyle = getStyle();
    gStyle->SetPalette(1,0); // 

    ostringstream ss;
    ostringstream ss1;
    std::string folder;
    std::string input_histName;
    std::string input_title_X;
    std::string input_title_Y;
    std::string slice_title_X;
    Int_t ny = 0;
    Float_t ylow = 0;
    Float_t yhigh = 0;
    std::string result_histName;
    std::string result_histTitle;
    std::string result_title_Y;
    std::string result_histNameEntries;
    std::string result_histTitleEntries;


    if(flag==1) {  // gas gain results
      folder="GasGain/";
      input_histName = "gas_gain_rechit_adc_3_3_sum_location_ME_";
      input_title_X="Location=(layer-1)*nsegm+segm";
      input_title_Y="3X3 ADC Sum";
      slice_title_X="3X3 ADC Sum Location";
      ny=30;
      ylow=1.0;
      yhigh=31.0;
      result_histName = "mean_gas_gain_vs_location_csc_ME_";
      result_histTitle="Mean 3X3 ADC Sum";
      result_title_Y="Location=(layer-1)*nsegm+segm";
      result_histNameEntries = "entries_gas_gain_vs_location_csc_ME_";
      result_histTitleEntries="Entries 3X3 ADC Sum";
    }

    if(flag==2) {  // AFEB timing results
      folder="AFEBTiming/";
      input_histName = "afeb_time_bin_vs_afeb_occupancy_ME_";
      input_title_X="AFEB";
      input_title_Y="Time Bin";
      slice_title_X="AFEB";
      ny=42;
      ylow=1.0;
      yhigh=42.0;
      result_histName = "mean_afeb_time_bin_vs_afeb_csc_ME_";
      result_histTitle="AFEB Mean Time Bin";
      result_title_Y="AFEB";
      result_histNameEntries = "entries_afeb_time_bin_vs_afeb_csc_ME_";
      result_histTitleEntries="Entries AFEB Time Bin";
    }

    if(flag==3) {  // Comparator timing results
      folder="CompTiming/";
      input_histName = "comp_time_bin_vs_cfeb_occupancy_ME_";
      input_title_X="CFEB";
      input_title_Y="Time Bin";
      slice_title_X="CFEB";
      ny=5;
      ylow=1.0;
      yhigh=6.0;
      result_histName = "mean_comp_time_bin_vs_cfeb_csc_ME_";
      result_histTitle="Comparator Mean Time Bin";
      result_title_Y="CFEB";
      result_histNameEntries = "entries_comp_time_bin_vs_cfeb_csc_ME_";
      result_histTitleEntries="Entries Comparator Time Bin";
    }

    if(flag==4) {  // Strip ADC timing results
      folder="ADCTiming/";
      input_histName = "adc_3_3_weight_time_bin_vs_cfeb_occupancy_ME_";
      input_title_X="CFEB";
      input_title_Y="Time Bin";
      slice_title_X="CFEB";
      ny=5;
      ylow=1.0;
      yhigh=6.0;
      result_histName = "mean_adc_time_bin_vs_cfeb_csc_ME_";
      result_histTitle="ADC 3X3 Mean Time Bin";
      result_title_Y="CFEB";
      result_histNameEntries = "entries_adc_time_bin_vs_cfeb_csc_ME_";
      result_histTitleEntries="Entries ADC 3X3 Time Bin";
    }


    std::vector<std::string> xTitle;
    xTitle.push_back("ME+1/1 CSC Chamber #"); xTitle.push_back("ME+1/2 CSC Chamber #");
    xTitle.push_back("ME+1/3 CSC Chamber #"); 
    xTitle.push_back("ME+2/1 CSC Chamber #"); xTitle.push_back("ME+2/2 CSC Chamber #");
    xTitle.push_back("ME+3/1 CSC Chamber #"); xTitle.push_back("ME+3/2 CSC Chamber #");
    xTitle.push_back("ME+4/1 CSC Chamber #"); xTitle.push_back("ME+4/2 CSC Chamber #");
    xTitle.push_back("ME-1/1 CSC Chamber #"); xTitle.push_back("ME-1/2 CSC Chamber #");
    xTitle.push_back("ME-1/3 CSC Chamber #");
    xTitle.push_back("ME-2/1 CSC Chamber #"); xTitle.push_back("ME-2/2 CSC Chamber #");
    xTitle.push_back("ME-3/1 CSC Chamber #"); xTitle.push_back("ME-3/2 CSC Chamber #");
    xTitle.push_back("ME-4/1 CSC Chamber #"); xTitle.push_back("ME-4/2 CSC Chamber #");

    Int_t esr[18]={111,112,113,121,122,131,132,141,142,
                   211,212,213,221,222,231,232,241,242};
    Int_t entries[18]={0,0,0,0,0,0,0,0,0,
                       0,0,0,0,0,0,0,0,0};
    TCanvas *c1=new TCanvas("c1","canvas");
    c1->cd();

    //if(flag==2) { // adding special case for AFEB timing
    ss.str("");
    ss<<"mean_afeb_time_bin_vs_csc_ME";
    ss1.str("");
    ss1<<"Mean AFEB time bin vs CSC and ME";
    gStyle->SetOptStat(0);
    TH2F *hb=new TH2F(ss.str().c_str(),ss1.str().c_str(),36,1.0,37.0,18,1.0,19.0);
    hb->SetStats(kFALSE);
    hb->GetXaxis()->SetTitle("CSC Chamber #");
    hb->GetZaxis()->SetLabelSize(0.03);
    hb->SetOption("COLZ");
    hb->GetYaxis()->SetBinLabel(1, "ME- 4/2");
    hb->GetYaxis()->SetBinLabel(2, "ME- 4/1");
    hb->GetYaxis()->SetBinLabel(3, "ME- 3/2");
    hb->GetYaxis()->SetBinLabel(4, "ME- 3/1");
    hb->GetYaxis()->SetBinLabel(5, "ME- 2/2");
    hb->GetYaxis()->SetBinLabel(6, "ME- 2/1");
    hb->GetYaxis()->SetBinLabel(7, "ME- 1/3");
    hb->GetYaxis()->SetBinLabel(8, "ME- 1/2");
    hb->GetYaxis()->SetBinLabel(9, "ME- 1/1");
    hb->GetYaxis()->SetBinLabel(10,"ME+ 1/1");
    hb->GetYaxis()->SetBinLabel(11,"ME+ 1/2");
    hb->GetYaxis()->SetBinLabel(12,"ME+ 1/3");
    hb->GetYaxis()->SetBinLabel(13,"ME+ 2/1");
    hb->GetYaxis()->SetBinLabel(14,"ME+ 2/2");
    hb->GetYaxis()->SetBinLabel(15,"ME+ 3/1");
    hb->GetYaxis()->SetBinLabel(16,"ME+ 3/2");
    hb->GetYaxis()->SetBinLabel(17,"ME+ 4/1");
    hb->GetYaxis()->SetBinLabel(18,"ME+ 4/2");
    //}

    for(Int_t jesr=0;jesr<18;jesr++) { 
      ss.str("");
      ss<<result_histName.c_str()<<esr[jesr];
      ss1.str("");
      ss1<<result_histTitle;
      TH2F *h=new TH2F(ss.str().c_str(),ss1.str().c_str(),40,0.0,40.0,ny,ylow,yhigh);
      h->SetStats(kFALSE);
      h->GetXaxis()->SetTitle(xTitle[jesr].c_str());
      h->GetYaxis()->SetTitle(result_title_Y.c_str());
      h->GetZaxis()->SetLabelSize(0.03);
      h->SetOption("COLZ");
      ss.str("");
      ss<<result_histNameEntries.c_str()<<esr[jesr];
      ss1.str("");
      ss1<<result_histTitleEntries;
      TH2F *hentr=new TH2F(ss.str().c_str(),ss1.str().c_str(),40,0.0,40.0,ny,ylow,yhigh);
      hentr->SetStats(kFALSE);
      hentr->GetXaxis()->SetTitle(xTitle[jesr].c_str());
      hentr->GetYaxis()->SetTitle(result_title_Y.c_str());
      hentr->GetZaxis()->SetLabelSize(0.03);
      hentr->SetOption("COLZ");
      TH2F *ha;

      if(flag==2) { // adding special cases for AFEB timing
        ss.str("");
        ss<<"normal_afeb_time_bin_vs_csc_ME_"<<esr[jesr];
        ss1.str("");
        ss1<<"Normalized AFEB time bin, %";
        ha=new TH2F(ss.str().c_str(),ss1.str().c_str(),40,0.0,40.0,16,0.0,16.0);
        ha->SetStats(kFALSE);
        ha->GetXaxis()->SetTitle(xTitle[jesr].c_str());
        ha->GetYaxis()->SetTitle("Time Bin");
        ha->GetZaxis()->SetLabelSize(0.03);
        ha->SetOption("COLZ");
      }

      for(Int_t csc=1;csc<37;csc++) {
        Int_t idchamber=esr[jesr]*100+csc;
        ss.str("");
        //ss<<folder.c_str()<<input_histName.c_str()<<idchamber;
        ss<<input_histName.c_str()<<idchamber;
        TH2F *h2 = (TH2F*)theMap[ss.str()].first;
        if(h2 != NULL) {

          // saving original, adding X,Y titles, color and "BOX" option
          h2->GetXaxis()->SetTitle(input_title_X.c_str());
          h2->GetYaxis()->SetTitle(input_title_Y.c_str());
          h2->GetYaxis()->SetTitleOffset(1.2);
          h2->SetFillColor(4);
          h2->SetOption("BOX");
          gStyle->SetOptStat(1001111);

          // saving Y projection of the whole 2D hist for given chamber
          ss.str("");
          ss<<input_histName.c_str()<<idchamber<<"_Y_all";
          TH1D *h1d = h2->ProjectionY(ss.str().c_str(),1,h2->GetNbinsX(),"");
          h1d->GetYaxis()->SetTitle("Entries");
          h1d->GetYaxis()->SetTitleOffset(1.2);
          gStyle->SetOptStat(1001111);

          if(flag==2 && h1d->GetEntries() > 0) {// adding spec. case for afeb timing
            Float_t entr=h1d->GetEntries();
            for(Int_t m=1; m<h1d->GetNbinsX();m++) {
              Float_t w=h1d->GetBinContent(m);
              w=100.0*w/entr;
              //ha->SetBinContent(csc+1,m,w);
              ha->SetBinContent(csc,m,w);
            }
            Float_t mean=h1d->GetMean();
            Int_t me;
            if(jesr<9) me=10+jesr;
            if(jesr>8) me=18-jesr;
            hb->SetBinContent(csc,me,mean);
          }
          delete h1d;   

          // saving slices, finding MEAN in each slice, fill 2D hist
          for(Int_t j=1;j<=h2->GetNbinsX();j++) {
            Int_t n=j;
            ss.str("");
            ss<<input_histName.c_str()<<idchamber<<"_Y_"<<n;
            TH1D *h1d = h2->ProjectionY(ss.str().c_str(),j,j,"");
            if(h1d->GetEntries() > 0) {
              Float_t mean=h1d->GetMean();
              Float_t entr=h1d->GetEntries();
              entries[jesr]=entries[jesr]+1;
              //h->SetBinContent(csc+1,j,mean);
              //hentr->SetBinContent(csc+1,j,entr);
              h->SetBinContent(csc,j,mean);
              hentr->SetBinContent(csc,j,entr);
              ss.str("");
              ss<<slice_title_X<<" "<<n;
              h1d->GetXaxis()->SetTitle(ss.str().c_str());
              h1d->GetYaxis()->SetTitle("Entries");
              h1d->GetYaxis()->SetTitleOffset(1.2);
              gStyle->SetOptStat(1001111);
            }
            delete h1d;
          }
        }
      }
      if(entries[jesr]>0) {
        h->SetStats(kFALSE);
        hentr->SetStats(kFALSE);
        c1->Update();

        // printing
     
        h->Draw();
        ss.str("");
        ss<<result_histName.c_str()<<esr[jesr]<<".gif";
        c1->Print(ss.str().c_str(),"gif");
     
        hentr->Draw();
        ss.str("");
        ss<<result_histNameEntries.c_str()<<esr[jesr]<<".gif";
        c1->Print(ss.str().c_str(),"gif");
      }
      delete h;
      delete hentr;
      if (flag == 2) delete ha;
    }
    if(flag==2) {
      hb->Draw();      
      ss.str("");
      ss<<"mean_afeb_time_bin_vs_csc_ME"<<".gif";      
      c1->Print(ss.str().c_str(),"gif");

      c1->Update();
      delete hb;    
    }
  }



