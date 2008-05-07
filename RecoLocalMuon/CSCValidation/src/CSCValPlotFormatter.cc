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

    gStyle = getStyle();

    string theName;
    string theTitle;
    string theFolder;
    TH1* theHisto;

    map<string,pair<TH1*,string> >::const_iterator mapit;
    for (mapit = theMap.begin(); mapit != theMap.end(); mapit++){
      theFolder = (*mapit).second.second;
      theName = (*mapit).first;
      theHisto = (*mapit).second.first;
      if (theName == "hOWires" || theName == "hOStrips" || theName == "hORecHits" || theName == "hOSegments"){
         make2DTemperaturePlot(theHisto, theName);
      }
      else {
        TCanvas *c = new TCanvas("c","my canvas",1);
        theHisto->UseCurrentStyle();
        theHisto->GetXaxis()->SetLabelSize(0.04);
        theHisto->GetYaxis()->SetLabelSize(0.04);
        theHisto->GetXaxis()->SetTitleOffset(0.7);
        theHisto->GetXaxis()->SetTitleSize(0.06);
        theHisto->GetXaxis()->SetNdivisions(208,kTRUE);
        gStyle->SetHistFillColor(92);
        theHisto->Draw();
        c->Update();
        string savename = theFolder + "_" + theName + ".gif";
        c->Print(savename.c_str());
      }
    }


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

    gStyle = getStyle();
    TGraph *result;
    if (type == "rechits"){
      TCanvas *c = new TCanvas("c","my canvas",1);
      c->SetCanvasSize(700,700);
      //station 1 +side
      t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 1 && rHpos.endcap == 1","goff");
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
      //station 2 +side
      t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 2 && rHpos.endcap == 1","goff");
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
      //station 3 +side
      t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 3 && rHpos.endcap == 1","goff");
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
      //station 4 +side
      t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 4 && rHpos.endcap == 1","goff");
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

      //station 1 -side
      t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 1 && rHpos.endcap == 2","goff");
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
      //station 2 +side
      t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 2 && rHpos.endcap == 2","goff");
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
      //station 3 +side
      t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 3 && rHpos.endcap == 2","goff");
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
      //station 4 +side
      t1->Draw("rHpos.globaly:rHpos.globalx","rHpos.station == 4 && rHpos.endcap == 2","goff");
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


  void CSCValPlotFormatter::make2DTemperaturePlot(TH1 *plot, string savename){

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

  }


  TStyle* CSCValPlotFormatter::getStyle(TString name){
    TStyle *theStyle;
    if ( name == "myStyle" ) {
      theStyle = new TStyle("myStyle", "myStyle");
      theStyle->SetOptStat(10);
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
    } else {
      // Avoid modifying the default style!
      theStyle = gStyle;
    }
    return theStyle;
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

