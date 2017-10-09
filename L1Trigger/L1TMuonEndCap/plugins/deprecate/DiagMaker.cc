/////////////////////////////////////////////////////
// A comparison analyzer which analyzes the        //
// output of two emulators, each of which perform  //
// trackfinding for the upgrades endcap muon       //
// trackfinder. Graphical and text comparions	   //
// are done.					   //
//						   //
// Author: G. Brown (UF)			   //
//						   //
/////////////////////////////////////////////////////
#include <memory>
#include <map>

#include <stdio.h>
#include <stdlib.h>
#include "Riostream.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "L1Trigger/L1TMuonEndCap/interface/SubsystemCollectorFactory.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include <TMath.h>
#include <TCanvas.h>
#include <TLorentzVector.h>

#include "TTree.h"
#include "TNtuple.h"
#include "TImage.h"
#include "TSystem.h"
    
#include <TStyle.h>
#include <TLegend.h>
#include <TF1.h>
#include <TH2.h>
#include <TH1F.h>
#include <TFile.h>
#include "L1Trigger/L1TMuonEndCap/interface/GeometryTranslator.h"
    
#include "L1Trigger/L1TMuonEndCap/interface/MuonTriggerPrimitive.h"
#include "L1Trigger/L1TMuonEndCap/interface/MuonTriggerPrimitiveFwd.h"

#include "L1Trigger/L1TMuonEndCap/interface/MuonInternalTrack.h"
#include "L1Trigger/L1TMuonEndCap/interface/MuonInternalTrackFwd.h"
#include "L1Trigger/L1TMuonEndCap/interface/PhiMemoryImage.h"
  
#include "TFile.h"
#include "TH1.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
    
#include <vector>
#include <iostream>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TGaxis.h"
#include "TCanvas.h"
#include "TGraphErrors.h"
#include "string.h"

using namespace L1TMuon;
using namespace edm;
using namespace reco;
using namespace std;

typedef edm::ParameterSet PSet;


class WordHist
{
  public:
    WordHist(string theVarName,int theNBits, double theXCornerVal, double theYCornerVal);
    void setPad();
    bool fillWord(int, int);
    void formatWordHist();
    TPad *pad;
  private:
    string varName;
    const int nBits;
    double xCornerVal; // x value of bottom-left corner of pad
    double yCornerVal; // y value of the bottom-left corner of pad
    int lineColor;
    int fillColor;
    double tPadLength;
    double tPadHeight;
    double leftMargin;
    double rightMargin;
    double yaxisTitleOffset;
    double yaxisLabelOffset;
    //TPad *pad;
    TH2F *hist;

    // Private member functions
    void setFormatValues();
};


class DiagMaker : public edm::EDAnalyzer
{
  public:
    DiagMaker(const PSet&);
    void analyze(const edm::Event&, const edm::EventSetup&);
    void beginJob();
    void endJob();
    int delayCount;
    int evCntr;
  private:
    TCanvas *c1, *c2;

    WordHist *wordTrkTheta;
    WordHist *wordTrkPhi;
    WordHist *wordTrknmb;
    WordHist *wordValid;
    WordHist *wordQuality;
    WordHist *wordKeywire;
    WordHist *wordStrip;
    WordHist *wordPattern;
    WordHist *wordBend;
    WordHist *wordBx;
    WordHist *wordMpclink;
    WordHist *wordBx0;
    WordHist *wordSyncErr;
    WordHist *wordCscID;
    WordHist *wordRank;
    WordHist *wordPHD1, *wordPHD2;
    WordHist *wordTHD1, *wordTHD2;

};


void formatHist(TH2F* hist, string histTitle, int nXBins, int xLow, int xHigh, string xTitle, 
                int nYBins, int yLow, int yHigh, string yTitle);
void pad2png(TH2F *h, string imgName);


DiagMaker::DiagMaker(const PSet& p)
{
evCntr = 0;   

}


void DiagMaker::analyze(const edm::Event& ev, const edm::EventSetup& es)
{


	cout<<"Begin DiagMaker Analyzer:::::::::::::::::::::::::::::::::::::\n:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n\n";

  cout << "Reached analyze part of DiagMaker.\n";

  edm::Handle<InternalTrackCollection> emuITC;
  edm::Handle<InternalTrackCollection> dataITC;

  ev.getByLabel( "sptf" , "EmuITC"  , emuITC  );
  ev.getByLabel( "L1TMuonTrkFinder" , "DataITC"  , dataITC );

  int trknmbData, validData, qualityData, keywireData, stripData, patternData; 
  int bendData, bxData, mpclinkData, bx0Data, syncErrData, cscIDData;
  int trknmbEmu, validEmu, qualityEmu, keywireEmu, stripEmu, patternEmu;
  int bendEmu, bxEmu, mpclinkEmu, bx0Emu, syncErrEmu, cscIDEmu;

  auto itEmu     = emuITC->cbegin();
  auto itendEmu  = emuITC->cend();

  auto itData    = dataITC->cbegin();
  auto itendData = dataITC->cend(); 
  int trkCntr = 0;
  // Loop over all tracks in the collection
  for( ; (itEmu != itendEmu) && (itData != itendData) ; ++itEmu, ++itData)
  {
      trkCntr++;
      cout << "TRACK(Emu) "  << trkCntr << " Phi:  " << itEmu->phi  << ",   Theta:  " << itEmu ->theta << endl;
      cout << "TRACK(Data) " << trkCntr << " Phi:  " << itData->phi << ",   Theta:  " << itData->theta << endl;
      cout << "Track(Emu)  " << trkCntr << " Rank: " << itEmu->rank << endl;
      cout << "Track(Data) " << trkCntr << " Rank: " << itData->rank<< endl;
      cout << "Emu:\n dp1- "<<itEmu->deltas[0][0]<<" dp2- "<<itEmu->deltas[0][1]<<" dt1- "<<itEmu->deltas[1][0]<<" dt2- "<<itEmu->deltas[1][1]<<endl;
      cout << "Data:\n dp1- "<<itData->deltas[0][0]<<" dp2- "<<itData->deltas[0][1]<<" dt1- "<<itData->deltas[1][0]<<" dt2- "<<itData->deltas[1][1]<<endl;

      wordTrkPhi   -> fillWord( itData->phi          , itEmu->phi         );
      wordTrkTheta -> fillWord( itData->theta        , itEmu->theta       );
      wordRank     -> fillWord( itData->rank         , itEmu->rank        );
      wordPHD1     -> fillWord( fabs(itData->deltas[0][0]) , fabs(itEmu->deltas[0][0]));
      wordPHD2     -> fillWord( fabs(itData->deltas[0][1]) , fabs(itEmu->deltas[0][1]));
      wordTHD1     -> fillWord( fabs(itData->deltas[1][0]) , fabs(itEmu->deltas[1][0]));
      wordTHD2     -> fillWord( fabs(itData->deltas[1][1]) , fabs(itEmu->deltas[1][1]));
	
      if(itData->phi != itEmu->phi){std::cout<<"Track Phi Mismatch\n\n";}
      if(itData->theta != itEmu->theta){std::cout<<"Track Theta Mismatch\n\n";}
      if(itData->rank != itEmu->rank){std::cout<<"Track Rank Mismatch\n\n";}
      if(fabs(itData->deltas[0][0]) != fabs(itEmu->deltas[0][0])){std::cout<<"00 D Mismatch\n";}
      if(fabs(itData->deltas[0][1]) != fabs(itEmu->deltas[0][1])){std::cout<<"01 D Mismatch\n";}
      if(fabs(itData->deltas[1][0]) != fabs(itEmu->deltas[1][0])){std::cout<<"10 D Mismatch\n";}
      if(fabs(itData->deltas[1][1]) != fabs(itEmu->deltas[1][1])){std::cout<<"11 D Mismatch\n";}
      
      
      


      // Creating station maps for this track
      TriggerPrimitiveStationMap tpsmData = itData->getStubs();
      TriggerPrimitiveStationMap tpsmEmu  = itEmu->getStubs();

      // Getting the unique station ID number for ME1
      const unsigned id = 4*L1TMuon::InternalTrack::kCSC;

      // Looping over all four stations
      for(unsigned meNum=id; meNum<(id+4); meNum++)
      { 
          // Getting the trig prim lists for this station
          //TriggerPrimitiveList tplData = tpsmData[meNum];
          //TriggerPrimitiveList tplEmu  = tpsmEmu[meNum];
		  std::vector<TriggerPrimitive> tplData = tpsmData[meNum];
          std::vector<TriggerPrimitive> tplEmu  = tpsmEmu[meNum];

          cout << "ME " << meNum-id+1 << " -  # Trig Prims = " << tplData.size() << endl;

          // Looping over all the trigger primitives in the lists
          for(unsigned tpNum = 0; (tpNum < tplData.size()) && (tpNum < tplEmu.size()) ; tpNum++)
          {
              cout << " ----- tp #" << tpNum << endl; 
              // Creating references to the trig prim info
              TriggerPrimitive tprData = tplData.at(tpNum);
              TriggerPrimitive tprEmu  = tplEmu.at(tpNum);

              trknmbData     = (tprData).getCSCData().trknmb;
              validData      = (tprData).getCSCData().valid;
              qualityData    = (tprData).getCSCData().quality;
              keywireData    = (tprData).getCSCData().keywire;
              stripData      = (tprData).getCSCData().strip;
              patternData    = (tprData).getCSCData().pattern;
              bendData       = (tprData).getCSCData().bend;
              bxData         = (tprData).getCSCData().bx;
              mpclinkData    = (tprData).getCSCData().mpclink;
              bx0Data        = (tprData).getCSCData().bx0;
              syncErrData    = (tprData).getCSCData().syncErr;
              cscIDData      = (tprData).getCSCData().cscID;
             
              trknmbEmu     = (tprEmu).getCSCData().trknmb;
              validEmu      = (tprEmu).getCSCData().valid;
              qualityEmu    = (tprEmu).getCSCData().quality;
              keywireEmu    = (tprEmu).getCSCData().keywire;
              stripEmu      = (tprEmu).getCSCData().strip;
              patternEmu    = (tprEmu).getCSCData().pattern;
              bendEmu       = (tprEmu).getCSCData().bend;
              bxEmu         = (tprEmu).getCSCData().bx;
              mpclinkEmu    = (tprEmu).getCSCData().mpclink;
              bx0Emu        = (tprEmu).getCSCData().bx0;
              syncErrEmu    = (tprEmu).getCSCData().syncErr;
              cscIDEmu      = (tprEmu).getCSCData().cscID;

              cout << "trknmb:   " << "data=" << trknmbData  << ",  emu=" << trknmbEmu  << endl;
              cout << "valid:    " << "data=" << validData   << ",  emu=" << validEmu   << endl;
              cout << "quality:  " << "data=" << qualityData << ",  emu=" << qualityEmu << endl;
              cout << "keywire:  " << "data=" << keywireData << ",  emu=" << keywireEmu << endl;
              cout << "strip:    " << "data=" << stripData   << ",  emu=" << stripEmu   << endl;
              cout << "pattern:  " << "data=" << patternData << ",  emu=" << patternEmu << endl;
              cout << "bend:     " << "data=" << bendData    << ",  emu=" << bendEmu    << endl;
              cout << "bx:       " << "data=" << bxData      << ",  emu=" << bxEmu      << endl;
              cout << "mpclink:  " << "data=" << mpclinkData << ",  emu=" << mpclinkEmu << endl;
              cout << "bx0:      " << "data=" << bx0Data     << ",  emu=" << bx0Emu     << endl;
              cout << "syncErr:  " << "data=" << syncErrData << ",  emu=" << syncErrEmu << endl;
              cout << "cscID:    " << "data=" << cscIDData   << ",  emu=" << cscIDEmu   << endl;
              cout << "--------------------------------\n";

              // Filling the word histograms
              wordTrknmb   -> fillWord( trknmbData  , trknmbEmu  );
              wordValid    -> fillWord( validData   , validEmu   );
              wordQuality  -> fillWord( qualityData , qualityEmu );
              wordKeywire  -> fillWord( keywireData , keywireEmu );
              wordStrip    -> fillWord( stripData   , stripEmu   );
              wordPattern  -> fillWord( patternData , patternEmu );
              wordBend     -> fillWord( bendData    , bendEmu    );
              wordBx       -> fillWord( bxData      , bxEmu      );
              wordMpclink  -> fillWord( mpclinkData , mpclinkEmu );
              wordBx0      -> fillWord( bx0Data     , bx0Emu     );
              wordSyncErr  -> fillWord( syncErrData , syncErrEmu );
              wordCscID    -> fillWord( cscIDData   , cscIDEmu   );
	      
	      if(keywireData != keywireEmu){std::cout<<"Wire Mismatch\n\n";}
	      if(stripData != stripEmu){std::cout<<"Strip Mismatch\n\n";}
	      

          }
      }
  }
  
  cout<<"End DiagMaker Analyzer:::::::::::::::::::::::::::::::::::::\n:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n\n";
  
}


void DiagMaker::beginJob()
{


	

    // Constructing all the word objects  
    c1 = new TCanvas();
    c1 -> cd();
    
    wordTrkPhi      = new WordHist( "Phi"     , 12 , 0   , 0   );
    c1 -> cd();
    wordTrkTheta    = new WordHist( "Theta"   ,  6 , 0   , 0.2 );
    c1 -> cd();
    wordTrknmb      = new WordHist( "Phhit"   ,  6 , 0   , 0.4 );
    c1 -> cd();
    wordValid       = new WordHist( "Valid"   , 12 , 0   , 0.6 );
    c1 -> cd();
    wordQuality     = new WordHist( "Quality" ,  7 , 0   , 0.8 );
    c1 -> cd();
    wordKeywire     = new WordHist( "Keywire" , 12 , 0.5 , 0   );
    c1 -> cd();
    wordStrip       = new WordHist( "Strip"   ,  7 , 0.5 , 0.2 );
    c1 -> cd();
    wordPattern     = new WordHist( "Pattern" , 12 , 0.5 , 0.4 );
    c1 -> cd();
    wordBend        = new WordHist( "Bend"    , 12 , 0.5 , 0.6 );
    c1 -> cd();
    wordBx          = new WordHist( "Bx"      , 12 , 0.5 , 0.8 );

    c2 = new TCanvas();
    c2 -> cd();
    wordMpclink     = new WordHist( "Mpclink" , 12 , 0   , 0   );
    c2 -> cd();
    wordBx0         = new WordHist( "Bx"      , 12 , 0   , 0.2 );
    c2 -> cd();
    wordSyncErr     = new WordHist( "SyncErr" , 12 , 0   , 0.4 );
    c2 -> cd();
    wordCscID       = new WordHist( "CscID"   , 12 , 0   , 0.6 );
    c2 -> cd();
    wordRank 	    = new WordHist( "Rank"    , 12 , 0.5 , 0   );
    c2 -> cd();
    wordPHD1        = new WordHist( "PHD1"    , 12 , 0.5 , 0.2 );
    c2 ->cd();
    wordPHD2        = new WordHist( "PHD2"    , 12 , 0.5 , 0.4 );
    c2 ->cd();
    wordTHD1        = new WordHist( "THD1"    , 12 , 0.5 , 0.6 );
    c2 ->cd();
    wordTHD2        = new WordHist( "THD2"    , 12 , 0.5 , 0.8 );

}

void DiagMaker::endJob()
{
  
  c1 -> cd();

  wordTrkPhi   -> pad -> cd();
  wordTrkPhi   -> formatWordHist();
  wordTrkTheta -> pad -> cd();
  wordTrkTheta -> formatWordHist();
  wordTrknmb   -> pad -> cd();
  wordTrknmb   -> formatWordHist();
  wordValid    -> pad -> cd();
  wordValid    -> formatWordHist();
  wordQuality  -> pad -> cd();
  wordQuality  -> formatWordHist();
  wordKeywire  -> pad -> cd();
  wordKeywire  -> formatWordHist();
  wordStrip    -> pad -> cd();
  wordStrip    -> formatWordHist();
  wordPattern  -> pad -> cd();
  wordPattern  -> formatWordHist();
  wordBend     -> pad -> cd();
  wordBend     -> formatWordHist();
  wordBx       -> pad -> cd();
  wordBx       -> formatWordHist();

  gSystem->ProcessEvents();
  TImage *img1 = TImage::Create();
  img1 -> FromPad(c1);
  img1 -> WriteImage("wordHist1.png");
  delete c1;
  delete img1;

  c2 -> cd();

  wordMpclink  -> pad -> cd();
  wordMpclink  -> formatWordHist();
  wordBx0      -> pad -> cd();
  wordBx0      -> formatWordHist();
  wordSyncErr  -> pad -> cd();
  wordSyncErr  -> formatWordHist();
  wordCscID    -> pad -> cd();
  wordCscID    -> formatWordHist();
  wordRank     -> pad -> cd();
  wordRank     -> formatWordHist();
  wordPHD1     -> pad -> cd();
  wordPHD1     -> formatWordHist();
  wordPHD2     -> pad -> cd();
  wordPHD2     -> formatWordHist();
  wordTHD1     -> pad -> cd();
  wordTHD1     -> formatWordHist();
  wordTHD2     -> pad -> cd();
  wordTHD2     -> formatWordHist();

  gSystem->ProcessEvents();
  TImage *img2 = TImage::Create();
  img2->FromPad(c2);
  img2->WriteImage("wordHist2.png");
  delete c2;
  delete img2;
  
  

}

void formatHist(TH2F* hist, string histTitle, int nXBins, int xLow, int xHigh, string xTitle,
                int nYBins, int yLow, int yHigh, string yTitle)
{
  TAxis *xaxis = hist->GetXaxis();
  xaxis -> SetTitle(xTitle.c_str());
  xaxis -> CenterTitle();
  xaxis -> SetNdivisions(nXBins,kFALSE);
  xaxis -> SetTickLength(1.0);
  xaxis -> CenterLabels();

  TAxis *yaxis = hist->GetYaxis();
  yaxis -> SetTitle(yTitle.c_str());
  yaxis -> CenterTitle();
  yaxis -> SetNdivisions(nYBins,kFALSE);
  yaxis -> SetTickLength(1.0);
  yaxis -> CenterLabels();

  hist -> SetTitle(histTitle.c_str());
  hist -> SetStats(0);

}

void pad2png(TH2F *h, string imgName)
{
    TCanvas *c = new TCanvas;
    h->Draw("colz");
    gSystem->ProcessEvents();
    TImage *img = TImage::Create();
    img->FromPad(c);
    img->WriteImage(imgName.c_str());
    delete c;
    delete img;
}


bool WordHist::fillWord(int wordData, int wordEmu)
{
  int lowBit = -1;
  int numBits = 0;
  if (lowBit < 0)
    lowBit = hist -> GetBinError(0);

  if (numBits < 1)
    numBits = hist -> GetNbinsX();

  wordData = wordData >> lowBit;
  wordEmu  = wordEmu  >> lowBit;

  int yFill;

  bool matchFlag = true;

  for (int i = 0; i < numBits; i++)
  {

    yFill = (wordData & 0x1) - (wordEmu & 0x1);

    matchFlag |= (yFill == 0);

    hist -> Fill (numBits - i - 0.5, yFill);

    wordData = wordData >> 1;
    wordEmu  = wordEmu >> 1;

  }

  return matchFlag;
}



// WordHist Constructor for 1D files
WordHist::WordHist(string theVarName, int theNBits, double theXCornerVal, double theYCornerVal) : nBits(theNBits)
{
    varName = theVarName;
    xCornerVal = theXCornerVal;
    yCornerVal = theYCornerVal;
    tPadHeight = 0.15;
    lineColor = kBlack;
    fillColor = kWhite;
    setFormatValues();

    if(tPadLength + xCornerVal > 1.0)
        {
         cout << "Error: pad length is too long for x corner value.\n";
         exit(1);
        }

    pad = new TPad("pad","",xCornerVal,yCornerVal,xCornerVal+tPadLength,yCornerVal+tPadHeight);
    hist = new TH2F("hist","",nBits,0,nBits,3,-1.5,1.5);

    pad -> SetFillColor(fillColor);
    pad -> SetLineColor(lineColor);
    pad -> SetBottomMargin(0.4);
    pad -> SetRightMargin(rightMargin);
    pad -> SetLeftMargin(leftMargin);
    pad -> Draw();
    pad -> cd();

}

void WordHist::formatWordHist()
{
///    cout << "Creating the functions.\n";
    TF1 *f1 = new TF1("f1","12-x",1,nBits+1);
    TF1 *f2 = new TF1("f2","x",-1,2);

///    cout << "Setting axis color, doing ah col.\n";
    hist -> SetAxisColor(lineColor);
    hist -> SetStats(kFALSE);
    hist -> Draw("ah col");

///    cout << "Creating xtick stuff.\n";
    TGaxis *xtick = new TGaxis(0, -1.5,nBits,-1.5,
                "f1",nBits+1,"US");

    xtick -> SetLineWidth(2.5);
    xtick -> SetTickSize(0.12);
    xtick -> Draw("same");

///    cout << "Creating xaxis stuff.\n";
    TGaxis *xaxis = new TGaxis(-1, -1.5,nBits-1,-1.5,
                "f1",nBits+1,"MNIBS");

    xaxis -> SetTickSize(0);
    xaxis -> SetLabelSize(0.14);
    xaxis -> SetLabelColor(lineColor);

    if(nBits > 3)  xaxis -> SetTitle((varName + " word bit").c_str());
    if(nBits == 3) xaxis -> SetTitle(("         " + varName + " word bit").c_str());
    if(nBits == 2) xaxis -> SetTitle(("            " + varName + " word bit").c_str());
    if(nBits == 1) xaxis -> SetTitle(("            " + varName + " word bit").c_str());

    xaxis -> SetTitleColor(lineColor);
    xaxis -> SetTitleSize(0.16);
    xaxis -> CenterTitle();

    xaxis -> Draw("same");

///    cout << "Creating yaxis stuff.\n";
    TGaxis *yaxis = new TGaxis(0, -1.5,0,1.5,
                "f2",4,"MNI");

    yaxis -> SetTickSize(-1);

    yaxis -> SetLabelColor(lineColor);
    yaxis -> SetLineWidth(2.5);
    yaxis -> SetLabelSize(0.14);
    yaxis -> SetLabelOffset(yaxisLabelOffset);

    yaxis -> SetTitle("data - emul");
    yaxis -> SetTitleColor(lineColor);
    yaxis -> SetTitleSize(0.15);
    yaxis -> SetTitleOffset(yaxisTitleOffset);

    yaxis -> Draw("same");

///    cout << "Creating xtop stuff.\n";
    TGaxis *xtop = new TGaxis(0,1.5,nBits,1.5,"f1",nBits+1,"US-");
    xtop -> SetLineColor(lineColor);
    xtop -> SetTickSize(0.12);
    xtop -> SetLineWidth(2.5);
    xtop -> Draw("same");

///    cout << "Creating yright stuff.\n";
    TGaxis *yright = new TGaxis(nBits,-1.5,nBits,1.5,"f2",4,"+US");
    yright -> SetLineColor(lineColor);
    yright -> SetLineWidth(2.5);
    yright -> Draw("same");

///    cout << "Creating arrays xPoints yPoints ex ey\n";
    double xPoints[(nBits-1)*2];
    double yPoints[(nBits-1)*2];
    double ex[(nBits-1)*2];
    double ey[(nBits-1)*2];

    for (int i = 0; i < nBits-1; i++){

      xPoints[2*i] = i+1;
      xPoints[2*i+1] = i+1;

      yPoints[2*i] = -0.5;
      yPoints[2*i+1] = 0.5;

    }

    for (int i = 0; i < (nBits-1)*2; i++){

      ex[i] = 0.25;
      ey[i] = 0.25;
    }
///    cout << "Creating gridgraph.\n";
    TGraphErrors *gridGraph;

    gridGraph = new TGraphErrors((nBits-1)*2, xPoints, yPoints, ex, ey);

    gridGraph -> SetLineColor(lineColor);
    gridGraph -> SetLineWidth(0.4);

    gridGraph -> Draw("pz same");
///    cout << "Updating the pad.\n";
    //pad -> Update();
///    cout << "Deleting f1 and f2.\n";
    delete f1;
    delete f2;

}




void WordHist::setFormatValues()
{

 // Setting the nBits-dependent histogram format values
  if (nBits > 3)
    {
     tPadLength = 0.0206*nBits + 0.046;
     leftMargin = 1.0 / (0.5542*nBits + 1.23);
     rightMargin = 1.0 / (2.624*nBits + 0.113);
     yaxisTitleOffset = 1.0 / (0.1942*nBits + 0.3902);
     yaxisLabelOffset = 1.0 / (4.0931*nBits + 1.732);
    }
  else
    {
     switch(nBits)
        {
         case 3:
          {
           tPadLength = 0.1274;
           leftMargin = 0.292;
           rightMargin = 0.225;
           yaxisTitleOffset = 0.80;
           yaxisLabelOffset = 0.05;
          }
          break;
        case 2:
          {
           tPadLength = 0.1274;
           leftMargin = 0.3708;
           rightMargin = 0.3034;
           yaxisTitleOffset = 1.0;
           yaxisLabelOffset = 0.05;
          }
          break;
        case 1:
          {
           tPadLength = 0.1274;
           leftMargin = 0.4494;
           rightMargin = 0.3820;
           yaxisTitleOffset = 1.50;
           yaxisLabelOffset = 0.05;
          }
          break;
        }
    }
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DiagMaker);





