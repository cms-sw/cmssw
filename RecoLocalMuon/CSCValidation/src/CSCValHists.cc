#include "RecoLocalMuon/CSCValidation/src/CSCValHists.h"

using namespace std;


  CSCValHists::CSCValHists(){

    cout << "Initializing Histogram Manager..." << endl;

  }  


  CSCValHists::~CSCValHists(){

  }


  void CSCValHists::writeHists(TFile* theFile){

    vector<string> theFolders;
    vector<string>::iterator fit;
    theFile->cd();

    map<string,pair<TH1*,string> >::const_iterator mapit;
    for (mapit = theMap.begin(); mapit != theMap.end(); mapit++){
      string folder = (*mapit).second.second.c_str();
      fit = find(theFolders.begin(), theFolders.end(), folder);
      if (fit == theFolders.end()){
        theFolders.push_back(folder);
        theFile->mkdir(folder.c_str());
      }
      theFile->cd((*mapit).second.second.c_str());
      (*mapit).second.first->Write();
      theFile->cd();
    }

  }

  
  void CSCValHists::writeTrees(TFile* theFile){

    theFile->cd("recHits");
    rHTree->Write();
    theFile->cd();

    theFile->cd("Segments");
    segTree->Write();
    theFile->cd();


  }

  
  void CSCValHists::setupTrees(){

    // Create the root tree to hold position info
    rHTree  = new TTree("rHPositions","Local and Global reconstructed positions for recHits");
    segTree = new TTree("segPositions","Local and Global reconstructed positions for segments");

    // Create a branch on the tree
    rHTree->Branch("rHpos",&rHpos,"endcap/I:station/I:ring/I:chamber/I:layer/I:localx/F:localy/F:globalx/F:globaly/F");
    segTree->Branch("segpos",&segpos,"endcap/I:station/I:ring/I:chamber/I:layer/I:localx/F:localy/F:globalx/F:globaly/F");

  }


  
  void CSCValHists::printPlots(){

    plotMaker = new CSCValPlotFormatter();
    plotMaker->makePlots(theMap);
    plotMaker->makeGlobalScatterPlots(rHTree,"rechits");
    plotMaker->makeGlobalScatterPlots(segTree,"segments");


  }

  
  void CSCValHists::printComparisonPlots(string refFile){

    plotMaker = new CSCValPlotFormatter();
    plotMaker->makeComparisonPlots(theMap,refFile);

  }

  
  void CSCValHists::fillRechitTree(float x, float y, float gx, float gy, int en, int st, int ri, int ch, int la){

    // Fill the rechit position branch
    rHpos.localx  = x;
    rHpos.localy  = y;
    rHpos.globalx = gx;
    rHpos.globaly = gy;
    rHpos.endcap  = en;
    rHpos.ring    = ri;
    rHpos.station = st;
    rHpos.chamber = ch;
    rHpos.layer   = la;
    rHTree->Fill();

  }
  
  void CSCValHists::fillSegmentTree(float x, float y, float gx, float gy, int en, int st, int ri, int ch){

    // Fill the segment position branch
    segpos.localx  = x;
    segpos.localy  = y;
    segpos.globalx = gx;
    segpos.globaly = gy;
    segpos.endcap  = en;
    segpos.ring    = ri;
    segpos.station = st;
    segpos.chamber = ch;
    segpos.layer   = 0;
    segTree->Fill();

  }
  

  void CSCValHists::fillCalibHist(float x, string name, string title, int bins, float xmin, float xmax,
                                  int bin, string folder){

    map<string,pair<TH1*,string> >::iterator it;
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH1I(name.c_str(),title.c_str(),bins,xmin,xmax), folder);
    }

    theMap[name].first->SetBinContent(bin,x);

  }


  void CSCValHists::fill1DHist(float x, string name, string title,
                               int bins, float xmin, float xmax, string folder){

    map<string,pair<TH1*,string> >::iterator it;
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH1I(name.c_str(),title.c_str(),bins,xmin,xmax), folder);
    }


    theMap[name].first->Fill(x);

  }


  void CSCValHists::fill2DHist(float x, float y, string name, string title,
                               int binsx, float xmin, float xmax,
                               int binsy, float ymin, float ymax, string folder){

    map<string,pair<TH1*,string> >::iterator it;
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH2F(name.c_str(),title.c_str(),binsx,xmin,xmax,binsy,ymin,ymax),folder);
    }

    theMap[name].first->Fill(x,y);

  }


  void CSCValHists::fill1DHistByType(float x, string name, string title, CSCDetId id,
                                     int bins, float xmin, float xmax, string folder){

    string endcap;
    if (id.endcap() == 1) endcap = "+";
    if (id.endcap() == 2) endcap = "-";

    map<string,pair<TH1*,string> >::iterator it;
    ostringstream oss1;
    ostringstream oss2;
    oss1 << name << endcap << id.station() << id.ring();
    oss2 << title << "  (ME " << endcap << id.station() << "/" << id.ring() << ")";
    name = oss1.str();
    title = oss2.str();
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH1F(name.c_str(),title.c_str(),bins,xmin,xmax), folder);
    }

    theMap[name].first->Fill(x);

  }

  void CSCValHists::fill2DHistByType(float x, float y, string name, string title, CSCDetId id,
                                   int binsx, float xmin, float xmax,
                                   int binsy, float ymin, float ymax, string folder){

    string endcap;
    if (id.endcap() == 1) endcap = "+";
    if (id.endcap() == 2) endcap = "-";

    map<string,pair<TH1*,string> >::iterator it;
    ostringstream oss1;
    ostringstream oss2;
    oss1 << name << endcap << id.station() << id.ring();
    oss2 << title << "  (ME " << endcap << id.station() << "/" << id.ring() << ")";
    name = oss1.str();
    title = oss2.str();
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH2F(name.c_str(),title.c_str(),binsx,xmin,xmax,binsy,ymin,ymax), folder);
    }

    theMap[name].first->Fill(x,y);

  }


  void CSCValHists::fill2DHistByStation(float x, float y, string name, string title, CSCDetId id,
                                        int binsx, float xmin, float xmax,
                                        int binsy, float ymin, float ymax, string folder){

    string endcap;
    if (id.endcap() == 1) endcap = "+";
    if (id.endcap() == 2) endcap = "-";

    map<string,pair<TH1*,string> >::iterator it;
    ostringstream oss1;
    ostringstream oss2;
    oss1 << name << endcap << id.station();
    oss2 << title << "  (Station " << endcap << id.station() << ")";
    name = oss1.str();
    title = oss2.str();
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH2F(name.c_str(),title.c_str(),binsx,xmin,xmax,binsy,ymin,ymax), folder);
    }

    theMap[name].first->Fill(x,y);

  }


  void CSCValHists::fill1DHistByChamber(float x, string name, string title, CSCDetId id,
                                       int bins, float xmin, float xmax, string folder){

    string endcap;
    if (id.endcap() == 1) endcap = "+";
    if (id.endcap() == 2) endcap = "-";

    map<string,pair<TH1*,string> >::iterator it;
    ostringstream oss1;
    ostringstream oss2;
    oss1 << name << "_" << endcap << id.station() << "_" << id.ring() << "_" << id.chamber();
    oss2 << title << "  (ME " << endcap << id.station() << "/" << id.ring() << "/" << id.chamber() << ")";
    name = oss1.str();
    title = oss2.str();
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH1F(name.c_str(),title.c_str(),bins,xmin,xmax),folder);
    }

    theMap[name].first->Fill(x);

  }


  void CSCValHists::fill2DHistByChamber(float x, float y, string name, string title, CSCDetId id,
                                        int binsx, float xmin, float xmax,
                                        int binsy, float ymin, float ymax, string folder){

    string endcap;
    if (id.endcap() == 1) endcap = "+";
    if (id.endcap() == 2) endcap = "-";

    map<string,pair<TH1*,string> >::iterator it;
    ostringstream oss1;
    ostringstream oss2;
    oss1 << name << "_" << endcap << id.station() << "_" << id.ring() << "_" << id.chamber();
    oss2 << title << "  (ME " << endcap << id.station() << "/" << id.ring() << "/" << id.chamber() << ")";
    name = oss1.str();
    title = oss2.str();
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH2F(name.c_str(),title.c_str(),binsx,xmin,xmax,binsy,ymin,ymax),folder);
    }

    theMap[name].first->Fill(x,y);

  }


  void CSCValHists::fillOccupancyHistos(const bool wo[2][4][4][36], const bool sto[2][4][4][36],
                                        const bool ro[2][4][4][36], const bool so[2][4][4][36]){

    map<string,pair<TH1*,string> >::iterator it;
    string name1 = "hOWires";
    string name2 = "hOStrips";
    string name3 = "hORecHits";
    string name4 = "hOSegments";
    it = theMap.find(name1);
    if (it == theMap.end()){
      theMap[name1] = pair<TH1*,string>(new TH2I("hOWires","Wire Digi Occupancy",36,0.5,36.5,20,0.5,20.5),"Digis");
    }
    it = theMap.find(name2);
    if (it == theMap.end()){
      theMap[name2] = pair<TH1*,string>(new TH2I("hOStrips","Strip Digi Occupancy",36,0.5,36.5,20,0.5,20.5),"Digis");
    }
    it = theMap.find(name3);
    if (it == theMap.end()){
      theMap[name3] = pair<TH1*,string>(new TH2I("hORecHits","RecHit Occupancy",36,0.5,36.5,20,0.5,20.5),"recHits");
    }
    it = theMap.find(name4);
    if (it == theMap.end()){
      theMap[name4] = pair<TH1*,string>(new TH2I("hOSegments","Segments Occupancy",36,0.5,36.5,20,0.5,20.5),"Segments");
    }


    for (int e = 0; e < 2; e++){
      for (int s = 0; s < 4; s++){
        for (int r = 0; r < 4; r++){
          for (int c = 0; c < 36; c++){
            int type = 0;
            if ((s+1) == 1) type = (r+1);
            else type = (s+1)*2 + (r+1);
            if ((e+1) == 1) type = type + 10;
            if ((e+1) == 2) type = 11 - type;
            if (wo[e][s][r][c]) theMap[name1].first->Fill((c+1),type);
            if (sto[e][s][r][c]) theMap[name2].first->Fill((c+1),type);
            if (ro[e][s][r][c]) theMap[name3].first->Fill((c+1),type);
            if (so[e][s][r][c]) theMap[name4].first->Fill((c+1),type);
          }
        }
      }
    }


  }


  void CSCValHists::fillProfile(float x, float y, string name, string title,
                                int binsx, float xmin, float xmax,
                                float ymin, float ymax, string folder){

    map<string,pair<TH1*,string> >::iterator it;
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TProfile*,string>(new TProfile(name.c_str(),title.c_str(),binsx,xmin,xmax,ymin,ymax), folder);
    }

    theMap[name].first->Fill(x,y);

  }


  void CSCValHists::fillProfileByType(float x, float y, string name, string title, CSCDetId id,
                                      int binsx, float xmin, float xmax,
                                      float ymin, float ymax, string folder){

    map<string,pair<TH1*,string> >::iterator it;
    string endcap;
    if (id.endcap() == 1) endcap = "+";
    if (id.endcap() == 2) endcap = "-";

    ostringstream oss1;
    ostringstream oss2;
    oss1 << name << endcap << id.station() << id.ring();
    oss2 << title << "  (ME " << endcap << id.station() << "/" << id.ring() << ")";
    name = oss1.str();
    title = oss2.str();

    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TProfile*,string>(new TProfile(name.c_str(),title.c_str(),binsx,xmin,xmax,ymin,ymax), folder);
    }

    theMap[name].first->Fill(x,y);

  }


  void CSCValHists::fillProfileByChamber(float x, float y, string name, string title, CSCDetId id,
                                         int binsx, float xmin, float xmax,
                                         float ymin, float ymax, string folder){

    map<string,pair<TH1*,string> >::iterator it;
    string endcap;
    if (id.endcap() == 1) endcap = "+";
    if (id.endcap() == 2) endcap = "-";

    ostringstream oss1;
    ostringstream oss2;
    oss1 << name << "_" << endcap << id.station() << "_" << id.ring() << "_" << id.chamber();
    oss2 << title << "  (ME " << endcap << id.station() << "/" << id.ring() << "/" << id.chamber() << ")";
    name = oss1.str();
    title = oss2.str();

    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TProfile*,string>(new TProfile(name.c_str(),title.c_str(),binsx,xmin,xmax,ymin,ymax), folder);
    }

    theMap[name].first->Fill(x,y);

  }


  unsigned short CSCValHists::tempChamberType( unsigned short istation, unsigned short iring ) {
    int i = 2 * istation + iring; // i=2S+R ok for S=2, 3, 4
    if ( istation == 1 ) {
      --i;                       // ring 1R -> i=1+R (2S+R-1=1+R for S=1)
      if ( i > 4 ) i = 1;        // But ring 1A (R=4) -> i=1
    }
    return i;
  }


  int CSCValHists::typeIndex(CSCDetId id){

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

