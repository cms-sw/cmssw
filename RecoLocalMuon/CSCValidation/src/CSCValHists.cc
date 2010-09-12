#include "RecoLocalMuon/CSCValidation/src/CSCValHists.h"
#include <algorithm>

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

  void CSCValHists::insertPlot(TH1* thePlot, string name, string folder){

    theMap[name] = pair<TH1*,string>(thePlot, folder);

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

  void CSCValHists::fill1DHistByLayer(float x, string name, string title, CSCDetId id,
                                      int bins, float xmin, float xmax, string folder){

    string endcap;
    if (id.endcap() == 1) endcap = "+";
    if (id.endcap() == 2) endcap = "-";

    map<string,pair<TH1*,string> >::iterator it;
    ostringstream oss1;
    ostringstream oss2;
    oss1 << name << "_" << endcap << id.station() << "_" << id.ring() << "_" << id.chamber() << "_L" << id.layer();
    oss2 << title << "  (ME " << endcap << id.station() << "/" << id.ring() << "/" << id.chamber() << "/L" << id.layer() << ")";
    name = oss1.str();
    title = oss2.str();
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH1F(name.c_str(),title.c_str(),bins,xmin,xmax),folder);
    }

    theMap[name].first->Fill(x);

  }


  void CSCValHists::fill2DHistByLayer(float x, float y, string name, string title, CSCDetId id,
                                      int binsx, float xmin, float xmax,
                                      int binsy, float ymin, float ymax, string folder){

    string endcap;
    if (id.endcap() == 1) endcap = "+";
    if (id.endcap() == 2) endcap = "-";

    map<string,pair<TH1*,string> >::iterator it;
    ostringstream oss1;
    ostringstream oss2;
    oss1 << name << "_" << endcap << id.station() << "_" << id.ring() << "_" << id.chamber() << "_L" << id.layer();;
    oss2 << title << "  (ME " << endcap << id.station() << "/" << id.ring() << "/" << id.chamber() << "/L" << id.layer() << ")";
    name = oss1.str();
    title = oss2.str();
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH2F(name.c_str(),title.c_str(),binsx,xmin,xmax,binsy,ymin,ymax),folder);
    }

    theMap[name].first->Fill(x,y);

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


  void CSCValHists::fill2DProfile(float x, float y, float z, string name, string title,
                                  int binsx, float xmin, float xmax,
                                  int binsy, float ymin, float ymax,
                                  float zmin, float zmax, string folder){

    map<string,pair<TH1*,string> >::iterator it;

    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TProfile2D*,string>(new TProfile2D(name.c_str(),title.c_str(),binsx,xmin,xmax,binsy,ymin,ymax,zmin,zmax), folder);
    }

    TProfile2D *tempp = (TProfile2D*)theMap[name].first;
    tempp->Fill(x,y,z);

  }

