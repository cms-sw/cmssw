#ifndef RecoLocalMuon_CSCValHists_H
#define RecoLocalMuon_CSCValHists_H


/** \class CSCValHists
 *
 *  Manages Histograms for CSCValidation
 *
 *  Andy Kubik - Northwestern University
 *
 */


// system include files
#include <memory>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <iomanip>
#include <fstream>
#include <cmath>

#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TProfile.h"
#include "TFile.h"
#include "TTree.h"

using namespace std;
using namespace edm;

class CSCValHists{

  public:

  // constructor
  CSCValHists(){

    cout << "Initializing Histogram Manager..." << endl;

  }


  // destructor
  ~CSCValHists(){

  }


  // write histograms
  void writeHists(TFile* theFile){

    theFile->mkdir("Digis");
    theFile->mkdir("recHits");
    theFile->mkdir("Segments");
    theFile->mkdir("Calib");
    theFile->mkdir("PedestalNoise");
    theFile->cd();    

    map<string,pair<TH1*,string> >::const_iterator mapit;
    for (mapit = theMap.begin(); mapit != theMap.end(); mapit++){
      theFile->cd((*mapit).second.second.c_str());
      (*mapit).second.first->Write();
      theFile->cd();
    }


  }

  void writeTrees(TFile* theFile){

    theFile->cd("recHits");
    rHTree->Write();
    theFile->cd();

    theFile->cd("Segments");
    segTree->Write();
    theFile->cd();


  }

  // setup trees
  void setupTrees(){

        // Create the root tree to hold position info
      rHTree  = new TTree("rHPositions","Local and Global reconstructed positions for recHits");
      segTree = new TTree("segPositions","Local and Global reconstructed positions for segments");

      // Create a branch on the tree
      rHTree->Branch("rHpos",&rHpos,"endcap/I:station/I:ring/I:chamber/I:layer/I:localx/F:localy/F:globalx/F:globaly/F");
      segTree->Branch("segpos",&segpos,"endcap/I:station/I:ring/I:chamber/I:layer/I:localx/F:localy/F:globalx/F:globaly/F");

  }

  void fillRechitTree(float x, float y, float gx, float gy,
                      int en, int st, int ri, int ch, int la){
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


  void fillSegmentTree(float x, float y, float gx, float gy,
                       int en, int st, int ri, int ch){

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

  // fill Calib histogram
  void fillCalibHist(float x, string name, string title, int bins, float xmin, float xmax,
                     int bin, string folder){

    map<string,pair<TH1*,string> >::iterator it;
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH1I(name.c_str(),title.c_str(),bins,xmin,xmax), folder);
    }

    theMap[name].first->SetBinContent(bin,x);

  }

  // fill 1D histogram 
  void fill1DHist(float x, string name, string title,
                  int bins, float xmin, float xmax, string folder){

    map<string,pair<TH1*,string> >::iterator it;
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH1I(name.c_str(),title.c_str(),bins,xmin,xmax), folder);
    }

    
    theMap[name].first->Fill(x);

  }

  // fill 2D histogram
  void fill2DHist(float x, float y, string name, string title,
                  int binsx, float xmin, float xmax,
                  int binsy, float ymin, float ymax, string folder){

    map<string,pair<TH1*,string> >::iterator it;
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TH1*,string>(new TH2F(name.c_str(),title.c_str(),binsx,xmin,xmax,binsy,ymin,ymax),folder);
    }

    theMap[name].first->Fill(x,y);

  }


  // fill 1D histogram
  // a histogram is created for every chamber type
  void fill1DHistByType(float x, string name, string title, CSCDetId id,
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

  // fill 2D histogram
  // a histogram is created for every chamber type
  void fill2DHistByType(float x, float y, string name, string title, CSCDetId id,
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

  // fill 2D histogram
  // a histogram is created for every chamber type
  void fill2DHistByStation(float x, float y, string name, string title, CSCDetId id,
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


  // fill 1D histogram
  // a histogram is created for every chamber
  void fill1DHistByChamber(float x, string name, string title, CSCDetId id,
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

  // fill 2D histogram
  // a histogram is created for every chamber
  void fill2DHistByChamber(float x, float y, string name, string title, CSCDetId id,
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

  // special plot inspired by CSC online DQM to summarize occupancies
  void fillOccupancyHistos(const bool wo[2][4][4][36], const bool sto[2][4][4][36],
                           const bool ro[2][4][4][36], const bool so[2][4][4][36]){

    map<string,pair<TH1*,string> >::iterator it;
    string name1 = "h0Wires";
    string name2 = "h0Strips";
    string name3 = "h0RecHits";
    string name4 = "h0Segments";
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

  // make a profile histogram
  void fillProfile(float x, float y, string name, string title, 
                   int binsx, float xmin, float xmax,
                   float ymin, float ymax, string folder){

    map<string,pair<TH1*,string> >::iterator it;
    it = theMap.find(name);
    if (it == theMap.end()){
      theMap[name] = pair<TProfile*,string>(new TProfile(name.c_str(),title.c_str(),binsx,xmin,xmax,ymin,ymax), folder);
    }

    theMap[name].first->Fill(x,y);

  } 

  // make a profile histogram
  // one will be made for every chamber type
  void fillProfileByType(float x, float y, string name, string title, CSCDetId id,
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

  // make a profile histogram
  // one will be made for every chamber type
  void fillProfileByChamber(float x, float y, string name, string title, CSCDetId id,
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

  unsigned short tempChamberType( unsigned short istation, unsigned short iring ) {
    int i = 2 * istation + iring; // i=2S+R ok for S=2, 3, 4
    if ( istation == 1 ) {
      --i;                       // ring 1R -> i=1+R (2S+R-1=1+R for S=1)
      if ( i > 4 ) i = 1;        // But ring 1A (R=4) -> i=1
    }   
    return i;
  }

  protected:

  private:


  int typeIndex(CSCDetId id){

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

  // map to hold histograms
  map<string,pair<TH1*,string> > theMap;

  // A struct for creating a Tree/Branch of position info
  struct posRecord {
    int endcap;
    int station;
    int ring;
    int chamber;
    int layer;
    float localx;
    float localy;
    float globalx;
    float globaly;
  } rHpos, segpos;

  // The root tree
  TTree *rHTree;
  TTree *segTree;


};

#endif   
