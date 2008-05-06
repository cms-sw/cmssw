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
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cmath>

#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TProfile.h"
#include "TFile.h"
#include "TTree.h"
#include "RecoLocalMuon/CSCValidation/src/CSCValPlotFormatter.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"


class CSCValHists{

  public:

  // constructor
  CSCValHists();

  // destructor
  ~CSCValHists();

    // write histograms the theFile
  void writeHists(TFile* theFile);

  // write trees to theFile
  void writeTrees(TFile* theFile);

  // setup trees
  void setupTrees();

  // produce a set of .gif files from produced histos
  void printPlots();

  // produce a set of compoarison .gif files from produced hists and same plots in reference file
  void printComparisonPlots(string refFile);

  // fill the global rechit position tree (this needs work!)  
  void fillRechitTree(float x, float y, float gx, float gy,
                      int en, int st, int ri, int ch, int la);

  // fill the global segment position tree
  void fillSegmentTree(float x, float y, float gx, float gy,
                       int en, int st, int ri, int ch);

  // calib hists are special because they are constants stored in a histogram, 1 per bin
  void fillCalibHist(float x, string name, string title, int bins, float xmin, float xmax,
                     int bin, string folder);

  // fill 1D histogram 
  void fill1DHist(float x, string name, string title,
                  int bins, float xmin, float xmax, string folder);

  // fill 2D histogram
  void fill2DHist(float x, float y, string name, string title,
                  int binsx, float xmin, float xmax,
                  int binsy, float ymin, float ymax, string folder);

  // fill 1D histogram
  // a histogram is created for every chamber type
  void fill1DHistByType(float x, string name, string title, CSCDetId id,
                        int bins, float xmin, float xmax, string folder);

  // fill 2D histogram
  // a histogram is created for every chamber type
  void fill2DHistByType(float x, float y, string name, string title, CSCDetId id,
                        int binsx, float xmin, float xmax,
                        int binsy, float ymin, float ymax, string folder);

  // fill 2D histogram
  // a histogram is created for every station
  void fill2DHistByStation(float x, float y, string name, string title, CSCDetId id,
                           int binsx, float xmin, float xmax,
                           int binsy, float ymin, float ymax, string folder);

  // fill 1D histogram
  // a histogram is created for every chamber
  void fill1DHistByChamber(float x, string name, string title, CSCDetId id,
                           int bins, float xmin, float xmax, string folder);

  // fill 2D histogram
  // a histogram is created for every chamber
  void fill2DHistByChamber(float x, float y, string name, string title, CSCDetId id,
                           int binsx, float xmin, float xmax,
                           int binsy, float ymin, float ymax, string folder);

  // special plot inspired by CSC online DQM to summarize occupancies
  void fillOccupancyHistos(const bool wo[2][4][4][36], const bool sto[2][4][4][36],
                           const bool ro[2][4][4][36], const bool so[2][4][4][36]);

  // make a profile histogram
  void fillProfile(float x, float y, string name, string title, 
                   int binsx, float xmin, float xmax,
                   float ymin, float ymax, string folder);

  // make a profile histogram
  // one will be made for every chamber type
  void fillProfileByType(float x, float y, string name, string title, CSCDetId id,
                         int binsx, float xmin, float xmax,
                         float ymin, float ymax, string folder);

  // make a profile histogram
  // one will be made for every chamber type
  void fillProfileByChamber(float x, float y, string name, string title, CSCDetId id,
                            int binsx, float xmin, float xmax,
                            float ymin, float ymax, string folder);

  // this is a temp function to mimic a function to be implemented in the CSCDetId for 2_1_0
  unsigned short tempChamberType( unsigned short istation, unsigned short iring );

  // linearlized index bases on endcap, station, and ring based on CSCDetId
  int typeIndex(CSCDetId id);

  protected:

  private:

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

  // If you want to make pretty plots
  CSCValPlotFormatter *plotMaker;
  bool makePlots;

};

#endif   
