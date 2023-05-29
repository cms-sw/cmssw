#ifndef RecoLocalMuon_CSCValHists_H
#define RecoLocalMuon_CSCValHists_H

/** \class CSCValHists
 *
 *  Manages Histograms for CSCValidation
 *
 *  original author:
 *  Andy Kubik (Northwestern University)
 *
 *  contact
 *  CSC DPG
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
#include "TProfile2D.h"
#include "TFile.h"
#include "TTree.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCValHists {
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

  // fill the global rechit position tree (this needs work!)
  void fillRechitTree(float x, float y, float gx, float gy, int en, int st, int ri, int ch, int la);

  // fill the global segment position tree
  void fillSegmentTree(float x, float y, float gx, float gy, int en, int st, int ri, int ch);

  // insert any TH1 into the big map
  void insertPlot(TH1* thePlot, std::string name, std::string folder);

  // calib hists are special because they are constants stored in a histogram, 1 per bin
  void fillCalibHist(
      float x, std::string name, std::string title, int bins, float xmin, float xmax, int bin, std::string folder);

  // fill 1D histogram
  void fill1DHist(float x, std::string name, std::string title, int bins, float xmin, float xmax, std::string folder);

  // fill 2D histogram
  void fill2DHist(float x,
                  float y,
                  std::string name,
                  std::string title,
                  int binsx,
                  float xmin,
                  float xmax,
                  int binsy,
                  float ymin,
                  float ymax,
                  std::string folder);

  // fill 1D histogram
  // a histogram is created for every chamber type
  void fill1DHistByType(
      float x, std::string name, std::string title, CSCDetId id, int bins, float xmin, float xmax, std::string folder);

  // fill 2D histogram
  // a histogram is created for every chamber type
  void fill2DHistByType(float x,
                        float y,
                        std::string name,
                        std::string title,
                        CSCDetId id,
                        int binsx,
                        float xmin,
                        float xmax,
                        int binsy,
                        float ymin,
                        float ymax,
                        std::string folder);

  // fill 1D histogram
  // a histogram is created for every peripheral crate
  void fill1DHistByCrate(
      float x, std::string name, std::string title, CSCDetId id, int bins, float xmin, float xmax, std::string folder);

  // fill 2D histogram
  // a histogram is created for every peripheral crate
  void fill2DHistByCrate(float x,
                         float y,
                         std::string name,
                         std::string title,
                         CSCDetId id,
                         int binsx,
                         float xmin,
                         float xmax,
                         int binsy,
                         float ymin,
                         float ymax,
                         std::string folder);

  // fill 2D histogram
  // a histogram is created for every station
  void fill1DHistByStation(
      float x, std::string name, std::string title, CSCDetId id, int bins, float xmin, float xmax, std::string folder);

  // fill 2D histogram
  // a histogram is created for every station
  void fill2DHistByStation(float x,
                           float y,
                           std::string name,
                           std::string title,
                           CSCDetId id,
                           int binsx,
                           float xmin,
                           float xmax,
                           int binsy,
                           float ymin,
                           float ymax,
                           std::string folder);

  // fill 1D histogram
  // a histogram is created for every chamber
  void fill1DHistByChamber(
      float x, std::string name, std::string title, CSCDetId id, int bins, float xmin, float xmax, std::string folder);

  // fill 2D histogram
  // a histogram is created for every chamber
  void fill2DHistByChamber(float x,
                           float y,
                           std::string name,
                           std::string title,
                           CSCDetId id,
                           int binsx,
                           float xmin,
                           float xmax,
                           int binsy,
                           float ymin,
                           float ymax,
                           std::string folder);

  // fill 2D histogram of entire detector
  // once per event
  void fill2DHistByEvent(
      int run, int event, float x, std::string name, std::string title, CSCDetId id, std::string folder);

  // fill 2D histogram of entire detector
  // with a value z for the specified chamber
  void fill2DHist(float z, std::string name, std::string title, CSCDetId id, std::string folder);

  // fill 1D histogram
  // a histogram is created for every layer in every chamber
  void fill1DHistByLayer(
      float x, std::string name, std::string title, CSCDetId id, int bins, float xmin, float xmax, std::string folder);

  // fill 2D histogram
  // a histogram is created for every layer in every chamber
  void fill2DHistByLayer(float x,
                         float y,
                         std::string name,
                         std::string title,
                         CSCDetId id,
                         int binsx,
                         float xmin,
                         float xmax,
                         int binsy,
                         float ymin,
                         float ymax,
                         std::string folder);

  // make a profile histogram
  void fillProfile(float x,
                   float y,
                   std::string name,
                   std::string title,
                   int binsx,
                   float xmin,
                   float xmax,
                   float ymin,
                   float ymax,
                   std::string folder);

  // make a profile histogram
  // one will be made for every chamber type
  void fillProfileByType(float x,
                         float y,
                         std::string name,
                         std::string title,
                         CSCDetId id,
                         int binsx,
                         float xmin,
                         float xmax,
                         float ymin,
                         float ymax,
                         std::string folder);

  // make a profile histogram
  // one will be made for every chamber
  void fillProfileByChamber(float x,
                            float y,
                            std::string name,
                            std::string title,
                            CSCDetId id,
                            int binsx,
                            float xmin,
                            float xmax,
                            float ymin,
                            float ymax,
                            std::string folder);

  // make a 2D profile histogram (usefull for summary plots)
  void fill2DProfile(float x,
                     float y,
                     float z,
                     std::string name,
                     std::string title,
                     int binsx,
                     float xmin,
                     float xmax,
                     int binsy,
                     float ymin,
                     float ymax,
                     float zmin,
                     float zmax,
                     std::string folder);

  // look-up which crate this chamber belongs to
  int crate_lookup(CSCDetId id);

protected:
private:
  // map to hold histograms
  std::map<std::string, std::pair<TH1*, std::string> > theMap;

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
  TTree* rHTree;
  TTree* segTree;
};

#endif
