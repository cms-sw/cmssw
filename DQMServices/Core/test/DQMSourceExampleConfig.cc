// -*- C++ -*-
//
// Package:    DQMServices/RootImpl
// Class:      DQMSourceExampleConfig
//
/**\class DQMSourceExampleConfig

Description: Simple example showing how to create a DQM Source creating & shipping
monitoring elements

Implementation:
<Notes on implementation>
*/
//
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <TRandom.h>  // this is just the random number generator

#include <cmath>
#include <vector>
#include <sstream>
using std::cout;
using std::endl;
using std::string;

//
// class declaration
//

class DQMSourceExampleConfig : public edm::EDAnalyzer {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit DQMSourceExampleConfig(const edm::ParameterSet&);
  ~DQMSourceExampleConfig() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void endJob() override;

  void recursiveBuild(string, int, int, int, DQMStore*, int);

private:
  // ----------member data ---------------------------

  //    MonitorElement * tree;		//Monitor Element containing all the structure

  std::vector<MonitorElement*> meContainer;

  float XMIN;
  float XMAX;
  // event counter
  int counter;
  // back-end interface
  DQMStore* dbe;

  const int NBINS;

  int message_size;
  int directories_number;
  int histograms_number;
  int histoCounter;
  int dirCounter;
  int dirSpacer;

  int difference;
  int histoDiff;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DQMSourceExampleConfig::DQMSourceExampleConfig(const edm::ParameterSet& iConfig)
    : meContainer(0), counter(0), NBINS(50) {
  message_size = iConfig.getUntrackedParameter<int>("mesSize", 1000);

  directories_number = iConfig.getUntrackedParameter<int>("dirNumber", 2);

  histograms_number = iConfig.getUntrackedParameter<int>("histoNumber", 10);

  //     get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();

  XMIN = 0;
  XMAX = 50;

  histoCounter = 1;
  dirCounter = 1;
  dirSpacer = 1;

  recursiveBuild("",
                 histograms_number / directories_number,
                 histograms_number,
                 iConfig.getUntrackedParameter<int>("lDepth", 2),
                 dbe,
                 directories_number);
}

DQMSourceExampleConfig::~DQMSourceExampleConfig() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void DQMSourceExampleConfig::endJob() { dbe->save("test.root"); }

//
// member functions
//

// // ------------ method called to produce the data  ------------
void DQMSourceExampleConfig::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Filling the histogram with random data
  srand(0);

  if (counter % 1000 == 0)
    cout << " # of cycles = " << counter << endl;

  for (int i = 0; i != 10; ++i) {
    for (auto& it : meContainer)
      it->Fill(gRandom->Gaus(30, 3), 1.0);
  }
  usleep(1000000);

  ++counter;
}

void DQMSourceExampleConfig::recursiveBuild(
    string dirName, int histo_X_dir, int histoLeft, int level_depth, DQMStore* dbe2, int directories_left) {
  if (histoLeft <= 0)
    return;

  std::stringstream ss;
  string str;
  ss << dirSpacer;
  ss >> str;

  dirName = dirName + "C" + str + "/";

  // create and cd into new folder
  dbe->setCurrentFolder(dirName);
  cout << "CURRENT FOLDER: " << dirName << "   SET!" << endl;

  for (int i = 0; i < histo_X_dir; ++i) {
    if (histoLeft - (i + 1) < 0)
      return;

    std::stringstream ss2;
    string str2;
    ss2 << histoCounter;
    ss2 >> str2;
    histoCounter++;

    string name = "histo" + str2;
    string title = "Example " + str2 + " 2D histogram.";
    cout << " Will create ME w/ name = " << name << " and title = " << title << " in directory " << dbe->pwd() << endl;

    meContainer.push_back(dbe->book1D(name, title, NBINS, XMIN, XMAX));
  }

  if (directories_left < 1)
    return;

  difference = (directories_left - 1) % level_depth;
  histoDiff = (histoLeft - histo_X_dir) % (level_depth);
  int dirDIFF = directories_left - 1;

  if ((directories_left - 1) / level_depth < 1) {
    histoDiff = (histoLeft - histo_X_dir) % (directories_left - 1);
    difference = 1;
  }

  for (int i = 0; i < level_depth; ++i) {
    if (directories_left <= 1)
      return;

    dirSpacer++;

    if (directories_left - 1 > level_depth) {
      recursiveBuild(dirName,
                     (((histoLeft - histo_X_dir) / level_depth) + histoDiff) /
                         (((directories_left - 1) / level_depth) + difference),
                     (histoLeft - histo_X_dir) / level_depth + histoDiff,
                     level_depth,
                     dbe2,
                     ((directories_left - 1) / level_depth) + difference /*2*(directories_left+difference)-1*/);
      directories_left = directories_left - ((directories_left - 1) / level_depth) + difference;
    } else {
      recursiveBuild(dirName,
                     (((histoLeft - histo_X_dir) / dirDIFF) + histoDiff),
                     (histoLeft - histo_X_dir) / (directories_left - 1) + histoDiff,
                     level_depth,
                     dbe2,
                     0 /*2*(directories_left+difference)-1*/);
      directories_left--;
    }

    dbe2->setCurrentFolder(dirName);
    histoDiff = 0;
    difference = 1;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DQMSourceExampleConfig);
