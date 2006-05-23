#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoPixelVertexing/PixelVertexFinding/interface/PVPositionBuilder.h"

#include <iostream>
#include <vector>
#include "TTree.h"
#include "TFile.h"
#include "TDirectory.h"

using namespace std;

class PixelVertexTest : public edm::EDAnalyzer {
public:
  explicit PixelVertexTest(const edm::ParameterSet& conf);
  ~PixelVertexTest();
  virtual void beginJob(const edm::EventSetup& es);
  virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
  virtual void endJob();
private:
  edm::ParameterSet conf_; 
  // Tree of simple vars for testing resolution eff etc
  TTree *t_;
  TFile *f_;
  int ntrk_;
  int nvtx_;
  double vz_[10];
  double vzwt_[10];
};

PixelVertexTest::PixelVertexTest(const edm::ParameterSet& conf)
  : conf_(conf),t_(0),f_(0)
{
  edm::LogInfo("PixelVertexTest")<<" CTOR";
}

PixelVertexTest::~PixelVertexTest()
{
  edm::LogInfo("PixelVertexTest")<<" DTOR";
  delete f_;
  //  delete t_;
}

void PixelVertexTest::beginJob(const edm::EventSetup& es) {
  // Make my little tree
  std::string file = conf_.getUntrackedParameter<std::string>("OutputTree","mytree.root");
  const char* cwd= gDirectory->GetPath();
  f_ = new TFile(file.c_str(),"RECREATE");
  t_ = new TTree("t","Pixel Vertex Testing");
  t_->Branch("vtx",&nvtx_,"nvtx/I");
  t_->Branch("vz",&vz_,"vz[nvtx]/D");
  t_->Branch("vzwt",&vzwt_,"vzwt[nvtx]/D");
  gDirectory->cd(cwd);
}

void PixelVertexTest::analyze(
    const edm::Event& ev, const edm::EventSetup& es)
{
  cout <<"*** PixelVertexTest, analyze event: " << ev.id() << endl;
  edm::Handle<reco::TrackCollection> trackCollection;
  std::string trackCollName = conf_.getParameter<std::string>("TrackCollection");
  ev.getByLabel(trackCollName,trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());

  std::vector< reco::TrackRef >trks;

  std::cout << *(trackCollection.provenance()) << std::endl;
  cout << "Reconstructed "<< tracks.size() << " tracks" << std::endl;
  for (unsigned int i=0; i<tracks.size(); i++) {
    cout << "\tmomentum: " << tracks[i].momentum()
         << "\tPT: " << tracks[i].pt()<< endl;
    cout << "\tvertex: " << tracks[i].vertex()
         << "\timpact parameter: " << tracks[i].d0()<< endl;
    cout << "\tcharge: " << tracks[i].charge()<< endl;
    trks.push_back( reco::TrackRef(trackCollection, i) );
  cout <<"------------------------------------------------"<<endl;
  }
  PVPositionBuilder pos;  
  nvtx_ = 1;
  vz_[0] = pos.average(trks).value();
  vzwt_[0] = pos.wtAverage(trks).value();
  t_->Fill();
  std::cout << "The average z-position of these tracks is " << vz_ << std::endl;
  std::cout << "The weighted average z-position of these tracks is " << vzwt_ << std::endl;
  
}

void PixelVertexTest::endJob() {
  if (t_) t_->Print();
  if (f_) {
    f_->Print();
    f_->Write();
  }
}

DEFINE_FWK_MODULE(PixelVertexTest)
