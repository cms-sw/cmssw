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
#include <cmath>
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
  // How noisy should I be
  int verbose_;
  // Tree of simple vars for testing resolution eff etc
  TTree *t_;
  TFile *f_;
  int ntrk_;
  static const int maxtrk_=1000;
  double pt_[maxtrk_];
  double z0_[maxtrk_];
  double errz0_[maxtrk_];
  double tanl_[maxtrk_];
  int nvtx_;
  static const int maxvtx_=15;
  double vz_[maxvtx_];
  double vzwt_[maxvtx_];
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
  // How noisy?
  verbose_ = conf_.getUntrackedParameter<unsigned int>("Verbosity",0);

  // Make my little tree
  std::string file = conf_.getUntrackedParameter<std::string>("OutputTree","mytree.root");
  const char* cwd= gDirectory->GetPath();
  f_ = new TFile(file.c_str(),"RECREATE");
  t_ = new TTree("t","Pixel Vertex Testing");
  t_->Branch("nvtx",&nvtx_,"nvtx/I");
  t_->Branch("vz",vz_,"vz[nvtx]/D");
  t_->Branch("vzwt",vzwt_,"vzwt[nvtx]/D");
  t_->Branch("ntrk",&ntrk_,"ntrk/I");
  t_->Branch("pt",pt_,"pt[ntrk]/D");
  t_->Branch("z0",z0_,"z0[ntrk]/D");
  t_->Branch("errz0",errz0_,"errz0[ntrk]/D");
  t_->Branch("tanl",tanl_,"tanl[ntrk]/D");
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
  if (verbose_ > 0) cout << "Reconstructed "<< tracks.size() << " tracks" << std::endl;
  ntrk_=0;
  for (unsigned int i=0; i<tracks.size(); i++) {
    if (verbose_ > 0) {
      cout << "\tmomentum: " << tracks[i].momentum()
	   << "\tPT: " << tracks[i].pt()<< endl;
      cout << "\tvertex: " << tracks[i].vertex()
	   << "\tZ0: " << tracks[i].dz() << " +- " << tracks[i].covariance().dzError() << endl;
      cout << "\tcharge: " << tracks[i].charge()<< endl;
    }
    trks.push_back( reco::TrackRef(trackCollection, i) );
    // Fill ntuple vars
    if (ntrk_ < maxtrk_) {
      pt_[ntrk_] = tracks[i].pt();
      z0_[ntrk_] = tracks[i].dz();
      //      errz0_[ntrk_] = std::sqrt( tracks[i].covariance(3,3) );
      errz0_[ntrk_] = tracks[i].covariance().dzError();
      tanl_[ntrk_] = tracks[i].tanDip();
      ntrk_++;
    }
    if (verbose_ > 0) cout <<"------------------------------------------------"<<endl;
  }
  PVPositionBuilder pos;  
  nvtx_ = 0;
  vz_[nvtx_] = pos.average(trks).value();
  vzwt_[nvtx_] = pos.wtAverage(trks).value();
  nvtx_++;
  if (verbose_ > 0) {
    std::cout << "The average z-position of these tracks is " << vz_[0] << std::endl;
    std::cout << "The weighted average z-position of these tracks is " << vzwt_[0] << std::endl;
  }
  // Finally, fill the tree with the above values
  t_->Fill();
}

void PixelVertexTest::endJob() {
  if (t_) t_->Print();
  if (f_) {
    f_->Print();
    f_->Write();
  }
}

DEFINE_FWK_MODULE(PixelVertexTest)
