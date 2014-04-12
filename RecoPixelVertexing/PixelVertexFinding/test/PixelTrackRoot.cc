
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
//#include "DataFormats/TrackReco/interface/print.h"

#include "./PixelTrackRoot.h"
using namespace std;

class PixelTrackRoot : public edm::EDAnalyzer {
public:
  explicit PixelTrackRoot(const edm::ParameterSet& conf);
  ~PixelTrackRoot();
  virtual void beginJob() { }
  virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
  virtual void endJob() { }
  void book();
  void store();
private:
  TFile *rootfile;
  TTree *tthtree;
  int Event;
  static const int numMaxTrks=100;
  int CMNumTrk;
  float CMTrkVtx[numMaxTrks];
  float CMTrkPt[numMaxTrks];
  float CMTrkP[numMaxTrks];
  float CMTrkIP[numMaxTrks];
  int HPNumTrk;
  float HPTrkVtx[numMaxTrks];
  float HPTrkPt[numMaxTrks];
  float HPTrkP[numMaxTrks];
  float HPTrkIP[numMaxTrks];
  int SiNumTrk;
  float SiTrkVtx[numMaxTrks];
  float SiTrkPt[numMaxTrks];
  float SiTrkP[numMaxTrks];
  float SiTrkIP[numMaxTrks];
};

PixelTrackRoot::PixelTrackRoot(const edm::ParameterSet& conf)
{
  rootfile = new TFile("pixel_parameters.root","RECREATE");
  tthtree = new TTree("T","pixTracTestCP");
  book();
  edm::LogInfo("PixelTrackRoot")<<" CTOR";
}

PixelTrackRoot::~PixelTrackRoot()
{
  rootfile->cd();
  tthtree->Write();
  rootfile->Close();
  delete rootfile;
  edm::LogInfo("PixelTrackRoot")<<" DTOR";
}

void PixelTrackRoot::analyze(
    const edm::Event& ev, const edm::EventSetup& es)
{ 
  typedef reco::TrackCollection::const_iterator IT;
  edm::Handle<reco::TrackCollection> trackCollection1;
  ev.getByLabel("tracks1",trackCollection1);
  const reco::TrackCollection tracks1 = *(trackCollection1.product());
  CMNumTrk= tracks1.size();
  Event=ev.id().event();
  int i =0;
  for (IT it=tracks1.begin(); it!=tracks1.end(); it++){
// myfillCM(*it);
        CMTrkP[i]=it->p();
        CMTrkVtx[i]=10*(it->vertex().z());
        CMTrkPt[i]=it->pt();
        CMTrkIP[i]=it->dz();
        i++;
  }
  i=0;
   edm::Handle<reco::TrackCollection> trackCollection2;
   ev.getByLabel("tracks2",trackCollection2);
   const reco::TrackCollection tracks2 = *(trackCollection2.product());
  HPNumTrk = tracks2.size();
  for (IT it=tracks2.begin(); it!=tracks2.end(); it++) {
//myfillHP(*it);
//   store(); 
        HPTrkP[i]=it->p();
        HPTrkVtx[i]=10*(it->vertex().z());
        HPTrkPt[i]=it->pt();
        HPTrkIP[i]=it->dz();
    	i++;
	}
  i=0;
  edm::Handle<reco::TrackCollection> silTracks;
  ev.getByLabel("trackp",silTracks);
   const reco::TrackCollection SiliconTrks = *(silTracks.product());
//  std::cout << "Silicon Tracks Size: "<< SiliconTrks.size()<<std::endl;
  SiNumTrk = SiliconTrks.size();
  for (IT it=SiliconTrks.begin(); it!=SiliconTrks.end(); it++) {
//myfillHP(*it);
//   store();
        SiTrkP[i]=it->p();
        SiTrkVtx[i]=10*(it->vertex().z());
        SiTrkPt[i]=it->pt();
        SiTrkIP[i]=it->dz();
        i++;
        }

store();
}
void PixelTrackRoot::book()
{
  tthtree->Branch("Event",&Event,"Event/I");
  tthtree->Branch("CMNumTracks", &CMNumTrk,"CMNumTrk/I");
  tthtree->Branch("CMTrackVtx", &CMTrkVtx,"CMTrkVtx[CMNumTrk]/F");
  tthtree->Branch("CMTrkPT", &CMTrkPt,"CMTrkPt[CMNumTrk]/F");
  tthtree->Branch("CMTrkMomentum", &CMTrkP,"CMTrkP[CMNumTrk]/F");
  tthtree->Branch("CMTrkImpactParam",&CMTrkIP,"CMTrkIP[CMNumTrk]/F");
  tthtree->Branch("HPNumTracks", &HPNumTrk,"HPNumTrk/I");
  tthtree->Branch("HPTrackVtx", &HPTrkVtx,"HPTrkVtx[HPNumTrk]/F");
  tthtree->Branch("HPTrkPT", &HPTrkPt,"HPTrkPt[HPNumTrk]/F");
  tthtree->Branch("HPTrkMomentum", &HPTrkP,"HPTrkP[HPNumTrk]/F");
  tthtree->Branch("TrkImpactParam",&HPTrkIP,"HPTrkIP[HPNumTrk]/F");
   tthtree->Branch("SiNumTracks", &SiNumTrk,"SiNumTrk/I");
  tthtree->Branch("SiTrackVtx", &SiTrkVtx,"SiTrkVtx[SiNumTrk]/F");
  tthtree->Branch("SiTrkPT", &SiTrkPt,"SiTrkPt[SiNumTrk]/F");
  tthtree->Branch("SiTrkMomentum", &SiTrkP,"SiTrkP[SiNumTrk]/F");
  tthtree->Branch("SiTrkImpactParam",&SiTrkIP,"SiTrkIP[SiNumTrk]/F");
  
}

void PixelTrackRoot::store(){
tthtree->Fill();
}
 
DEFINE_FWK_MODULE(PixelTrackRoot);
