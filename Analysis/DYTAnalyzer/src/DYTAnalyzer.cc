#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/DYTInfo.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace std;

class DYTAnalyzer : public edm::EDAnalyzer {
public:
  explicit DYTAnalyzer(const edm::ParameterSet&);
  ~DYTAnalyzer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  int          matchedGen(double muonEta, double muonPhi);  
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  string       intToeta(int);
  int          etaToint(double);

  edm::Handle<reco::GenParticleCollection> genParticles;
  typedef edm::ValueMap<reco::DYTInfo> DYTestimators;

  TH1F *NStnUsed_B, *NStnUsed_E, *NStnUsed;
  TH1F *GLBNStnUsed_B, *GLBNStnUsed_E, *GLBNStnUsed;
  TH1F *pTDYT_B , *pTDYT_E_08_12, *pTDYT_E_12_16, *pTDYT_E_16_20, *pTDYT_E_20, *pTDYT;
  TH1F *pTPCK_B , *pTPCK_E_08_12, *pTPCK_E_12_16, *pTPCK_E_16_20, *pTPCK_E_20, *pTPCK;
  TH1F *pTCOK_B , *pTCOK_E, *pTCOK;


  
};


DYTAnalyzer::DYTAnalyzer(const edm::ParameterSet& iConfig)
{}


DYTAnalyzer::~DYTAnalyzer()
{}


string DYTAnalyzer::intToeta(int index)
{
  if (index == 0) return "00_02";
  if (index == 1) return "02_04";
  if (index == 2) return "04_06";
  if (index == 3) return "06_08";
  if (index == 4) return "08_10";
  if (index == 5) return "10_12";
  if (index == 6) return "12_14";
  if (index == 7) return "14_16";
  if (index == 8) return "16_18";
  if (index == 9) return "18_20";
  if (index == 10) return "20_22";
  if (index == 11) return "22_24";
  return "not valid";
}


int DYTAnalyzer::etaToint(double Eta)
{
  if (fabs(Eta) >= 0.0 && fabs(Eta) < 0.2) return 0;
  if (fabs(Eta) >= 0.2 && fabs(Eta) < 0.4) return 1;
  if (fabs(Eta) >= 0.4 && fabs(Eta) < 0.6) return 2;
  if (fabs(Eta) >= 0.6 && fabs(Eta) < 0.8) return 3;
  if (fabs(Eta) >= 0.8 && fabs(Eta) < 1.0) return 4;
  if (fabs(Eta) >= 1.0 && fabs(Eta) < 1.2) return 5;
  if (fabs(Eta) >= 1.2 && fabs(Eta) < 1.4) return 6;
  if (fabs(Eta) >= 1.4 && fabs(Eta) < 1.6) return 7;
  if (fabs(Eta) >= 1.6 && fabs(Eta) < 1.8) return 8;
  if (fabs(Eta) >= 1.8 && fabs(Eta) < 2.0) return 9;
  if (fabs(Eta) >= 2.0 && fabs(Eta) < 2.2) return 10;
  if (fabs(Eta) >= 2.2) return 11;
  return -1;
}


int DYTAnalyzer::matchedGen(double muonEta, double muonPhi)
{
  double maxDR = 0.1;
  int index    = -1;
  int i        = 0;
  for (reco::GenParticleCollection::const_iterator mcIter=genParticles->begin(); mcIter != genParticles->end(); mcIter++ ) {
    if (fabs(mcIter->pdgId()) != 13) continue;
    if (deltaR(mcIter->eta(), mcIter->phi(), muonEta, muonPhi) < maxDR) {
      maxDR = deltaR(mcIter->eta(), mcIter->phi(), muonEta, muonPhi);
      index = i;
    }
    i++;
  }
  return index;
}


void DYTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;

  try{iEvent.getByLabel("genParticles", genParticles);} catch (...) {return;}
  
  Handle<DYTestimators> dytInfoH;
  try {iEvent.getByLabel("tevMuons", "dytInfo", dytInfoH);} catch (...) {return;}
  const DYTestimators &dytInfoC = *dytInfoH; 
  
  Handle<TrackToTrackMap> pmrMapH_4;
  try {iEvent.getByLabel("tevMuons","dyt",pmrMapH_4);} catch (...) {return;}
  const TrackToTrackMap pmrMap_4 =*(pmrMapH_4.product());

  Handle<MuonCollection> muons;
  try {iEvent.getByLabel("muons", muons);} catch (...) {return;}
  for(size_t i = 0; i != muons->size(); ++i) {
    try {
      DYTInfo dytInfo = dytInfoC[muons->at(i).globalTrack()];
      vector<double> estimators = dytInfo.DYTEstimators();
      //for (unsigned int i = 0; i < estimators.size(); i++)
      //cout << estimators[i] << endl;
	//vector<bool> usedSt = dytInfo.UsedStations();

      if (fabs(muons->at(i).eta()) >= 2.4) continue;
      //      if (!muons->at(i).isGlobalMuon()) continue;
      //      if (!muons->at(i).isPFMuon()) continue;
      //      if (muons->at(i).innerTrack()->hitPattern().trackerLayersWithMeasurement() <= 5) continue;
      //      if (muons->at(i).innerTrack()->hitPattern().numberOfValidPixelHits() < 1) continue;


      ///////////////////////////// 
      // Number of used stations //
      ///////////////////////////// 
      int nGLBStUs = muons->at(i).numberOfMatchedStations();
      int nDYTStUs = dytInfo.NStUsed();
      GLBNStnUsed->Fill(nGLBStUs);
      NStnUsed->Fill(nDYTStUs);
      if (fabs(muons->at(i).eta()) < 0.8) {GLBNStnUsed_B->Fill(nGLBStUs); NStnUsed_B->Fill(nDYTStUs);}
      else {GLBNStnUsed_E->Fill(nGLBStUs); NStnUsed_E->Fill(nDYTStUs);}
      /////////////////////////////
      ///////////////////////////// 
      ///////////////////////////// 


      TrackToTrackMap::const_iterator pmrTrack_4 = pmrMap_4.find(muons->at(i).globalTrack());
      TrackRef DYT = (*pmrTrack_4).val;
      pTDYT->Fill(DYT->pt());
      if (fabs(muons->at(i).eta()) <= 0.8) pTDYT_B->Fill(DYT->pt());
      if (fabs(muons->at(i).eta()) > 0.8 && fabs(muons->at(i).eta()) <= 1.2) pTDYT_E_08_12->Fill(DYT->pt());
      if (fabs(muons->at(i).eta()) > 1.2 && fabs(muons->at(i).eta()) <= 1.6) pTDYT_E_12_16->Fill(DYT->pt());
      if (fabs(muons->at(i).eta()) > 1.6 && fabs(muons->at(i).eta()) <= 2.0) pTDYT_E_16_20->Fill(DYT->pt());
      if (fabs(muons->at(i).eta()) > 2.0) pTDYT_E_20->Fill(DYT->pt());
      TrackRef PCK = muons->at(i).pickyTrack();
      pTPCK->Fill(PCK->pt());
      if (fabs(muons->at(i).eta()) <= 0.8) pTPCK_B->Fill(PCK->pt());
      if (fabs(muons->at(i).eta()) > 0.8 && fabs(muons->at(i).eta()) <= 1.2) pTPCK_E_08_12->Fill(PCK->pt());
      if (fabs(muons->at(i).eta()) > 1.2 && fabs(muons->at(i).eta()) <= 1.6) pTPCK_E_12_16->Fill(PCK->pt());
      if (fabs(muons->at(i).eta()) > 1.6 && fabs(muons->at(i).eta()) <= 2.0) pTPCK_E_16_20->Fill(PCK->pt());
      if (fabs(muons->at(i).eta()) > 2.0) pTPCK_E_20->Fill(PCK->pt());
      Muon COK = muons->at(i);
      pTCOK->Fill(COK.pt());
      if (fabs(muons->at(i).eta()) < 0.8) pTCOK_B->Fill(COK.pt());
      else pTCOK_E->Fill(COK.pt());
    } catch (...) {continue;}
  }
}


void DYTAnalyzer::beginJob()
{
  edm::Service<TFileService> fs;
  pTDYT         = fs->make<TH1F>("pTDYT"  , "pTDYT"  , 20000, 0., 2000);
  pTDYT_B       = fs->make<TH1F>("pTDYT_B", "pTDYT_B", 20000, 0., 2000);
  pTDYT_E_08_12 = fs->make<TH1F>("pTDYT_E_08_12", "pTDYT_E_08_12", 20000, 0., 2000);
  pTDYT_E_12_16 = fs->make<TH1F>("pTDYT_E_12_16", "pTDYT_E_12_16", 20000, 0., 2000);
  pTDYT_E_16_20 = fs->make<TH1F>("pTDYT_E_16_20", "pTDYT_E_16_20", 20000, 0., 2000);
  pTDYT_E_20    = fs->make<TH1F>("pTDYT_E_20", "pTDYT_E_20", 20000, 0., 2000);
  pTPCK_E_08_12 = fs->make<TH1F>("pTPCK_E_08_12", "pTPCK_E_08_12", 20000, 0., 2000);
  pTPCK_E_12_16 = fs->make<TH1F>("pTPCK_E_12_16", "pTPCK_E_12_16", 20000, 0., 2000);
  pTPCK_E_16_20 = fs->make<TH1F>("pTPCK_E_16_20", "pTPCK_E_16_20", 20000, 0., 2000);
  pTPCK_E_20    = fs->make<TH1F>("pTPCK_E_20", "pTPCK_E_20", 20000, 0., 2000);
  pTPCK         = fs->make<TH1F>("pTPCK"  , "pTPCK"  , 20000, 0., 2000);
  pTPCK_B       = fs->make<TH1F>("pTPCK_B", "pTPCK_B", 20000, 0., 2000);
  pTCOK         = fs->make<TH1F>("pTCOK"  , "pTCOK"  , 20000, 0., 2000);
  pTCOK_B       = fs->make<TH1F>("pTCOK_B", "pTCOK_B", 20000, 0., 2000);
  pTCOK_E       = fs->make<TH1F>("pTCOK_E", "pTCOK_E", 20000, 0., 2000);
  NStnUsed      = fs->make<TH1F>("NStnUsed"  , "NStnUsed"  , 5, 0., 5);
  NStnUsed_B    = fs->make<TH1F>("NStnUsed_B", "NStnUsed_B", 5, 0., 5);
  NStnUsed_E    = fs->make<TH1F>("NStnUsed_E", "NStnUsed_E", 5, 0., 5);
  GLBNStnUsed   = fs->make<TH1F>("GLBNStnUsed"  , "NStnUsed"  , 5, 0., 5);
  GLBNStnUsed_B = fs->make<TH1F>("GLBNStnUsed_B", "NStnUsed_B", 5, 0., 5);
  GLBNStnUsed_E = fs->make<TH1F>("GLBNStnUsed_E", "NStnUsed_E", 5, 0., 5);
}


void DYTAnalyzer::endJob() 
{}


void DYTAnalyzer::beginRun(edm::Run const&, edm::EventSetup const& es)
{}


void DYTAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{}


void DYTAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{}


void DYTAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{}

void DYTAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
DEFINE_FWK_MODULE(DYTAnalyzer);
