// $Id: QcdLowPtDQM.cc,v 1.20 2013/01/02 13:59:43 wmtan Exp $

#include "DQM/Physics/src/QcdLowPtDQM.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include <TString.h>
#include <TMath.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>

using namespace std;
using namespace edm;

#define CP(level) \
  if (level>=verbose_)

struct deleter {
  void operator()(TH3F *&h) { delete h; h=0;}
};

//--------------------------------------------------------------------------------------------------
bool compareTracklets(const QcdLowPtDQM::Tracklet &a, const QcdLowPtDQM::Tracklet &b) 
{
  return (TMath::Abs(a.deta())<TMath::Abs(b.deta()));
}

//--------------------------------------------------------------------------------------------------
QcdLowPtDQM::QcdLowPtDQM(const ParameterSet &parameters) :
  hltResName_(parameters.getUntrackedParameter<string>("hltTrgResults","TriggerResults")),
  pixelName_(parameters.getUntrackedParameter<string>("pixelRecHits","siPixelRecHits")),
  clusterVtxName_(parameters.getUntrackedParameter<string>("clusterVertices","")),
  ZVCut_(parameters.getUntrackedParameter<double>("ZVertexCut",10)),
  ZVEtaRegion_(parameters.getUntrackedParameter<double>("ZVertexEtaRegion",2)),
  ZVVtxRegion_(parameters.getUntrackedParameter<double>("ZVertexVtxRegion",2)),
  dPhiVc_(parameters.getUntrackedParameter<double>("dPhiVertexCut",0.08)),
  dZVc_(parameters.getUntrackedParameter<double>("dZVertexCut",0.25)),
  sigEtaCut_(parameters.getUntrackedParameter<double>("signalEtaCut",0.1)),
  sigPhiCut_(parameters.getUntrackedParameter<double>("signalPhiCut",1.5)),
  bkgEtaCut_(parameters.getUntrackedParameter<double>("backgroundEtaCut",0.1)),
  bkgPhiCut_(parameters.getUntrackedParameter<double>("backgroundPhiCut",3.0)),
  verbose_(parameters.getUntrackedParameter<int>("verbose",3)),
  pixLayers_(parameters.getUntrackedParameter<int>("pixLayerCombinations",12)),
  clusLayers_(parameters.getUntrackedParameter<int>("clusLayerCombinations",12)),
  useRecHitQ_(parameters.getUntrackedParameter<bool>("useRecHitQualityWord",false)),
  usePixelQ_(parameters.getUntrackedParameter<bool>("usePixelQualityWord",true)),
  AlphaTracklets12_(0),
  AlphaTracklets13_(0),
  AlphaTracklets23_(0),
  tgeo_(0),
  theDbe_(0),
  repSumMap_(0),
  repSummary_(0),
  h2TrigCorr_(0)
{
  // Constructor.

  if (parameters.exists("hltTrgNames"))
    hltTrgNames_ = parameters.getUntrackedParameter<vector<string> >("hltTrgNames");

  if (parameters.exists("hltProcNames"))
     hltProcNames_ = parameters.getUntrackedParameter<vector<string> >("hltProcNames");
  else {
     hltProcNames_.push_back("FU");
     hltProcNames_.push_back("HLT");
  }

  if ((pixLayers_!=12) && (pixLayers_!=13) && (pixLayers_!=23)) {
    print(2,Form("Value for pixLayerCombinations must be one of 12,13, or 23. "
                 "Got %d, set value to 12", pixLayers_));
    pixLayers_ = 12;
  }
}

//--------------------------------------------------------------------------------------------------
QcdLowPtDQM::~QcdLowPtDQM()
{
  // Destructor.

  std::for_each(NsigTracklets12_.begin(),  NsigTracklets12_.end(),  deleter());
  std::for_each(NbkgTracklets12_.begin(),  NbkgTracklets12_.end(),  deleter());
  deleter()(AlphaTracklets12_);
  std::for_each(NsigTracklets13_.begin(),  NsigTracklets13_.end(),  deleter());
  std::for_each(NbkgTracklets13_.begin(),  NbkgTracklets13_.end(),  deleter());
  deleter()(AlphaTracklets13_);
  std::for_each(NsigTracklets23_.begin(),  NsigTracklets23_.end(),  deleter());
  std::for_each(NbkgTracklets23_.begin(),  NbkgTracklets23_.end(),  deleter());
  deleter()(AlphaTracklets23_);
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::analyze(const Event &iEvent, const EventSetup &iSetup) 
{
  // Analyze the given event.

  // get tracker geometry
  ESHandle<TrackerGeometry> trackerHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  tgeo_ = trackerHandle.product();
  if (!tgeo_) {
    print(3,"QcdLowPtDQM::analyze -- Could not obtain pointer to TrackerGeometry. Return.");
    return;
  }

  fillHltBits(iEvent);
  fillPixels(iEvent, iSetup);
  trackletVertexUnbinned(iEvent, pixLayers_);
  fillTracklets(iEvent, pixLayers_);
  fillPixelClusterInfos(iEvent, clusLayers_);
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::beginJob() 
{
  // Begin job and setup the DQM store.

  theDbe_ = Service<DQMStore>().operator->();
  if (!theDbe_) {
    print(3,"QcdLowPtDQM::beginJob -- Could not obtain pointer to DQMStore. Return.");
    return;
  }
  theDbe_->setCurrentFolder("Physics/QcdLowPt");
  yieldAlphaHistogram(pixLayers_);
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::beginLuminosityBlock(const LuminosityBlock &l, 
                                       const EventSetup &iSetup)
{
  // At the moment, nothing needed to be done.
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::beginRun(const Run &run, const EventSetup &iSetup)
{
  // Begin run, get or create needed structures.  TODO: can this be called several times in DQM??? -> YES!

  bool isinit = false;
  bool isHltCfgChanged = false; // for new HLTConfigProvider::init
  string teststr;
  for(size_t i=0; i<hltProcNames_.size(); ++i) {
    if (i>0) 
      teststr += ", ";
    teststr += hltProcNames_.at(i);
    if( hltConfig_.init(  run, iSetup, hltProcNames_.at(i), isHltCfgChanged )  ) {
      isinit = true;
      hltUsedResName_ = hltResName_;
      if (hltResName_.find(':')==string::npos)
        hltUsedResName_ += "::";
      else 
        hltUsedResName_ += ":";
      hltUsedResName_ += hltProcNames_.at(i);
      break;
    }
  }

  if (!isinit)
    print(3,Form("Could not obtain HLT config for process name(s) %s", teststr.c_str()));

  // setup "Any" bit
  hltTrgBits_.clear();
  hltTrgBits_.push_back(-1);
  hltTrgDeci_.clear();
  hltTrgDeci_.push_back(true);
  hltTrgUsedNames_.clear();
  hltTrgUsedNames_.push_back("Any");

  // figure out relation of trigger name to trigger bit and store used trigger names/bits
  for(size_t i=0;i<hltTrgNames_.size();++i) {
    const string &n1(hltTrgNames_.at(i));
    bool found = 0;
    for(size_t j=0;j<hltConfig_.size();++j) {
      const string &n2(hltConfig_.triggerName(j));
      if(0) print(0,Form("Checking trigger name %s for %s", n2.c_str(), n1.c_str()));
      if (n2==n1) {
        hltTrgBits_.push_back(j);
        hltTrgUsedNames_.push_back(n1);
        hltTrgDeci_.push_back(false);
        print(0,Form("Added trigger %d with name %s for bit %d", 
                     int(hltTrgBits_.size()-1), n1.c_str(), int(j)));
        found = 1;
        break;
      }
    }      
    if (!found) {
      CP(2) print(2,Form("Could not find trigger bit for %s", n1.c_str()));
    }
  }

  // ensure that trigger collections are of same size
  if (hltTrgBits_.size()!=hltTrgUsedNames_.size())
    print(3,Form("Size of trigger bits not equal used names: %d %d",
                 int(hltTrgBits_.size()), int(hltTrgUsedNames_.size())));
  if (hltTrgDeci_.size()!=hltTrgUsedNames_.size())
    print(3,Form("Size of decision bits not equal names: %d %d",
                 int(hltTrgDeci_.size()), int(hltTrgUsedNames_.size())));

  // setup correction histograms
  if (AlphaTracklets12_) {
    for(size_t i=0;i<hltTrgUsedNames_.size();++i) {
      TH3F *h2 = (TH3F*)AlphaTracklets12_->Clone(Form("NsigTracklets12-%s",
                                                      hltTrgUsedNames_.at(i).c_str()));
      NsigTracklets12_.push_back(h2);
      TH3F *h3 = (TH3F*)AlphaTracklets12_->Clone(Form("NbkgTracklets12-%s",
                                                      hltTrgUsedNames_.at(i).c_str()));
      NbkgTracklets12_.push_back(h3);
    }
  }
  if (AlphaTracklets13_) {
    for(size_t i=0;i<hltTrgUsedNames_.size();++i) {
      TH3F *h2 = (TH3F*)AlphaTracklets13_->Clone(Form("NsigTracklets13-%s",
                                                      hltTrgUsedNames_.at(i).c_str()));
      NsigTracklets13_.push_back(h2);
      TH3F *h3 = (TH3F*)AlphaTracklets13_->Clone(Form("NbkgTracklets13-%s",
                                                      hltTrgUsedNames_.at(i).c_str()));
      NbkgTracklets13_.push_back(h3);
    }
  }
  if (AlphaTracklets23_) {
    for(size_t i=0;i<hltTrgUsedNames_.size();++i) {
      TH3F *h2 = (TH3F*)AlphaTracklets23_->Clone(Form("NsigTracklets23-%s",
                                                      hltTrgUsedNames_.at(i).c_str()));
      NsigTracklets23_.push_back(h2);
      TH3F *h3 = (TH3F*)AlphaTracklets23_->Clone(Form("NbkgTracklets23-%s",
                                                      hltTrgUsedNames_.at(i).c_str()));
      NbkgTracklets23_.push_back(h3);
    }
  }

  // book monitoring histograms
  createHistos();
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::book1D(std::vector<MonitorElement*> &mes, 
                         const std::string &name, const std::string &title, 
                         int nx, double x1, double x2, bool sumw2, bool sbox)
{
  // Book 1D histos.

  for(size_t i=0;i<hltTrgUsedNames_.size();++i) {
    MonitorElement *e = theDbe_->book1D(Form("%s_%s",name.c_str(),hltTrgUsedNames_.at(i).c_str()),
                                        Form("%s: %s",hltTrgUsedNames_.at(i).c_str(), title.c_str()), 
                                        nx, x1, x2);
    TH1 *h1 = e->getTH1();
    if (sumw2)
      h1->Sumw2();
    h1->SetStats(sbox);
    mes.push_back(e);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::book2D(std::vector<MonitorElement*> &mes, 
                         const std::string &name, const std::string &title, 
                         int nx, double x1, double x2, int ny, double y1, double y2, 
                         bool sumw2, bool sbox)
{
  // Book 2D histos.

  for(size_t i=0;i<hltTrgUsedNames_.size();++i) {
    MonitorElement *e = theDbe_->book2D(Form("%s_%s",name.c_str(),hltTrgUsedNames_.at(i).c_str()),
                                        Form("%s: %s",hltTrgUsedNames_.at(i).c_str(), title.c_str()), 
                                        nx, x1, x2, ny, y1, y2);
    TH1 *h1 = e->getTH1();
    if (sumw2)
      h1->Sumw2();
    h1->SetStats(sbox);
    mes.push_back(e);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::create1D(std::vector<TH1F*> &hs, 
                         const std::string &name, const std::string &title, 
                         int nx, double x1, double x2, bool sumw2, bool sbox)
{
  // Create 1D histos.

  for(size_t i=0;i<hltTrgUsedNames_.size();++i) {
    TH1F *h1 = new TH1F(Form("%s_%s",name.c_str(),hltTrgUsedNames_.at(i).c_str()),
                        Form("%s: %s",hltTrgUsedNames_.at(i).c_str(), title.c_str()), 
                        nx, x1, x2);
    if (sumw2)
      h1->Sumw2();
    h1->SetStats(sbox);
    hs.push_back(h1);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::create2D(std::vector<TH2F*> &hs, 
                           const std::string &name, const std::string &title, 
                           int nx, double x1, double x2, int ny, double y1, double y2, 
                           bool sumw2, bool sbox)
{
  // Create 2D histos.

  for(size_t i=0;i<hltTrgUsedNames_.size();++i) {
    TH2F *h1 = new TH2F(Form("%s_%s",name.c_str(),hltTrgUsedNames_.at(i).c_str()),
                        Form("%s: %s",hltTrgUsedNames_.at(i).c_str(), title.c_str()), 
                        nx, x1, x2, ny, y1, y2);
    if (sumw2)
      h1->Sumw2();
    h1->SetStats(sbox);
    hs.push_back(h1);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::createHistos()
{
  // Book histograms if needed.
  
  if (hNhitsL1_.size())
    return; // histograms already booked

  if (1) {
    theDbe_->setCurrentFolder("Physics/EventInfo/");
    repSumMap_  = theDbe_->book2D("reportSummaryMap","reportSummaryMap",1,0,1,1,0,1);
    repSummary_ = theDbe_->bookFloat("reportSummary");
  }

  if (1) {
    theDbe_->setCurrentFolder("Physics/QcdLowPt/");
    const int Nx = hltTrgUsedNames_.size();
    const double x1 = -0.5;
    const double x2 = Nx-0.5;
    h2TrigCorr_ = theDbe_->book2D("h2TriCorr","Trigger bit x vs y (y&&!x,x&&y)",Nx,x1,x2,Nx,x1,x2);
    for(size_t i=1;i<=hltTrgUsedNames_.size();++i) {
      h2TrigCorr_->setBinLabel(i,hltTrgUsedNames_.at(i-1),1);
      h2TrigCorr_->setBinLabel(i,hltTrgUsedNames_.at(i-1),2);
    }
    TH1 *h = h2TrigCorr_->getTH1();
    if (h)
      h->SetStats(0);
  }
  if (1) {
    theDbe_->setCurrentFolder("Physics/QcdLowPt/HitsLayer/");
    const int Nx = 30;
    const double x1 = -0.5;
    const double x2 = 149.5;
    book1D(hNhitsL1_,"hNhitsLayer1","number of hits on layer 1;#hits;#",Nx,x1,x2);
    if(0) book1D(hNhitsL2_,"hNhitsLayer2","number of hits on layer 2;#hits;#",Nx,x1,x2);
    if(0) book1D(hNhitsL3_,"hNhitsLayer3","number of hits on layer 3;#hits;#",Nx,x1,x2);
  }
  if (1) {
    theDbe_->setCurrentFolder("Physics/QcdLowPt/HitsLayerZoom/");
    const int Nx = 15;
    const double x1 = -0.5;
    const double x2 = 14.5;
    book1D(hNhitsL1z_,"hNhitsLayer1Zoom","number of hits on layer 1;#hits;#",Nx,x1,x2);
    if(0) book1D(hNhitsL2z_,"hNhitsLayer2Zoom","number of hits on layer 2;#hits;#",Nx,x1,x2);
    if(0) book1D(hNhitsL3z_,"hNhitsLayer3Zoom","number of hits on layer 3;#hits;#",Nx,x1,x2);
  }
  if (1) {
    theDbe_->setCurrentFolder("Physics/QcdLowPt/HitsLayerEta/");
    const int Nx = 60;
    const double x1 = -3;
    const double x2 = +3;
    book1D(hdNdEtaHitsL1_,"hdNdEtaHitsLayer1","Hits on layer 1;detector #eta;#",Nx,x1,x2);
    if(0) book1D(hdNdEtaHitsL2_,"hdNdEtaHitsLayer2","Hits on layer 2;detector #eta;#",Nx,x1,x2);
    if(0) book1D(hdNdEtaHitsL3_,"hdNdEtaHitsLayer3","Hits on layer 3;detector #eta;#",Nx,x1,x2);
  }
  if (1) {
    theDbe_->setCurrentFolder("Physics/QcdLowPt/HitsLayerPhi/");
    const int Nx = 64;
    const double x1 = -3.2;
    const double x2 = +3.2;
    book1D(hdNdPhiHitsL1_,"hdNdPhiHitsLayer1","Hits on layer 1;#phi;#",Nx,x1,x2);
    if(0) book1D(hdNdPhiHitsL2_,"hdNdPhiHitsLayer2","Hits on layer 2;#phi;#",Nx,x1,x2);
    if(0) book1D(hdNdPhiHitsL3_,"hdNdPhiHitsLayer3","Hits on layer 3;#phi;#",Nx,x1,x2);
  }
  if (1) {
    theDbe_->setCurrentFolder("Physics/QcdLowPt/TrackletVtxZ/");
    const int Nx = 100;
    const double x1 = -25;
    const double x2 = +25;
    if (pixLayers_>=12)
      book1D(hTrkVtxZ12_,"hTrackletVtxZ12","z vertex from tracklets12;vz [cm];#",Nx,x1,x2);
    if (pixLayers_>=13)
      book1D(hTrkVtxZ13_,"hTrackletVtxZ13","z vertex from tracklets13;vz [cm];#",Nx,x1,x2);
    if (pixLayers_>=23)
      book1D(hTrkVtxZ23_,"hTrackletVtxZ23","z vertex from tracklets23;vz [cm];#",Nx,x1,x2);
  }

  if (1) {
    theDbe_->setCurrentFolder("Physics/QcdLowPt/TrackletEtaVtxZ/");
    const int Nx = 60;
    const double x1 = -3;
    const double x2 = +3;
    const int Ny = 2*(int)ZVCut_;
    const double y1 = -ZVCut_;
    const double y2 = +ZVCut_;
    if (pixLayers_>=12)
      book2D(hRawTrkEtaVtxZ12_,"hRawTrkEtaVtxZ12",
             "raw #eta vs z vertex from tracklets12;#eta;vz [cm]",Nx,x1,x2,Ny,y1,y2,0,0);
    if (pixLayers_>=13)
      book2D(hRawTrkEtaVtxZ13_,"hRawTrkEtaVtxZ13",
             "raw #eta vs z vertex from tracklets13;#eta;vz [cm]",Nx,x1,x2,Ny,y1,y2,0,0);
    if (pixLayers_>=23)
      book2D(hRawTrkEtaVtxZ23_,"hRawTrkEtaVtxZ23",
             "raw #eta vs z vertex from tracklets23;#eta;vz [cm]",Nx,x1,x2,Ny,y1,y2,0,0);
  }
  if (0) {
    theDbe_->setCurrentFolder("Physics/QcdLowPt/TrackletDetaDphi/");
    const int Nx = 60;
    const double x1 = -3;
    const double x2 = +3;
    const int Ny = 64;
    const double y1 = -3.2;
    const double y2 = +3.2;
    if (pixLayers_>=12)
      book2D(hTrkRawDetaDphi12_,"hTracklet12RawDetaDphi",
             "tracklet12 raw #Delta#eta vs #Delta#phi;#Delta#eta;#Delta#phi",Nx,x1,x2,Ny,y1,y2,0,0);
    if (pixLayers_>=13)
      book2D(hTrkRawDetaDphi13_,"hTracklet13RawDetaDphi",
             "tracklet13 raw #Delta#eta vs #Delta#phi;#Delta#eta;#Delta#phi",Nx,x1,x2,Ny,y1,y2,0,0);
    if (pixLayers_>=23)
      book2D(hTrkRawDetaDphi23_,"hTracklet23RawDetaDphi",
             "tracklet12 raw #Delta#eta vs #Delta#phi;#Delta#eta;#Delta#phi",Nx,x1,x2,Ny,y1,y2,0,0);
  }
  if (0) {
    theDbe_->setCurrentFolder("Physics/QcdLowPt/TrackletDeta/");
    const int Nx = 60;
    const double x1 = -3;
    const double x2 = +3;
    if (pixLayers_>=12)
      book1D(hTrkRawDeta12_,"hTracklet12RawDeta",
             "tracklet12 raw dN/#Delta#eta;#Delta#eta;dN/#Delta#eta",Nx,x1,x2,0,0); 
    if (pixLayers_>=13)
      book1D(hTrkRawDeta13_,"hTracklet13RawDeta",
             "tracklet13 raw dN/#Delta#eta;#Delta#eta;dN/#Delta#eta",Nx,x1,x2,0,0); 
    if (pixLayers_>=23)
      book1D(hTrkRawDeta23_,"hTracklet23RawDeta",
             "tracklet23 raw dN/#Delta#eta;#Delta#eta;dN/#Delta#eta",Nx,x1,x2,0,0); 
  }
  if (0) {
    theDbe_->setCurrentFolder("Physics/QcdLowPt/TrackletDphi/");
    const int Nx = 64;
    const double x1 = -3.2;
    const double x2 = +3.2;
    if (pixLayers_>=12)
      book1D(hTrkRawDphi12_,"hTracklet12RawDphi",
             "tracklet12 raw dN/#Delta#phi;#Delta#phi;dN/#Delta#phi",Nx,x1,x2,0,0); 
    if (pixLayers_>=13)
      book1D(hTrkRawDphi13_,"hTracklet13RawDphi",
             "tracklet13 raw dN/#Delta#phi;#Delta#phi;dN/#Delta#phi",Nx,x1,x2,0,0); 
    if (pixLayers_>=23)
      book1D(hTrkRawDphi23_,"hTracklet23RawDphi",
             "tracklet23 raw dN/#Delta#phi;#Delta#phi;dN/#Delta#phi",Nx,x1,x2,0,0); 
  }
  if (AlphaTracklets12_) {
    TAxis *xa = AlphaTracklets12_->GetXaxis();
    const int Nx = xa->GetNbins();
    const double x1 = xa->GetBinLowEdge(1);
    const double x2 = xa->GetBinLowEdge(Nx+1);
    theDbe_->setCurrentFolder("Physics/QcdLowPt/RawTracklets/");
    book1D(hdNdEtaRawTrkl12_,"hdNdEtaRawTracklets12",
           "raw dN/d#eta for tracklets12;#eta;dN/d#eta",Nx,x1,x2,0,0); 
    theDbe_->setCurrentFolder("Physics/QcdLowPt/SubTracklets/");
    book1D(hdNdEtaSubTrkl12_,"hdNdEtaSubTracklets12",
           "(1-#beta) dN/d#eta for tracklets12;#eta;dN/d#eta",Nx,x1,x2,0,0); 
    theDbe_->setCurrentFolder("Physics/QcdLowPt/CorTracklets/");
    book1D(hdNdEtaTrklets12_,"hdNdEtaTracklets12",
           "dN/d#eta for tracklets12;#eta;dN/d#eta",Nx,x1,x2,0,0); 
    create1D(hEvtCountsPerEta12_,"hEventCountsPerEta12_",
             "Events per vtx-#eta bin from tracklets12;#eta;#",1,-ZVEtaRegion_,ZVEtaRegion_,0,0);
  }
  if (AlphaTracklets13_) {
    TAxis *xa = AlphaTracklets13_->GetXaxis();
    const int Nx = xa->GetNbins();
    const double x1 = xa->GetBinLowEdge(1);
    const double x2 = xa->GetBinLowEdge(Nx+1);
    theDbe_->setCurrentFolder("Physics/QcdLowPt/RawTracklets/");
    book1D(hdNdEtaRawTrkl13_,"hdNdEtaRawTracklets13",
           "raw dN/d#eta for tracklets13;#eta;dN/d#eta",Nx,x1,x2,0,0); 
    theDbe_->setCurrentFolder("Physics/QcdLowPt/SubTracklets/");
    book1D(hdNdEtaSubTrkl13_,"hdNdEtaSubTracklets13",
           "(1-#beta) dN/d#eta for tracklets13;#eta;dN/d#eta",Nx,x1,x2,0,0); 
    theDbe_->setCurrentFolder("Physics/QcdLowPt/CorTracklets/");
    book1D(hdNdEtaTrklets13_,"hdNdEtaTracklets13",
           "dN/d#eta for tracklets13;#eta;dN/d#eta",Nx,x1,x2,0,0); 
    create1D(hEvtCountsPerEta13_,"hEventCountsPerEta13",
             "Events per vtx-#eta bin from tracklets13;#eta;#",1,-ZVEtaRegion_,ZVEtaRegion_,0,0);
  }
  if (AlphaTracklets23_) {
    TAxis *xa = AlphaTracklets23_->GetXaxis();
    const int Nx = xa->GetNbins();
    const double x1 = xa->GetBinLowEdge(1);
    const double x2 = xa->GetBinLowEdge(Nx+1);
    theDbe_->setCurrentFolder("Physics/QcdLowPt/RawTracklets/");
    book1D(hdNdEtaRawTrkl23_,"hdNdEtaRawTracklets23",
           "raw dN/d#eta for tracklets23;#eta;dN/d#eta",Nx,x1,x2,0,0); 
    theDbe_->setCurrentFolder("Physics/QcdLowPt/SubTracklets/");
    book1D(hdNdEtaSubTrkl23_,"hdNdEtaSubTracklets23",
           "(1-#beta) dN/d#eta for tracklets23;#eta;dN/d#eta",Nx,x1,x2,0,0); 
    theDbe_->setCurrentFolder("Physics/QcdLowPt/CorTracklets/");
    book1D(hdNdEtaTrklets23_,"hdNdEtaTracklets23",
           "dN/d#eta for tracklets23;#eta;dN/d#eta",Nx,x1,x2,0,0); 
    create1D(hEvtCountsPerEta23_,"hEventCountsPerEta23",
             "Events per vtx-#eta bin from tracklets23;#eta;#",1,-ZVEtaRegion_,ZVEtaRegion_,0,0);
  }
  if (1) {
    if (1) {
      const int Nx = 100;
      const double x1 = -25;
      const double x2 = +25;
      theDbe_->setCurrentFolder("Physics/QcdLowPt/ClusterVtxZ/");
      book1D(hClusterVertexZ_,"hClusterVertex","z vertex from clusters12;vz [cm];#",Nx,x1,x2);
    }
    if (1) {
      theDbe_->setCurrentFolder("Physics/QcdLowPt/ClusterSize/");
      const int Nx = 60;
      const double x1 = -3;
      const double x2 = +3;
      const int Ny= 25;
      const double y1 = -0.5;
      const double y2 = 24.5;
      if (clusLayers_>=12)
        book2D(hClusterYSize1_,"hClusterYSize1",
               "cluster #eta vs local y size on layer 1;#eta;size",Nx,x1,x2,Ny,y1,y2,0,0);
      if (clusLayers_>=13)
        book2D(hClusterYSize2_,"hClusterYSize2",
               "cluster #eta vs local y size on layer 2;#eta;size",Nx,x1,x2,Ny,y1,y2,0,0);
      if (clusLayers_>=23)
        book2D(hClusterYSize3_,"hClusterYSize3",
               "cluster #eta vs local y size on layer 3;#eta;size",Nx,x1,x2,Ny,y1,y2,0,0);
    }
    if (1) {
      theDbe_->setCurrentFolder("Physics/QcdLowPt/ClusterCharge/");
      const int Nx = 60;
      const double x1 = -3;
      const double x2 = +3;
      const int Ny= 125;
      const double y1 = 0;
      const double y2 = 2500;
      if (clusLayers_>=12)
        book2D(hClusterADC1_,"hClusterADC1",
               "cluster #eta vs adc on layer 1;#eta;adc",Nx,x1,x2,Ny,y1,y2,0,0);
      if (clusLayers_>=13)
        book2D(hClusterADC2_,"hClusterADC2",
               "cluster #eta vs adc on layer 2;#eta;adc",Nx,x1,x2,Ny,y1,y2,0,0);
      if (clusLayers_>=23)
        book2D(hClusterADC3_,"hClusterADC3",
               "cluster #eta vs adc on layer 3;#eta;adc",Nx,x1,x2,Ny,y1,y2,0,0);
    }
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::endJob(void) 
{
  // At the moment, nothing needed to be done.
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::filldNdeta(const TH3F *AlphaTracklets,
                             const std::vector<TH3F*> &NsigTracklets,
                             const std::vector<TH3F*> &NbkgTracklets,
                             const std::vector<TH1F*> &NEvsPerEta,
                             std::vector<MonitorElement*> &hdNdEtaRawTrkl,
                             std::vector<MonitorElement*> &hdNdEtaSubTrkl,
                             std::vector<MonitorElement*> &hdNdEtaTrklets)
{
  // Fill raw and corrected dNdeta into histograms.

  if (!AlphaTracklets)
    return;

  const int netabins = AlphaTracklets->GetNbinsX();
  const int nhitbins = AlphaTracklets->GetNbinsY();
  const int nvzbins  = AlphaTracklets->GetNbinsZ();
  //  const int zvlbin   = AlphaTracklets->GetZaxis()->FindFixBin(-ZVVtxRegion_)-1; // UNUSED
  //  const int zvhbin   = AlphaTracklets->GetZaxis()->FindFixBin(+ZVVtxRegion_)+1; // UNUSED

  for(size_t i=0;i<hdNdEtaRawTrkl.size();++i) {
    MonitorElement *mrawtrk = hdNdEtaRawTrkl.at(i);
    MonitorElement *msubtrk = hdNdEtaSubTrkl.at(i);
    MonitorElement *mtrklet = hdNdEtaTrklets.at(i);

    mrawtrk->Reset();
    msubtrk->Reset();
    mtrklet->Reset();

    TH3F *hsig = NsigTracklets.at(i);
    TH3F *hbkg = NbkgTracklets.at(i);
    TH1F *hepa = NEvsPerEta.at(i);
    
    for(int etabin=1;etabin<=netabins;++etabin) {
      const double etaval   = AlphaTracklets->GetXaxis()->GetBinCenter(etabin);
      const double etawidth = AlphaTracklets->GetXaxis()->GetBinWidth(etabin);
      const int    zvetabin = hepa->GetXaxis()->FindFixBin(etaval);
      const double events   = hepa->GetBinContent(zvetabin);
      if (!events)
        continue;

//       int zvbin1 = 1; // UNUSED
//       int zvbin2 = nvzbins; // UNUSED
//       if (zvetabin==0) { // reduced phase space
//         zvbin1 = zvhbin;
//       } else if (zvetabin==2) {
//         zvbin2 = zvlbin;
//       }

      double dndetaraw     = 0;
      double dndetasub     = 0;
      double dndeta        = 0;
      double dndetarawerr  = 0;
      double dndetasuberr  = 0;
      double dndetaerr     = 0;
      for(int hitbin=1;hitbin<=nhitbins;++hitbin) {
        for(int vzbin=1;vzbin<=nvzbins;++vzbin) {
          int gbin = AlphaTracklets->GetBin(etabin,hitbin,vzbin);
          const double nsig = hsig->GetBinContent(gbin);
          dndetaraw += nsig;
          const double nbkg = hbkg->GetBinContent(gbin);
          const double nsub = nsig - nbkg;
          if (nsub<0) {
            CP(2) print(2,Form("Got negative contributions: %d %d %d %f",etabin,hitbin,vzbin,nsub));
            continue;
          }
          dndetasub += nsub;
          const double alpha = AlphaTracklets->GetBinContent(gbin);
          dndeta += alpha*nsub;
          double nsig2  = nsig*nsig;
          double nsub2  = nsub*nsub;
          double alpha2 = alpha*alpha;
          dndetarawerr  += nsig2;
          dndetasuberr  += nsub2;
          dndetaerr     += alpha2*nsub2;
        }
      }

      double norm  = etawidth * events;
      double enorm = etawidth * norm;
      dndetaraw    /= norm;
      dndetasub    /= norm;
      dndeta       /= norm;
      dndetarawerr /= enorm;
      dndetasuberr /= enorm;
      dndetaerr    /= enorm;
      double dndetarawsigma2 = (dndetaraw*dndetaraw - dndetarawerr) / events;
      double dndetasubsigma2 = (dndetasub*dndetasub - dndetasuberr) / events;
      double dndetasigma2    = (dndeta*dndeta - dndetaerr) / events;
      mrawtrk->setBinContent(etabin,dndetaraw);
      mrawtrk->setBinError(etabin,TMath::Sqrt(TMath::Abs(dndetarawsigma2)));
      msubtrk->setBinContent(etabin,dndetasub);
      msubtrk->setBinError(etabin,TMath::Sqrt(TMath::Abs(dndetasubsigma2)));
      mtrklet->setBinContent(etabin,dndeta);
      mtrklet->setBinError(etabin,TMath::Sqrt(TMath::Abs(dndetasigma2)));
    }
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::endLuminosityBlock(const LuminosityBlock &l, 
                                     const EventSetup &iSetup)
{
  // Update various histograms.

  repSummary_->Fill(1.);
  repSumMap_->Fill(0.5,0.5,1.);

  filldNdeta(AlphaTracklets12_,NsigTracklets12_,NbkgTracklets12_,
             hEvtCountsPerEta12_,hdNdEtaRawTrkl12_,hdNdEtaSubTrkl12_,hdNdEtaTrklets12_);
  filldNdeta(AlphaTracklets13_,NsigTracklets13_,NbkgTracklets13_,
             hEvtCountsPerEta13_,hdNdEtaRawTrkl13_,hdNdEtaSubTrkl13_,hdNdEtaTrklets13_);
  filldNdeta(AlphaTracklets23_,NsigTracklets23_,NbkgTracklets23_,
             hEvtCountsPerEta23_,hdNdEtaRawTrkl23_,hdNdEtaSubTrkl23_,hdNdEtaTrklets23_);
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::endRun(const Run &, const EventSetup &)
{
  // End run, cleanup. TODO: can this be called several times in DQM???

  for(size_t i=0;i<NsigTracklets12_.size();++i) {
    NsigTracklets12_.at(i)->Reset();
    NbkgTracklets12_.at(i)->Reset();
  }
  for(size_t i=0;i<NsigTracklets13_.size();++i) {
    NsigTracklets13_.at(i)->Reset();
    NbkgTracklets13_.at(i)->Reset();
  }
  for(size_t i=0;i<NsigTracklets23_.size();++i) {
    NsigTracklets23_.at(i)->Reset();
    NbkgTracklets23_.at(i)->Reset();
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fill1D(std::vector<TH1F*> &hs, double val, double w)
{
  // Loop over histograms and fill if trigger has fired.

  for(size_t i=0;i<hs.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    hs.at(i)->Fill(val,w);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fill1D(std::vector<MonitorElement*> &mes, double val, double w)
{
  // Loop over histograms and fill if trigger has fired.

  for(size_t i=0;i<mes.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    mes.at(i)->Fill(val,w);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fill2D(std::vector<TH2F*> &hs, double valx, double valy, double w)
{
  // Loop over histograms and fill if trigger has fired.

  for(size_t i=0;i<hs.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    hs.at(i)->Fill(valx, valy ,w);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fill2D(std::vector<MonitorElement*> &mes, double valx, double valy, double w)
{
  // Loop over histograms and fill if trigger has fired.

  for(size_t i=0;i<mes.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    mes.at(i)->Fill(valx, valy ,w);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fill3D(std::vector<TH3F*> &hs, int gbin, double w)
{
  // Loop over histograms and fill if trigger has fired.

  for(size_t i=0;i<hs.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    hs.at(i)->AddBinContent(gbin, w);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fillHltBits(const Event &iEvent)
{
  // Fill HLT trigger bits.

  Handle<TriggerResults> triggerResultsHLT;
  getProduct(hltUsedResName_, triggerResultsHLT, iEvent);

  for(size_t i=0;i<hltTrgBits_.size();++i) {
    if (hltTrgBits_.at(i)<0) 
      continue; //ignore unknown trigger 
    size_t tbit = hltTrgBits_.at(i);
    if (tbit<triggerResultsHLT->size()) {
      hltTrgDeci_[i] = triggerResultsHLT->accept(tbit);
      if (0) print(0,Form("Decision %i for %s",
                          (int)hltTrgDeci_.at(i), hltTrgUsedNames_.at(i).c_str()));
    } else {
      print(2,Form("Problem slot %i for bit %i for %s",
                   int(i), int(tbit), hltTrgUsedNames_.at(i).c_str()));
    }
  }

  // fill correlation histogram
  for(size_t i=0;i<hltTrgBits_.size();++i) {
    if (hltTrgDeci_.at(i))
      h2TrigCorr_->Fill(i,i);
    for(size_t j=i+1;j<hltTrgBits_.size();++j) {
      if (hltTrgDeci_.at(i) && hltTrgDeci_.at(j))
        h2TrigCorr_->Fill(i,j);
      if (hltTrgDeci_.at(i) && !hltTrgDeci_.at(j))
        h2TrigCorr_->Fill(j,i);
    }
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fillPixels(const Event &iEvent, const edm::EventSetup& iSetup) 
{
  // Fill pixel hit collections.

  bpix1_.clear();
  bpix2_.clear();
  bpix3_.clear();

  Handle<SiPixelRecHitCollection> hRecHits;
  if (!getProductSafe(pixelName_, hRecHits, iEvent)) {
    CP(2) print(2,Form("Can not obtain pixel hit collection with name %s", pixelName_.c_str()));
    return;
  }

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  const SiPixelRecHitCollection *hits = hRecHits.product();
  for(SiPixelRecHitCollection::DataContainer::const_iterator hit = hits->data().begin(), 
        end = hits->data().end(); hit != end; ++hit) {

    if (!hit->isValid())
      continue;

    if (useRecHitQ_) {
      if (hit->isOnEdge() || hit->hasBadPixels())
        continue;
    }

    DetId id(hit->geographicalId());
    if(id.subdetId() != int(PixelSubdetector::PixelBarrel))
      continue;

    const PixelGeomDetUnit *pgdu = static_cast<const PixelGeomDetUnit*>(tgeo_->idToDet(id));

    if (usePixelQ_) {
      const PixelTopology *pixTopo = &(pgdu->specificTopology());
      vector<SiPixelCluster::Pixel> pixels(hit->cluster()->pixels());
      bool pixelOnEdge = false;
      for(std::vector<SiPixelCluster::Pixel>::const_iterator pixel = pixels.begin(); 
          pixel != pixels.end(); ++pixel) {
        int pixelX = pixel->x;
        int pixelY = pixel->y;
        if(pixTopo->isItEdgePixelInX(pixelX) || pixTopo->isItEdgePixelInY(pixelY)) {
          pixelOnEdge = true;
          break;
        }
      }
      if (pixelOnEdge)
        continue;
    }

    LocalPoint lpos = LocalPoint(hit->localPosition().x(),
                                 hit->localPosition().y(),
                                 hit->localPosition().z());
    GlobalPoint gpos = pgdu->toGlobal(lpos);
    double adc   = hit->cluster()->charge()/135.;
    double sizex = hit->cluster()->sizeX();
    double sizey = hit->cluster()->sizeY();

    Pixel pix(gpos, adc, sizex, sizey);

    
    int layer = tTopo->pxbLayer(id);

    if (layer==1) {
      bpix1_.push_back(pix);     
      fill1D(hdNdEtaHitsL1_,pix.eta());
      fill1D(hdNdPhiHitsL1_,pix.phi());
    } else if (layer==2) {
      bpix2_.push_back(pix);     
      fill1D(hdNdEtaHitsL2_,pix.eta());
      fill1D(hdNdPhiHitsL2_,pix.phi());
    } else {
      bpix3_.push_back(pix);     
      fill1D(hdNdEtaHitsL3_,pix.eta());
      fill1D(hdNdPhiHitsL3_,pix.phi());
    }
  }

  // fill overall histograms
  fill1D(hNhitsL1_,bpix1_.size());
  fill1D(hNhitsL2_,bpix2_.size());
  fill1D(hNhitsL3_,bpix3_.size());
  fill1D(hNhitsL1z_,bpix1_.size());
  fill1D(hNhitsL2z_,bpix2_.size());
  fill1D(hNhitsL3z_,bpix3_.size());
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fillPixelClusterInfos(const Event &iEvent, int which)
{
  // Get information related to pixel cluster counting methods.

  double vz = -999;

  if (clusterVtxName_.size()) { // get vertex from producer
    Handle<reco::VertexCollection> hVertexCollection;
    if (!getProductSafe(clusterVtxName_, hVertexCollection, iEvent)) {
      CP(2) print(2,Form("Can not obtain pixel vertex from cluster collection with name %s", 
                         clusterVtxName_.c_str()));
      return;
    }

    const reco::VertexCollection *vertices = hVertexCollection.product();
    if (!vertices || vertices->size()==0)
      return;
    reco::VertexCollection::const_iterator vertex = vertices->begin();
    vz = vertex->z();
  } else { // calculate vertex from clusters
    std::vector<Pixel> allp(bpix1_);
    allp.insert(allp.end(),bpix2_.begin(),bpix2_.end());
    allp.insert(allp.end(),bpix3_.begin(),bpix3_.end());
    vz = vertexZFromClusters(allp);
  }
  if (vz<=-999)
    return;

  fill1D(hClusterVertexZ_,vz);
  if (which>=12)
    fillPixelClusterInfos(vz, bpix1_, hClusterYSize1_, hClusterADC1_);
  if (which>=13)
  fillPixelClusterInfos(vz, bpix2_, hClusterYSize2_, hClusterADC2_);
  if (which>=23)
    fillPixelClusterInfos(vz, bpix3_, hClusterYSize3_, hClusterADC3_);
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fillPixelClusterInfos(const double vz,
                                        const std::vector<Pixel> &pix, 
                                        std::vector<MonitorElement*> &hClusterYSize,
                                        std::vector<MonitorElement*> &hClusterADC)
{
  // Fill histograms with pixel cluster counting related infos.

  for(size_t i = 0; i<pix.size(); ++i) {
    const Pixel &p(pix.at(i));
    const GlobalPoint tmp(p.x(),p.y(),p.z()-vz);
    fill2D(hClusterYSize, tmp.eta(), p.sizey());
    fill2D(hClusterADC, tmp.eta(), p.adc());
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fillTracklets(const Event &iEvent, int which) 
{
  // Fill tracklet collections.

  if (which>=12) {
    fillTracklets(btracklets12_,bpix1_,bpix2_,trackletV12_); 
    fillTracklets(btracklets12_, bpix1_, trackletV12_, AlphaTracklets12_,
                  NsigTracklets12_, NbkgTracklets12_, hEvtCountsPerEta12_, hTrkRawDetaDphi12_,
                  hTrkRawDeta12_, hTrkRawDphi12_, hRawTrkEtaVtxZ12_);
  }
  if (which>=13) {
    fillTracklets(btracklets13_,bpix1_,bpix3_,trackletV12_);
    fillTracklets(btracklets13_,bpix1_,trackletV13_,AlphaTracklets13_,
                  NsigTracklets13_, NbkgTracklets13_, hEvtCountsPerEta13_, hTrkRawDetaDphi13_, 
                  hTrkRawDeta13_, hTrkRawDphi13_, hRawTrkEtaVtxZ13_);
  }
  if (which>=23) {
    fillTracklets(btracklets23_,bpix2_,bpix3_,trackletV12_);
    fillTracklets(btracklets23_,bpix1_, trackletV12_, AlphaTracklets23_,
                  NsigTracklets23_, NbkgTracklets23_, hEvtCountsPerEta23_, hTrkRawDetaDphi23_,
                  hTrkRawDeta23_, hTrkRawDphi23_, hRawTrkEtaVtxZ23_);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fillTracklets(std::vector<Tracklet> &tracklets, 
                                const std::vector<Pixel> &pix1, 
                                const std::vector<Pixel> &pix2,
                                const Vertex &trackletV)
{
  // Fill tracklet collection from given pixel hit collections.

  tracklets.clear();

  if (TMath::Abs(trackletV.z())>ZVCut_)
    return;

  // build tracklets
  std::vector<Tracklet> tmptrkls;
  tmptrkls.reserve(pix1.size()*pix2.size());
  for(size_t i = 0; i<pix1.size(); ++i) {
    const GlobalPoint tmp1(pix1.at(i).x(),pix1.at(i).y(),pix1.at(i).z()-trackletV.z());
    const Pixel p1(tmp1);
    for(size_t j = 0; j<pix2.size(); ++j) {
      const GlobalPoint tmp2(pix2.at(j).x(),pix2.at(j).y(),pix2.at(j).z()-trackletV.z());
      const Pixel &p2(tmp2);
      Tracklet tracklet(p1,p2);
      tracklet.seti1(i);
      tracklet.seti2(j);
      tmptrkls.push_back(tracklet);
    }
  }

  // sort tracklets according to deltaeta
  sort(tmptrkls.begin(),tmptrkls.end(),compareTracklets);

  // clean tracklets
  vector<bool> secused(pix2.size(),false);
  for(size_t k=0; k<tmptrkls.size(); ++k) {
    const Tracklet &tl(tmptrkls.at(k));
    size_t p2ind = tl.i2();
    if (secused.at(p2ind)) 
      continue;
    secused[p2ind] = true;
    tracklets.push_back(tl);
    if (tracklets.size()==pix2.size())
      break; // can not have more tracklets than pixels on "second" layer
   }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fillTracklets(const std::vector<Tracklet> &tracklets, 
                                const std::vector<Pixel> &pixels,
                                const Vertex &trackletV,
                                const TH3F *AlphaTracklets,
                                std::vector<TH3F*> &NsigTracklets,
                                std::vector<TH3F*> &NbkgTracklets,
                                std::vector<TH1F*> &eventpereta,
                                std::vector<MonitorElement*> &detaphi,
                                std::vector<MonitorElement*> &deta,
                                std::vector<MonitorElement*> &dphi,
                                std::vector<MonitorElement*> &etavsvtx)
{
  // Fill tracklet related histograms.

  if (!AlphaTracklets)
    return;

  if (tracklets.size()==0)
    return;

  // fill events per etabin per trigger bit
  for(size_t i=0;i<eventpereta.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    TH1 *h = eventpereta.at(i);
    h->AddBinContent(1,1);
    if (trackletV.z()>+ZVVtxRegion_)
      h->AddBinContent(0,1);
    else if (trackletV.z()<-ZVVtxRegion_)
      h->AddBinContent(2,1);
  }

  // fill tracklet based info
  TAxis *xa = AlphaTracklets->GetXaxis();
  int ybin  = AlphaTracklets->GetYaxis()->FindFixBin(pixels.size());
  int zbin  = AlphaTracklets->GetZaxis()->FindFixBin(trackletV.z());
  int tbin  = AlphaTracklets->GetBin(0,ybin,zbin);
  for(size_t k=0; k<tracklets.size(); ++k) {
    const Tracklet &tl(tracklets.at(k));
    fill2D(detaphi,tl.deta(),tl.dphi());
    fill1D(deta,tl.deta());
    fill1D(dphi,tl.dphi());
    int ebin = xa->FindFixBin(tl.eta());
    int gbin = ebin + tbin;
    fill2D(etavsvtx,tl.eta(),trackletV.z());

    double deta = TMath::Abs(tl.deta());
    double dphi = TMath::Abs(tl.dphi());

    if ((deta<sigEtaCut_) && (dphi<sigPhiCut_))
      fill3D(NsigTracklets,gbin);
    else if ((deta<bkgEtaCut_) && (dphi<bkgPhiCut_) && (dphi>sigPhiCut_))
      fill3D(NbkgTracklets,gbin);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::reallyPrint(int level, const char *msg)
{
  // Print out information dependent on level and verbosity.

  if (level==0) {
    printf("QcdLowPtDQM: %s\n", msg); 
  } else if (level==1) {
    LogWarning("QcdLowPtDQM") << msg << std::endl;
  } else if (level==2) {
    LogError("QcdLowPtDQM") << msg << std::endl;
  } else if (level==3) {
    LogError("QcdLowPtDQM") << msg << std::endl;
    throw edm::Exception(errors::Configuration, "QcdLowPtDQM\n") << msg << std::endl;
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::trackletVertexUnbinned(const Event &iEvent, int which)
{
  // Estimate tracklet based z vertex.

  if (which>=12) {
    trackletVertexUnbinned(bpix1_,bpix2_,trackletV12_);
    fill1D(hTrkVtxZ12_,trackletV12_.z());
  }
  if (which>=13) {
    trackletVertexUnbinned(bpix1_,bpix3_,trackletV13_);
    fill1D(hTrkVtxZ13_,trackletV13_.z());
  }
  if (which>=23) {
    trackletVertexUnbinned(bpix2_,bpix3_,trackletV23_);
    fill1D(hTrkVtxZ23_,trackletV23_.z());
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::trackletVertexUnbinned(std::vector<Pixel> &pix1, 
                                         std::vector<Pixel> &pix2,
                                         Vertex &vtx)
{
  // Calculate tracklet based z vertex position. 
  // At first build zvertex candidates from tracklet prototypes,
  // then group zvertex candidates and calculate mean position
  // from most likely cluster.

  vector<double> zvCands;
  zvCands.reserve(pix1.size()*pix2.size());

  // build candidates
  for(size_t i = 0; i<pix1.size(); ++i) {
    const Pixel &p1(pix1.at(i));
    const double r12 = p1.x()*p1.x()+p1.y()*p1.y();
    for(size_t j = 0; j<pix2.size(); ++j) {
      const Pixel &p2(pix2.at(j));
      if (TMath::Abs(Geom::deltaPhi(p1.phi(),p2.phi()))>dPhiVc_)
        continue;
      const double r22 = p2.x()*p2.x()+p2.y()*p2.y();
      const double vz = p1.z() - (p2.z()-p1.z())/(TMath::Sqrt(r22/r12)-1);
      if (TMath::IsNaN(vz))
        continue;
      if (TMath::Abs(vz)>25)
        continue;
      zvCands.push_back(vz);
    }
  }

  // sort cluster candidates
  sort(zvCands.begin(),zvCands.end());

  int    mcl=0;
  double ms2=1e10;
  double mzv=1e10;

  // cluster candidates and calculate mean z position
  for(size_t i = 0; i<zvCands.size(); ++i) {
    double z1 = zvCands.at(i);
    int    ncl   = 0;
    double mean  = 0;
    double mean2 = 0;
    for(size_t j = i; j<zvCands.size(); ++j) {
      double z2 = zvCands.at(j);
      if (TMath::Abs(z1-z2)>dZVc_)
        break;
      ++ncl;
      mean += z2;
      mean2 += z2*z2;
    }      
    if (ncl>0) {
      mean /= ncl;
      mean2 /= ncl;
    }
    double_t s2 = mean*mean - mean2;

    if ((ncl<mcl) || (ncl==mcl && s2>ms2))
      continue;

    mzv = mean;
    ms2 = s2;
    mcl = ncl;
  }

  // set the vertex
  vtx.set(mcl, mzv, ms2);
}

//--------------------------------------------------------------------------------------------------
double QcdLowPtDQM::vertexZFromClusters(const std::vector<Pixel> &pix) const
{
  // Estimate z vertex position from clusters.

  double chi_max = 1e+9;
  double z_best  = -999;
  int nhits_max  = 0;

  for(double z0 = -15.9; z0 <= 15.95; z0 += 0.1) {
    int nhits  = 0;
    double chi = 0;
    for(size_t i=0; i<pix.size(); ++i) {
      const Pixel &p = pix.at(i);

      // predicted cluster width in y direction
      double pval = 2*TMath::Abs(p.z()-z0)/p.rho() + 0.5; // FIXME
      double chitest = TMath::Abs(pval - p.sizey());
      if(chitest <= 1.) { 
        chi += chitest;
        ++nhits;
      }
    }

    if(nhits <= 0)
      continue;

    if(nhits < nhits_max)
      continue;

    if ((nhits > nhits_max) || (chi < chi_max)) { 
      z_best    = z0; 
      nhits_max = nhits; 
      chi_max   = chi; 
    }
  }

  return z_best;
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::yieldAlphaHistogram(int which)
{
  // Create alpha histogram. Code created by Yen-Jie and included by hand:
  // Alpha value for 1st + 2nd tracklet calculated from 1.9 M PYTHIA 900 GeV
  // sample produced by Yetkin with CMS official tune.

  if (which>=12) {

    const int nEtaBin = 12;
    const int nHitBin = 14;
    const int nVzBin  = 10;

    double HitBins[nHitBin+1] = {0,5,10,15,20,25,30,35,40,50,60,80,100,200,700};

    double EtaBins[nEtaBin+1];
    for (int i=0;i<=nEtaBin;i++)
      EtaBins[i] = (double)i*6.0/(double)nEtaBin-3.0;
    double VzBins[nVzBin+1];
    for (int i=0;i<=nVzBin;i++)
      VzBins[i] = (double)i*20.0/(double)nVzBin-10.0;

    AlphaTracklets12_ = new TH3F("hAlphaTracklets12",
                                 "Alpha for tracklets12;#eta;#hits;vz [cm]",
                                 nEtaBin, EtaBins, nHitBin, HitBins, nVzBin, VzBins);
    AlphaTracklets12_->SetDirectory(0);

    AlphaTracklets12_->SetBinContent(2,1,7,3.55991);
    AlphaTracklets12_->SetBinContent(2,1,8,2.40439);
    AlphaTracklets12_->SetBinContent(2,1,9,1.82051);
    AlphaTracklets12_->SetBinContent(2,1,10,1.46392);
    AlphaTracklets12_->SetBinContent(2,2,7,4.54825);
    AlphaTracklets12_->SetBinContent(2,2,8,2.88097);
    AlphaTracklets12_->SetBinContent(2,2,9,2.28488);
    AlphaTracklets12_->SetBinContent(2,2,10,1.81763);
    AlphaTracklets12_->SetBinContent(2,3,7,4.72804);
    AlphaTracklets12_->SetBinContent(2,3,8,3.10072);
    AlphaTracklets12_->SetBinContent(2,3,9,2.28212);
    AlphaTracklets12_->SetBinContent(2,3,10,1.81332);
    AlphaTracklets12_->SetBinContent(2,4,7,4.80419);
    AlphaTracklets12_->SetBinContent(2,4,8,3.00091);
    AlphaTracklets12_->SetBinContent(2,4,9,2.26618);
    AlphaTracklets12_->SetBinContent(2,4,10,1.87004);
    AlphaTracklets12_->SetBinContent(2,5,7,4.74302);
    AlphaTracklets12_->SetBinContent(2,5,8,3.16951);
    AlphaTracklets12_->SetBinContent(2,5,9,2.37798);
    AlphaTracklets12_->SetBinContent(2,5,10,1.95397);
    AlphaTracklets12_->SetBinContent(2,6,7,4.67748);
    AlphaTracklets12_->SetBinContent(2,6,8,3.11594);
    AlphaTracklets12_->SetBinContent(2,6,9,2.31264);
    AlphaTracklets12_->SetBinContent(2,6,10,1.93345);
    AlphaTracklets12_->SetBinContent(2,7,7,4.96439);
    AlphaTracklets12_->SetBinContent(2,7,8,3.15187);
    AlphaTracklets12_->SetBinContent(2,7,9,2.28339);
    AlphaTracklets12_->SetBinContent(2,7,10,1.84058);
    AlphaTracklets12_->SetBinContent(2,8,7,4.86141);
    AlphaTracklets12_->SetBinContent(2,8,8,3.27879);
    AlphaTracklets12_->SetBinContent(2,8,9,2.29148);
    AlphaTracklets12_->SetBinContent(2,8,10,1.91509);
    AlphaTracklets12_->SetBinContent(2,9,7,5.10376);
    AlphaTracklets12_->SetBinContent(2,9,8,3.25626);
    AlphaTracklets12_->SetBinContent(2,9,9,2.3682);
    AlphaTracklets12_->SetBinContent(2,9,10,2.0008);
    AlphaTracklets12_->SetBinContent(2,10,7,5.09227);
    AlphaTracklets12_->SetBinContent(2,10,8,3.41062);
    AlphaTracklets12_->SetBinContent(2,10,9,2.44921);
    AlphaTracklets12_->SetBinContent(2,10,10,1.95082);
    AlphaTracklets12_->SetBinContent(2,11,7,5.14087);
    AlphaTracklets12_->SetBinContent(2,11,8,3.3516);
    AlphaTracklets12_->SetBinContent(2,11,9,2.63825);
    AlphaTracklets12_->SetBinContent(2,11,10,2.22036);
    AlphaTracklets12_->SetBinContent(2,12,7,5.51994);
    AlphaTracklets12_->SetBinContent(2,12,8,3.71484);
    AlphaTracklets12_->SetBinContent(2,12,9,3.06667);
    AlphaTracklets12_->SetBinContent(2,12,10,1.86875);
    AlphaTracklets12_->SetBinContent(2,13,7,5.58242);
    AlphaTracklets12_->SetBinContent(2,13,8,3.64103);
    AlphaTracklets12_->SetBinContent(2,13,9,2.11688);
    AlphaTracklets12_->SetBinContent(2,13,10,1.72093);
    AlphaTracklets12_->SetBinContent(2,14,7,0);
    AlphaTracklets12_->SetBinContent(2,14,8,0);
    AlphaTracklets12_->SetBinContent(2,14,9,0);
    AlphaTracklets12_->SetBinContent(2,14,10,0);
    AlphaTracklets12_->SetBinContent(3,1,1,3.28916);
    AlphaTracklets12_->SetBinContent(3,1,2,1.92188);
    AlphaTracklets12_->SetBinContent(3,1,3,1.38314);
    AlphaTracklets12_->SetBinContent(3,1,4,1.25626);
    AlphaTracklets12_->SetBinContent(3,1,5,1.0602);
    AlphaTracklets12_->SetBinContent(3,1,6,1.04012);
    AlphaTracklets12_->SetBinContent(3,1,7,1.05597);
    AlphaTracklets12_->SetBinContent(3,1,8,1.04124);
    AlphaTracklets12_->SetBinContent(3,1,9,1.05176);
    AlphaTracklets12_->SetBinContent(3,1,10,1);
    AlphaTracklets12_->SetBinContent(3,2,1,4.15175);
    AlphaTracklets12_->SetBinContent(3,2,2,2.22027);
    AlphaTracklets12_->SetBinContent(3,2,3,1.60242);
    AlphaTracklets12_->SetBinContent(3,2,4,1.31468);
    AlphaTracklets12_->SetBinContent(3,2,5,1.11846);
    AlphaTracklets12_->SetBinContent(3,2,6,1.04985);
    AlphaTracklets12_->SetBinContent(3,2,7,1.05568);
    AlphaTracklets12_->SetBinContent(3,2,8,1.05742);
    AlphaTracklets12_->SetBinContent(3,2,9,1.02244);
    AlphaTracklets12_->SetBinContent(3,2,10,1.04331);
    AlphaTracklets12_->SetBinContent(3,3,1,4.36126);
    AlphaTracklets12_->SetBinContent(3,3,2,2.22884);
    AlphaTracklets12_->SetBinContent(3,3,3,1.62803);
    AlphaTracklets12_->SetBinContent(3,3,4,1.33845);
    AlphaTracklets12_->SetBinContent(3,3,5,1.13194);
    AlphaTracklets12_->SetBinContent(3,3,6,1.05987);
    AlphaTracklets12_->SetBinContent(3,3,7,1.06358);
    AlphaTracklets12_->SetBinContent(3,3,8,1.06075);
    AlphaTracklets12_->SetBinContent(3,3,9,1.04847);
    AlphaTracklets12_->SetBinContent(3,3,10,1.06997);
    AlphaTracklets12_->SetBinContent(3,4,1,4.40845);
    AlphaTracklets12_->SetBinContent(3,4,2,2.26589);
    AlphaTracklets12_->SetBinContent(3,4,3,1.69427);
    AlphaTracklets12_->SetBinContent(3,4,4,1.37433);
    AlphaTracklets12_->SetBinContent(3,4,5,1.13619);
    AlphaTracklets12_->SetBinContent(3,4,6,1.07042);
    AlphaTracklets12_->SetBinContent(3,4,7,1.08095);
    AlphaTracklets12_->SetBinContent(3,4,8,1.07258);
    AlphaTracklets12_->SetBinContent(3,4,9,1.0585);
    AlphaTracklets12_->SetBinContent(3,4,10,1.05834);
    AlphaTracklets12_->SetBinContent(3,5,1,3.94027);
    AlphaTracklets12_->SetBinContent(3,5,2,2.33404);
    AlphaTracklets12_->SetBinContent(3,5,3,1.67866);
    AlphaTracklets12_->SetBinContent(3,5,4,1.36896);
    AlphaTracklets12_->SetBinContent(3,5,5,1.16415);
    AlphaTracklets12_->SetBinContent(3,5,6,1.07235);
    AlphaTracklets12_->SetBinContent(3,5,7,1.08951);
    AlphaTracklets12_->SetBinContent(3,5,8,1.07149);
    AlphaTracklets12_->SetBinContent(3,5,9,1.06005);
    AlphaTracklets12_->SetBinContent(3,5,10,1.0835);
    AlphaTracklets12_->SetBinContent(3,6,1,4.13972);
    AlphaTracklets12_->SetBinContent(3,6,2,2.37481);
    AlphaTracklets12_->SetBinContent(3,6,3,1.68345);
    AlphaTracklets12_->SetBinContent(3,6,4,1.39681);
    AlphaTracklets12_->SetBinContent(3,6,5,1.16352);
    AlphaTracklets12_->SetBinContent(3,6,6,1.09194);
    AlphaTracklets12_->SetBinContent(3,6,7,1.09722);
    AlphaTracklets12_->SetBinContent(3,6,8,1.10568);
    AlphaTracklets12_->SetBinContent(3,6,9,1.08304);
    AlphaTracklets12_->SetBinContent(3,6,10,1.07302);
    AlphaTracklets12_->SetBinContent(3,7,1,3.88559);
    AlphaTracklets12_->SetBinContent(3,7,2,2.35657);
    AlphaTracklets12_->SetBinContent(3,7,3,1.70128);
    AlphaTracklets12_->SetBinContent(3,7,4,1.41423);
    AlphaTracklets12_->SetBinContent(3,7,5,1.17898);
    AlphaTracklets12_->SetBinContent(3,7,6,1.11033);
    AlphaTracklets12_->SetBinContent(3,7,7,1.103);
    AlphaTracklets12_->SetBinContent(3,7,8,1.09125);
    AlphaTracklets12_->SetBinContent(3,7,9,1.09621);
    AlphaTracklets12_->SetBinContent(3,7,10,1.11348);
    AlphaTracklets12_->SetBinContent(3,8,1,4.43869);
    AlphaTracklets12_->SetBinContent(3,8,2,2.31694);
    AlphaTracklets12_->SetBinContent(3,8,3,1.75354);
    AlphaTracklets12_->SetBinContent(3,8,4,1.43747);
    AlphaTracklets12_->SetBinContent(3,8,5,1.18856);
    AlphaTracklets12_->SetBinContent(3,8,6,1.13039);
    AlphaTracklets12_->SetBinContent(3,8,7,1.13314);
    AlphaTracklets12_->SetBinContent(3,8,8,1.11606);
    AlphaTracklets12_->SetBinContent(3,8,9,1.10825);
    AlphaTracklets12_->SetBinContent(3,8,10,1.02513);
    AlphaTracklets12_->SetBinContent(3,9,1,3.78413);
    AlphaTracklets12_->SetBinContent(3,9,2,2.35657);
    AlphaTracklets12_->SetBinContent(3,9,3,1.75728);
    AlphaTracklets12_->SetBinContent(3,9,4,1.44676);
    AlphaTracklets12_->SetBinContent(3,9,5,1.21071);
    AlphaTracklets12_->SetBinContent(3,9,6,1.13125);
    AlphaTracklets12_->SetBinContent(3,9,7,1.13183);
    AlphaTracklets12_->SetBinContent(3,9,8,1.12655);
    AlphaTracklets12_->SetBinContent(3,9,9,1.12749);
    AlphaTracklets12_->SetBinContent(3,9,10,1.14435);
    AlphaTracklets12_->SetBinContent(3,10,1,4.25485);
    AlphaTracklets12_->SetBinContent(3,10,2,2.47167);
    AlphaTracklets12_->SetBinContent(3,10,3,1.75454);
    AlphaTracklets12_->SetBinContent(3,10,4,1.47723);
    AlphaTracklets12_->SetBinContent(3,10,5,1.22674);
    AlphaTracklets12_->SetBinContent(3,10,6,1.15583);
    AlphaTracklets12_->SetBinContent(3,10,7,1.15841);
    AlphaTracklets12_->SetBinContent(3,10,8,1.17341);
    AlphaTracklets12_->SetBinContent(3,10,9,1.14979);
    AlphaTracklets12_->SetBinContent(3,10,10,1.16575);
    AlphaTracklets12_->SetBinContent(3,11,1,4.29655);
    AlphaTracklets12_->SetBinContent(3,11,2,2.40128);
    AlphaTracklets12_->SetBinContent(3,11,3,1.85054);
    AlphaTracklets12_->SetBinContent(3,11,4,1.48338);
    AlphaTracklets12_->SetBinContent(3,11,5,1.27537);
    AlphaTracklets12_->SetBinContent(3,11,6,1.17767);
    AlphaTracklets12_->SetBinContent(3,11,7,1.19398);
    AlphaTracklets12_->SetBinContent(3,11,8,1.17399);
    AlphaTracklets12_->SetBinContent(3,11,9,1.18336);
    AlphaTracklets12_->SetBinContent(3,11,10,1.17576);
    AlphaTracklets12_->SetBinContent(3,12,1,5.12727);
    AlphaTracklets12_->SetBinContent(3,12,2,2.47253);
    AlphaTracklets12_->SetBinContent(3,12,3,1.95985);
    AlphaTracklets12_->SetBinContent(3,12,4,1.57112);
    AlphaTracklets12_->SetBinContent(3,12,5,1.31289);
    AlphaTracklets12_->SetBinContent(3,12,6,1.234);
    AlphaTracklets12_->SetBinContent(3,12,7,1.21577);
    AlphaTracklets12_->SetBinContent(3,12,8,1.1701);
    AlphaTracklets12_->SetBinContent(3,12,9,1.21698);
    AlphaTracklets12_->SetBinContent(3,12,10,1.16594);
    AlphaTracklets12_->SetBinContent(3,13,1,0);
    AlphaTracklets12_->SetBinContent(3,13,2,2.62025);
    AlphaTracklets12_->SetBinContent(3,13,3,2.00495);
    AlphaTracklets12_->SetBinContent(3,13,4,1.59611);
    AlphaTracklets12_->SetBinContent(3,13,5,1.44306);
    AlphaTracklets12_->SetBinContent(3,13,6,1.25417);
    AlphaTracklets12_->SetBinContent(3,13,7,1.20339);
    AlphaTracklets12_->SetBinContent(3,13,8,1.11554);
    AlphaTracklets12_->SetBinContent(3,13,9,1.67033);
    AlphaTracklets12_->SetBinContent(3,13,10,1.07812);
    AlphaTracklets12_->SetBinContent(3,14,1,0);
    AlphaTracklets12_->SetBinContent(3,14,2,0);
    AlphaTracklets12_->SetBinContent(3,14,3,0);
    AlphaTracklets12_->SetBinContent(3,14,4,0);
    AlphaTracklets12_->SetBinContent(3,14,5,0);
    AlphaTracklets12_->SetBinContent(3,14,6,0);
    AlphaTracklets12_->SetBinContent(3,14,7,0);
    AlphaTracklets12_->SetBinContent(3,14,8,0);
    AlphaTracklets12_->SetBinContent(3,14,9,0);
    AlphaTracklets12_->SetBinContent(3,14,10,0);
    AlphaTracklets12_->SetBinContent(4,1,1,0.990826);
    AlphaTracklets12_->SetBinContent(4,1,2,1.07515);
    AlphaTracklets12_->SetBinContent(4,1,3,1.07357);
    AlphaTracklets12_->SetBinContent(4,1,4,1.03503);
    AlphaTracklets12_->SetBinContent(4,1,5,1.07278);
    AlphaTracklets12_->SetBinContent(4,1,6,1.08397);
    AlphaTracklets12_->SetBinContent(4,1,7,1.04423);
    AlphaTracklets12_->SetBinContent(4,1,8,1.03926);
    AlphaTracklets12_->SetBinContent(4,1,9,1.0966);
    AlphaTracklets12_->SetBinContent(4,1,10,1.08765);
    AlphaTracklets12_->SetBinContent(4,2,1,1.06809);
    AlphaTracklets12_->SetBinContent(4,2,2,1.10094);
    AlphaTracklets12_->SetBinContent(4,2,3,1.06723);
    AlphaTracklets12_->SetBinContent(4,2,4,1.03158);
    AlphaTracklets12_->SetBinContent(4,2,5,1.07765);
    AlphaTracklets12_->SetBinContent(4,2,6,1.0695);
    AlphaTracklets12_->SetBinContent(4,2,7,1.04128);
    AlphaTracklets12_->SetBinContent(4,2,8,1.03578);
    AlphaTracklets12_->SetBinContent(4,2,9,1.08542);
    AlphaTracklets12_->SetBinContent(4,2,10,1.08325);
    AlphaTracklets12_->SetBinContent(4,3,1,1.08696);
    AlphaTracklets12_->SetBinContent(4,3,2,1.08909);
    AlphaTracklets12_->SetBinContent(4,3,3,1.07443);
    AlphaTracklets12_->SetBinContent(4,3,4,1.0436);
    AlphaTracklets12_->SetBinContent(4,3,5,1.0787);
    AlphaTracklets12_->SetBinContent(4,3,6,1.06901);
    AlphaTracklets12_->SetBinContent(4,3,7,1.04298);
    AlphaTracklets12_->SetBinContent(4,3,8,1.03716);
    AlphaTracklets12_->SetBinContent(4,3,9,1.06357);
    AlphaTracklets12_->SetBinContent(4,3,10,1.08584);
    AlphaTracklets12_->SetBinContent(4,4,1,1.07849);
    AlphaTracklets12_->SetBinContent(4,4,2,1.10686);
    AlphaTracklets12_->SetBinContent(4,4,3,1.0799);
    AlphaTracklets12_->SetBinContent(4,4,4,1.04335);
    AlphaTracklets12_->SetBinContent(4,4,5,1.07862);
    AlphaTracklets12_->SetBinContent(4,4,6,1.0706);
    AlphaTracklets12_->SetBinContent(4,4,7,1.05704);
    AlphaTracklets12_->SetBinContent(4,4,8,1.04802);
    AlphaTracklets12_->SetBinContent(4,4,9,1.07438);
    AlphaTracklets12_->SetBinContent(4,4,10,1.06484);
    AlphaTracklets12_->SetBinContent(4,5,1,1.08867);
    AlphaTracklets12_->SetBinContent(4,5,2,1.09531);
    AlphaTracklets12_->SetBinContent(4,5,3,1.09437);
    AlphaTracklets12_->SetBinContent(4,5,4,1.05235);
    AlphaTracklets12_->SetBinContent(4,5,5,1.08007);
    AlphaTracklets12_->SetBinContent(4,5,6,1.08627);
    AlphaTracklets12_->SetBinContent(4,5,7,1.05187);
    AlphaTracklets12_->SetBinContent(4,5,8,1.05953);
    AlphaTracklets12_->SetBinContent(4,5,9,1.07976);
    AlphaTracklets12_->SetBinContent(4,5,10,1.02724);
    AlphaTracklets12_->SetBinContent(4,6,1,1.08141);
    AlphaTracklets12_->SetBinContent(4,6,2,1.11234);
    AlphaTracklets12_->SetBinContent(4,6,3,1.10276);
    AlphaTracklets12_->SetBinContent(4,6,4,1.06745);
    AlphaTracklets12_->SetBinContent(4,6,5,1.08746);
    AlphaTracklets12_->SetBinContent(4,6,6,1.08324);
    AlphaTracklets12_->SetBinContent(4,6,7,1.06103);
    AlphaTracklets12_->SetBinContent(4,6,8,1.05525);
    AlphaTracklets12_->SetBinContent(4,6,9,1.08795);
    AlphaTracklets12_->SetBinContent(4,6,10,1.09399);
    AlphaTracklets12_->SetBinContent(4,7,1,1.12134);
    AlphaTracklets12_->SetBinContent(4,7,2,1.1278);
    AlphaTracklets12_->SetBinContent(4,7,3,1.09721);
    AlphaTracklets12_->SetBinContent(4,7,4,1.07323);
    AlphaTracklets12_->SetBinContent(4,7,5,1.1028);
    AlphaTracklets12_->SetBinContent(4,7,6,1.08665);
    AlphaTracklets12_->SetBinContent(4,7,7,1.07679);
    AlphaTracklets12_->SetBinContent(4,7,8,1.07463);
    AlphaTracklets12_->SetBinContent(4,7,9,1.09972);
    AlphaTracklets12_->SetBinContent(4,7,10,1.09861);
    AlphaTracklets12_->SetBinContent(4,8,1,1.10215);
    AlphaTracklets12_->SetBinContent(4,8,2,1.11929);
    AlphaTracklets12_->SetBinContent(4,8,3,1.12204);
    AlphaTracklets12_->SetBinContent(4,8,4,1.07363);
    AlphaTracklets12_->SetBinContent(4,8,5,1.12044);
    AlphaTracklets12_->SetBinContent(4,8,6,1.09414);
    AlphaTracklets12_->SetBinContent(4,8,7,1.08568);
    AlphaTracklets12_->SetBinContent(4,8,8,1.07426);
    AlphaTracklets12_->SetBinContent(4,8,9,1.10824);
    AlphaTracklets12_->SetBinContent(4,8,10,1.08596);
    AlphaTracklets12_->SetBinContent(4,9,1,1.1135);
    AlphaTracklets12_->SetBinContent(4,9,2,1.12824);
    AlphaTracklets12_->SetBinContent(4,9,3,1.11945);
    AlphaTracklets12_->SetBinContent(4,9,4,1.08922);
    AlphaTracklets12_->SetBinContent(4,9,5,1.12099);
    AlphaTracklets12_->SetBinContent(4,9,6,1.11183);
    AlphaTracklets12_->SetBinContent(4,9,7,1.09291);
    AlphaTracklets12_->SetBinContent(4,9,8,1.08373);
    AlphaTracklets12_->SetBinContent(4,9,9,1.09873);
    AlphaTracklets12_->SetBinContent(4,9,10,1.09694);
    AlphaTracklets12_->SetBinContent(4,10,1,1.07263);
    AlphaTracklets12_->SetBinContent(4,10,2,1.14475);
    AlphaTracklets12_->SetBinContent(4,10,3,1.13479);
    AlphaTracklets12_->SetBinContent(4,10,4,1.10806);
    AlphaTracklets12_->SetBinContent(4,10,5,1.12293);
    AlphaTracklets12_->SetBinContent(4,10,6,1.12197);
    AlphaTracklets12_->SetBinContent(4,10,7,1.09523);
    AlphaTracklets12_->SetBinContent(4,10,8,1.10271);
    AlphaTracklets12_->SetBinContent(4,10,9,1.12494);
    AlphaTracklets12_->SetBinContent(4,10,10,1.06304);
    AlphaTracklets12_->SetBinContent(4,11,1,1.114);
    AlphaTracklets12_->SetBinContent(4,11,2,1.23277);
    AlphaTracklets12_->SetBinContent(4,11,3,1.15434);
    AlphaTracklets12_->SetBinContent(4,11,4,1.10469);
    AlphaTracklets12_->SetBinContent(4,11,5,1.15189);
    AlphaTracklets12_->SetBinContent(4,11,6,1.14769);
    AlphaTracklets12_->SetBinContent(4,11,7,1.11872);
    AlphaTracklets12_->SetBinContent(4,11,8,1.13645);
    AlphaTracklets12_->SetBinContent(4,11,9,1.15785);
    AlphaTracklets12_->SetBinContent(4,11,10,1.12472);
    AlphaTracklets12_->SetBinContent(4,12,1,1.03819);
    AlphaTracklets12_->SetBinContent(4,12,2,1.11594);
    AlphaTracklets12_->SetBinContent(4,12,3,1.16559);
    AlphaTracklets12_->SetBinContent(4,12,4,1.13169);
    AlphaTracklets12_->SetBinContent(4,12,5,1.16292);
    AlphaTracklets12_->SetBinContent(4,12,6,1.14114);
    AlphaTracklets12_->SetBinContent(4,12,7,1.15091);
    AlphaTracklets12_->SetBinContent(4,12,8,1.12486);
    AlphaTracklets12_->SetBinContent(4,12,9,1.17958);
    AlphaTracklets12_->SetBinContent(4,12,10,1.00685);
    AlphaTracklets12_->SetBinContent(4,13,1,1.48148);
    AlphaTracklets12_->SetBinContent(4,13,2,1.38065);
    AlphaTracklets12_->SetBinContent(4,13,3,1.16892);
    AlphaTracklets12_->SetBinContent(4,13,4,1.24301);
    AlphaTracklets12_->SetBinContent(4,13,5,1.23199);
    AlphaTracklets12_->SetBinContent(4,13,6,1.1597);
    AlphaTracklets12_->SetBinContent(4,13,7,1.12086);
    AlphaTracklets12_->SetBinContent(4,13,8,1.03462);
    AlphaTracklets12_->SetBinContent(4,13,9,1.15603);
    AlphaTracklets12_->SetBinContent(4,13,10,1.6);
    AlphaTracklets12_->SetBinContent(4,14,1,0);
    AlphaTracklets12_->SetBinContent(4,14,2,0);
    AlphaTracklets12_->SetBinContent(4,14,3,0);
    AlphaTracklets12_->SetBinContent(4,14,4,0);
    AlphaTracklets12_->SetBinContent(4,14,5,0);
    AlphaTracklets12_->SetBinContent(4,14,6,0);
    AlphaTracklets12_->SetBinContent(4,14,7,0);
    AlphaTracklets12_->SetBinContent(4,14,8,0);
    AlphaTracklets12_->SetBinContent(4,14,9,0);
    AlphaTracklets12_->SetBinContent(4,14,10,0);
    AlphaTracklets12_->SetBinContent(5,1,1,1.03968);
    AlphaTracklets12_->SetBinContent(5,1,2,1.05345);
    AlphaTracklets12_->SetBinContent(5,1,3,1.04633);
    AlphaTracklets12_->SetBinContent(5,1,4,1.12907);
    AlphaTracklets12_->SetBinContent(5,1,5,1.06015);
    AlphaTracklets12_->SetBinContent(5,1,6,1.03527);
    AlphaTracklets12_->SetBinContent(5,1,7,1.09402);
    AlphaTracklets12_->SetBinContent(5,1,8,1.08955);
    AlphaTracklets12_->SetBinContent(5,1,9,1.03349);
    AlphaTracklets12_->SetBinContent(5,1,10,1.10138);
    AlphaTracklets12_->SetBinContent(5,2,1,1.10817);
    AlphaTracklets12_->SetBinContent(5,2,2,1.03446);
    AlphaTracklets12_->SetBinContent(5,2,3,1.03933);
    AlphaTracklets12_->SetBinContent(5,2,4,1.10827);
    AlphaTracklets12_->SetBinContent(5,2,5,1.05962);
    AlphaTracklets12_->SetBinContent(5,2,6,1.03234);
    AlphaTracklets12_->SetBinContent(5,2,7,1.08623);
    AlphaTracklets12_->SetBinContent(5,2,8,1.08523);
    AlphaTracklets12_->SetBinContent(5,2,9,1.06888);
    AlphaTracklets12_->SetBinContent(5,2,10,1.10313);
    AlphaTracklets12_->SetBinContent(5,3,1,1.11103);
    AlphaTracklets12_->SetBinContent(5,3,2,1.04611);
    AlphaTracklets12_->SetBinContent(5,3,3,1.04251);
    AlphaTracklets12_->SetBinContent(5,3,4,1.0974);
    AlphaTracklets12_->SetBinContent(5,3,5,1.05685);
    AlphaTracklets12_->SetBinContent(5,3,6,1.03265);
    AlphaTracklets12_->SetBinContent(5,3,7,1.07989);
    AlphaTracklets12_->SetBinContent(5,3,8,1.09071);
    AlphaTracklets12_->SetBinContent(5,3,9,1.06286);
    AlphaTracklets12_->SetBinContent(5,3,10,1.08559);
    AlphaTracklets12_->SetBinContent(5,4,1,1.1403);
    AlphaTracklets12_->SetBinContent(5,4,2,1.05719);
    AlphaTracklets12_->SetBinContent(5,4,3,1.04482);
    AlphaTracklets12_->SetBinContent(5,4,4,1.10446);
    AlphaTracklets12_->SetBinContent(5,4,5,1.05641);
    AlphaTracklets12_->SetBinContent(5,4,6,1.03653);
    AlphaTracklets12_->SetBinContent(5,4,7,1.07571);
    AlphaTracklets12_->SetBinContent(5,4,8,1.10329);
    AlphaTracklets12_->SetBinContent(5,4,9,1.06268);
    AlphaTracklets12_->SetBinContent(5,4,10,1.10508);
    AlphaTracklets12_->SetBinContent(5,5,1,1.11105);
    AlphaTracklets12_->SetBinContent(5,5,2,1.04341);
    AlphaTracklets12_->SetBinContent(5,5,3,1.04372);
    AlphaTracklets12_->SetBinContent(5,5,4,1.10328);
    AlphaTracklets12_->SetBinContent(5,5,5,1.05977);
    AlphaTracklets12_->SetBinContent(5,5,6,1.03404);
    AlphaTracklets12_->SetBinContent(5,5,7,1.07557);
    AlphaTracklets12_->SetBinContent(5,5,8,1.09717);
    AlphaTracklets12_->SetBinContent(5,5,9,1.06695);
    AlphaTracklets12_->SetBinContent(5,5,10,1.06387);
    AlphaTracklets12_->SetBinContent(5,6,1,1.08211);
    AlphaTracklets12_->SetBinContent(5,6,2,1.04597);
    AlphaTracklets12_->SetBinContent(5,6,3,1.05082);
    AlphaTracklets12_->SetBinContent(5,6,4,1.10173);
    AlphaTracklets12_->SetBinContent(5,6,5,1.06579);
    AlphaTracklets12_->SetBinContent(5,6,6,1.03693);
    AlphaTracklets12_->SetBinContent(5,6,7,1.07898);
    AlphaTracklets12_->SetBinContent(5,6,8,1.10964);
    AlphaTracklets12_->SetBinContent(5,6,9,1.04831);
    AlphaTracklets12_->SetBinContent(5,6,10,1.06437);
    AlphaTracklets12_->SetBinContent(5,7,1,1.13549);
    AlphaTracklets12_->SetBinContent(5,7,2,1.07831);
    AlphaTracklets12_->SetBinContent(5,7,3,1.06233);
    AlphaTracklets12_->SetBinContent(5,7,4,1.10089);
    AlphaTracklets12_->SetBinContent(5,7,5,1.06987);
    AlphaTracklets12_->SetBinContent(5,7,6,1.04683);
    AlphaTracklets12_->SetBinContent(5,7,7,1.07656);
    AlphaTracklets12_->SetBinContent(5,7,8,1.11144);
    AlphaTracklets12_->SetBinContent(5,7,9,1.05706);
    AlphaTracklets12_->SetBinContent(5,7,10,1.06274);
    AlphaTracklets12_->SetBinContent(5,8,1,1.13355);
    AlphaTracklets12_->SetBinContent(5,8,2,1.03648);
    AlphaTracklets12_->SetBinContent(5,8,3,1.04849);
    AlphaTracklets12_->SetBinContent(5,8,4,1.11914);
    AlphaTracklets12_->SetBinContent(5,8,5,1.07882);
    AlphaTracklets12_->SetBinContent(5,8,6,1.05009);
    AlphaTracklets12_->SetBinContent(5,8,7,1.07961);
    AlphaTracklets12_->SetBinContent(5,8,8,1.1057);
    AlphaTracklets12_->SetBinContent(5,8,9,1.07644);
    AlphaTracklets12_->SetBinContent(5,8,10,1.06677);
    AlphaTracklets12_->SetBinContent(5,9,1,1.11487);
    AlphaTracklets12_->SetBinContent(5,9,2,1.08571);
    AlphaTracklets12_->SetBinContent(5,9,3,1.07066);
    AlphaTracklets12_->SetBinContent(5,9,4,1.12828);
    AlphaTracklets12_->SetBinContent(5,9,5,1.07553);
    AlphaTracklets12_->SetBinContent(5,9,6,1.05537);
    AlphaTracklets12_->SetBinContent(5,9,7,1.08956);
    AlphaTracklets12_->SetBinContent(5,9,8,1.10227);
    AlphaTracklets12_->SetBinContent(5,9,9,1.08283);
    AlphaTracklets12_->SetBinContent(5,9,10,1.0679);
    AlphaTracklets12_->SetBinContent(5,10,1,1.13219);
    AlphaTracklets12_->SetBinContent(5,10,2,1.07567);
    AlphaTracklets12_->SetBinContent(5,10,3,1.05962);
    AlphaTracklets12_->SetBinContent(5,10,4,1.14353);
    AlphaTracklets12_->SetBinContent(5,10,5,1.09577);
    AlphaTracklets12_->SetBinContent(5,10,6,1.06696);
    AlphaTracklets12_->SetBinContent(5,10,7,1.09694);
    AlphaTracklets12_->SetBinContent(5,10,8,1.12094);
    AlphaTracklets12_->SetBinContent(5,10,9,1.0803);
    AlphaTracklets12_->SetBinContent(5,10,10,1.07967);
    AlphaTracklets12_->SetBinContent(5,11,1,1.15616);
    AlphaTracklets12_->SetBinContent(5,11,2,1.11086);
    AlphaTracklets12_->SetBinContent(5,11,3,1.07053);
    AlphaTracklets12_->SetBinContent(5,11,4,1.15105);
    AlphaTracklets12_->SetBinContent(5,11,5,1.0944);
    AlphaTracklets12_->SetBinContent(5,11,6,1.06956);
    AlphaTracklets12_->SetBinContent(5,11,7,1.11397);
    AlphaTracklets12_->SetBinContent(5,11,8,1.13037);
    AlphaTracklets12_->SetBinContent(5,11,9,1.10512);
    AlphaTracklets12_->SetBinContent(5,11,10,1.11042);
    AlphaTracklets12_->SetBinContent(5,12,1,1.21461);
    AlphaTracklets12_->SetBinContent(5,12,2,1.0732);
    AlphaTracklets12_->SetBinContent(5,12,3,1.09322);
    AlphaTracklets12_->SetBinContent(5,12,4,1.1366);
    AlphaTracklets12_->SetBinContent(5,12,5,1.10895);
    AlphaTracklets12_->SetBinContent(5,12,6,1.07012);
    AlphaTracklets12_->SetBinContent(5,12,7,1.12098);
    AlphaTracklets12_->SetBinContent(5,12,8,1.1574);
    AlphaTracklets12_->SetBinContent(5,12,9,1.10957);
    AlphaTracklets12_->SetBinContent(5,12,10,1.07937);
    AlphaTracklets12_->SetBinContent(5,13,1,1.26923);
    AlphaTracklets12_->SetBinContent(5,13,2,1.08);
    AlphaTracklets12_->SetBinContent(5,13,3,1.13374);
    AlphaTracklets12_->SetBinContent(5,13,4,1.19821);
    AlphaTracklets12_->SetBinContent(5,13,5,1.1152);
    AlphaTracklets12_->SetBinContent(5,13,6,1.07795);
    AlphaTracklets12_->SetBinContent(5,13,7,1.14628);
    AlphaTracklets12_->SetBinContent(5,13,8,1.1102);
    AlphaTracklets12_->SetBinContent(5,13,9,1.04605);
    AlphaTracklets12_->SetBinContent(5,13,10,1.0303);
    AlphaTracklets12_->SetBinContent(5,14,1,0);
    AlphaTracklets12_->SetBinContent(5,14,2,0);
    AlphaTracklets12_->SetBinContent(5,14,3,0);
    AlphaTracklets12_->SetBinContent(5,14,4,0);
    AlphaTracklets12_->SetBinContent(5,14,5,0);
    AlphaTracklets12_->SetBinContent(5,14,6,0);
    AlphaTracklets12_->SetBinContent(5,14,7,0);
    AlphaTracklets12_->SetBinContent(5,14,8,0);
    AlphaTracklets12_->SetBinContent(5,14,9,0);
    AlphaTracklets12_->SetBinContent(5,14,10,0);
    AlphaTracklets12_->SetBinContent(6,1,1,1.02538);
    AlphaTracklets12_->SetBinContent(6,1,2,1.07457);
    AlphaTracklets12_->SetBinContent(6,1,3,1.20019);
    AlphaTracklets12_->SetBinContent(6,1,4,1.03629);
    AlphaTracklets12_->SetBinContent(6,1,5,1.01693);
    AlphaTracklets12_->SetBinContent(6,1,6,1.17647);
    AlphaTracklets12_->SetBinContent(6,1,7,1.09023);
    AlphaTracklets12_->SetBinContent(6,1,8,1.03759);
    AlphaTracklets12_->SetBinContent(6,1,9,1.13465);
    AlphaTracklets12_->SetBinContent(6,1,10,1.06627);
    AlphaTracklets12_->SetBinContent(6,2,1,1.01715);
    AlphaTracklets12_->SetBinContent(6,2,2,1.07756);
    AlphaTracklets12_->SetBinContent(6,2,3,1.1533);
    AlphaTracklets12_->SetBinContent(6,2,4,1.02735);
    AlphaTracklets12_->SetBinContent(6,2,5,1.01693);
    AlphaTracklets12_->SetBinContent(6,2,6,1.18795);
    AlphaTracklets12_->SetBinContent(6,2,7,1.08635);
    AlphaTracklets12_->SetBinContent(6,2,8,1.0176);
    AlphaTracklets12_->SetBinContent(6,2,9,1.08771);
    AlphaTracklets12_->SetBinContent(6,2,10,1.07654);
    AlphaTracklets12_->SetBinContent(6,3,1,1.01612);
    AlphaTracklets12_->SetBinContent(6,3,2,1.06576);
    AlphaTracklets12_->SetBinContent(6,3,3,1.14196);
    AlphaTracklets12_->SetBinContent(6,3,4,1.02305);
    AlphaTracklets12_->SetBinContent(6,3,5,1.00157);
    AlphaTracklets12_->SetBinContent(6,3,6,1.16768);
    AlphaTracklets12_->SetBinContent(6,3,7,1.07825);
    AlphaTracklets12_->SetBinContent(6,3,8,1.01489);
    AlphaTracklets12_->SetBinContent(6,3,9,1.07518);
    AlphaTracklets12_->SetBinContent(6,3,10,1.10601);
    AlphaTracklets12_->SetBinContent(6,4,1,1.00681);
    AlphaTracklets12_->SetBinContent(6,4,2,1.07045);
    AlphaTracklets12_->SetBinContent(6,4,3,1.14901);
    AlphaTracklets12_->SetBinContent(6,4,4,1.01946);
    AlphaTracklets12_->SetBinContent(6,4,5,1.00322);
    AlphaTracklets12_->SetBinContent(6,4,6,1.16512);
    AlphaTracklets12_->SetBinContent(6,4,7,1.08265);
    AlphaTracklets12_->SetBinContent(6,4,8,1.01165);
    AlphaTracklets12_->SetBinContent(6,4,9,1.09646);
    AlphaTracklets12_->SetBinContent(6,4,10,1.12964);
    AlphaTracklets12_->SetBinContent(6,5,1,0.998519);
    AlphaTracklets12_->SetBinContent(6,5,2,1.04886);
    AlphaTracklets12_->SetBinContent(6,5,3,1.14277);
    AlphaTracklets12_->SetBinContent(6,5,4,1.02615);
    AlphaTracklets12_->SetBinContent(6,5,5,0.997915);
    AlphaTracklets12_->SetBinContent(6,5,6,1.15839);
    AlphaTracklets12_->SetBinContent(6,5,7,1.07053);
    AlphaTracklets12_->SetBinContent(6,5,8,1.01179);
    AlphaTracklets12_->SetBinContent(6,5,9,1.09174);
    AlphaTracklets12_->SetBinContent(6,5,10,1.11879);
    AlphaTracklets12_->SetBinContent(6,6,1,1.0106);
    AlphaTracklets12_->SetBinContent(6,6,2,1.06176);
    AlphaTracklets12_->SetBinContent(6,6,3,1.13031);
    AlphaTracklets12_->SetBinContent(6,6,4,1.024);
    AlphaTracklets12_->SetBinContent(6,6,5,0.997452);
    AlphaTracklets12_->SetBinContent(6,6,6,1.16314);
    AlphaTracklets12_->SetBinContent(6,6,7,1.07361);
    AlphaTracklets12_->SetBinContent(6,6,8,1.0104);
    AlphaTracklets12_->SetBinContent(6,6,9,1.08648);
    AlphaTracklets12_->SetBinContent(6,6,10,1.13622);
    AlphaTracklets12_->SetBinContent(6,7,1,1.00384);
    AlphaTracklets12_->SetBinContent(6,7,2,1.05675);
    AlphaTracklets12_->SetBinContent(6,7,3,1.1286);
    AlphaTracklets12_->SetBinContent(6,7,4,1.02819);
    AlphaTracklets12_->SetBinContent(6,7,5,1.00004);
    AlphaTracklets12_->SetBinContent(6,7,6,1.16002);
    AlphaTracklets12_->SetBinContent(6,7,7,1.06537);
    AlphaTracklets12_->SetBinContent(6,7,8,1.00777);
    AlphaTracklets12_->SetBinContent(6,7,9,1.09696);
    AlphaTracklets12_->SetBinContent(6,7,10,1.10052);
    AlphaTracklets12_->SetBinContent(6,8,1,1.03113);
    AlphaTracklets12_->SetBinContent(6,8,2,1.04933);
    AlphaTracklets12_->SetBinContent(6,8,3,1.13683);
    AlphaTracklets12_->SetBinContent(6,8,4,1.02928);
    AlphaTracklets12_->SetBinContent(6,8,5,0.998406);
    AlphaTracklets12_->SetBinContent(6,8,6,1.1698);
    AlphaTracklets12_->SetBinContent(6,8,7,1.08267);
    AlphaTracklets12_->SetBinContent(6,8,8,1.01394);
    AlphaTracklets12_->SetBinContent(6,8,9,1.0941);
    AlphaTracklets12_->SetBinContent(6,8,10,1.11171);
    AlphaTracklets12_->SetBinContent(6,9,1,1.02197);
    AlphaTracklets12_->SetBinContent(6,9,2,1.07902);
    AlphaTracklets12_->SetBinContent(6,9,3,1.12756);
    AlphaTracklets12_->SetBinContent(6,9,4,1.02713);
    AlphaTracklets12_->SetBinContent(6,9,5,1.00314);
    AlphaTracklets12_->SetBinContent(6,9,6,1.16519);
    AlphaTracklets12_->SetBinContent(6,9,7,1.06929);
    AlphaTracklets12_->SetBinContent(6,9,8,1.02008);
    AlphaTracklets12_->SetBinContent(6,9,9,1.0938);
    AlphaTracklets12_->SetBinContent(6,9,10,1.12301);
    AlphaTracklets12_->SetBinContent(6,10,1,0.984718);
    AlphaTracklets12_->SetBinContent(6,10,2,1.08155);
    AlphaTracklets12_->SetBinContent(6,10,3,1.13823);
    AlphaTracklets12_->SetBinContent(6,10,4,1.03401);
    AlphaTracklets12_->SetBinContent(6,10,5,1.00468);
    AlphaTracklets12_->SetBinContent(6,10,6,1.17216);
    AlphaTracklets12_->SetBinContent(6,10,7,1.09054);
    AlphaTracklets12_->SetBinContent(6,10,8,1.02898);
    AlphaTracklets12_->SetBinContent(6,10,9,1.09892);
    AlphaTracklets12_->SetBinContent(6,10,10,1.14286);
    AlphaTracklets12_->SetBinContent(6,11,1,0.998394);
    AlphaTracklets12_->SetBinContent(6,11,2,1.08218);
    AlphaTracklets12_->SetBinContent(6,11,3,1.13267);
    AlphaTracklets12_->SetBinContent(6,11,4,1.04014);
    AlphaTracklets12_->SetBinContent(6,11,5,1.01666);
    AlphaTracklets12_->SetBinContent(6,11,6,1.17376);
    AlphaTracklets12_->SetBinContent(6,11,7,1.07127);
    AlphaTracklets12_->SetBinContent(6,11,8,1.03848);
    AlphaTracklets12_->SetBinContent(6,11,9,1.10348);
    AlphaTracklets12_->SetBinContent(6,11,10,1.12675);
    AlphaTracklets12_->SetBinContent(6,12,1,1.05515);
    AlphaTracklets12_->SetBinContent(6,12,2,1.11328);
    AlphaTracklets12_->SetBinContent(6,12,3,1.14302);
    AlphaTracklets12_->SetBinContent(6,12,4,1.03223);
    AlphaTracklets12_->SetBinContent(6,12,5,1.01058);
    AlphaTracklets12_->SetBinContent(6,12,6,1.15428);
    AlphaTracklets12_->SetBinContent(6,12,7,1.101);
    AlphaTracklets12_->SetBinContent(6,12,8,1.00769);
    AlphaTracklets12_->SetBinContent(6,12,9,1.11319);
    AlphaTracklets12_->SetBinContent(6,12,10,1.14468);
    AlphaTracklets12_->SetBinContent(6,13,1,1.25);
    AlphaTracklets12_->SetBinContent(6,13,2,0.995098);
    AlphaTracklets12_->SetBinContent(6,13,3,1.10938);
    AlphaTracklets12_->SetBinContent(6,13,4,1.01046);
    AlphaTracklets12_->SetBinContent(6,13,5,1.02681);
    AlphaTracklets12_->SetBinContent(6,13,6,1.15179);
    AlphaTracklets12_->SetBinContent(6,13,7,1.03564);
    AlphaTracklets12_->SetBinContent(6,13,8,1.07958);
    AlphaTracklets12_->SetBinContent(6,13,9,1.05488);
    AlphaTracklets12_->SetBinContent(6,13,10,0.910448);
    AlphaTracklets12_->SetBinContent(6,14,1,0);
    AlphaTracklets12_->SetBinContent(6,14,2,0);
    AlphaTracklets12_->SetBinContent(6,14,3,0);
    AlphaTracklets12_->SetBinContent(6,14,4,0);
    AlphaTracklets12_->SetBinContent(6,14,5,0);
    AlphaTracklets12_->SetBinContent(6,14,6,0);
    AlphaTracklets12_->SetBinContent(6,14,7,0);
    AlphaTracklets12_->SetBinContent(6,14,8,0);
    AlphaTracklets12_->SetBinContent(6,14,9,0);
    AlphaTracklets12_->SetBinContent(6,14,10,0);
    AlphaTracklets12_->SetBinContent(7,1,1,1.08242);
    AlphaTracklets12_->SetBinContent(7,1,2,1.15022);
    AlphaTracklets12_->SetBinContent(7,1,3,1.00658);
    AlphaTracklets12_->SetBinContent(7,1,4,1.09757);
    AlphaTracklets12_->SetBinContent(7,1,5,1.19617);
    AlphaTracklets12_->SetBinContent(7,1,6,1.03192);
    AlphaTracklets12_->SetBinContent(7,1,7,1.05936);
    AlphaTracklets12_->SetBinContent(7,1,8,1.17951);
    AlphaTracklets12_->SetBinContent(7,1,9,1.04196);
    AlphaTracklets12_->SetBinContent(7,1,10,0.995536);
    AlphaTracklets12_->SetBinContent(7,2,1,1.10448);
    AlphaTracklets12_->SetBinContent(7,2,2,1.06772);
    AlphaTracklets12_->SetBinContent(7,2,3,0.994499);
    AlphaTracklets12_->SetBinContent(7,2,4,1.07909);
    AlphaTracklets12_->SetBinContent(7,2,5,1.18577);
    AlphaTracklets12_->SetBinContent(7,2,6,1.01608);
    AlphaTracklets12_->SetBinContent(7,2,7,1.03972);
    AlphaTracklets12_->SetBinContent(7,2,8,1.15459);
    AlphaTracklets12_->SetBinContent(7,2,9,1.05456);
    AlphaTracklets12_->SetBinContent(7,2,10,1.02492);
    AlphaTracklets12_->SetBinContent(7,3,1,1.12552);
    AlphaTracklets12_->SetBinContent(7,3,2,1.08399);
    AlphaTracklets12_->SetBinContent(7,3,3,1.00918);
    AlphaTracklets12_->SetBinContent(7,3,4,1.06957);
    AlphaTracklets12_->SetBinContent(7,3,5,1.17435);
    AlphaTracklets12_->SetBinContent(7,3,6,1.01223);
    AlphaTracklets12_->SetBinContent(7,3,7,1.04135);
    AlphaTracklets12_->SetBinContent(7,3,8,1.16282);
    AlphaTracklets12_->SetBinContent(7,3,9,1.06044);
    AlphaTracklets12_->SetBinContent(7,3,10,1.01423);
    AlphaTracklets12_->SetBinContent(7,4,1,1.08318);
    AlphaTracklets12_->SetBinContent(7,4,2,1.07302);
    AlphaTracklets12_->SetBinContent(7,4,3,0.991867);
    AlphaTracklets12_->SetBinContent(7,4,4,1.06212);
    AlphaTracklets12_->SetBinContent(7,4,5,1.17793);
    AlphaTracklets12_->SetBinContent(7,4,6,1.00762);
    AlphaTracklets12_->SetBinContent(7,4,7,1.03909);
    AlphaTracklets12_->SetBinContent(7,4,8,1.15405);
    AlphaTracklets12_->SetBinContent(7,4,9,1.065);
    AlphaTracklets12_->SetBinContent(7,4,10,0.993322);
    AlphaTracklets12_->SetBinContent(7,5,1,1.10568);
    AlphaTracklets12_->SetBinContent(7,5,2,1.07694);
    AlphaTracklets12_->SetBinContent(7,5,3,0.999683);
    AlphaTracklets12_->SetBinContent(7,5,4,1.06275);
    AlphaTracklets12_->SetBinContent(7,5,5,1.16791);
    AlphaTracklets12_->SetBinContent(7,5,6,1.00979);
    AlphaTracklets12_->SetBinContent(7,5,7,1.0364);
    AlphaTracklets12_->SetBinContent(7,5,8,1.15145);
    AlphaTracklets12_->SetBinContent(7,5,9,1.05838);
    AlphaTracklets12_->SetBinContent(7,5,10,1.00598);
    AlphaTracklets12_->SetBinContent(7,6,1,1.11908);
    AlphaTracklets12_->SetBinContent(7,6,2,1.07491);
    AlphaTracklets12_->SetBinContent(7,6,3,0.984059);
    AlphaTracklets12_->SetBinContent(7,6,4,1.06128);
    AlphaTracklets12_->SetBinContent(7,6,5,1.17745);
    AlphaTracklets12_->SetBinContent(7,6,6,1.00342);
    AlphaTracklets12_->SetBinContent(7,6,7,1.03929);
    AlphaTracklets12_->SetBinContent(7,6,8,1.12591);
    AlphaTracklets12_->SetBinContent(7,6,9,1.04643);
    AlphaTracklets12_->SetBinContent(7,6,10,0.968427);
    AlphaTracklets12_->SetBinContent(7,7,1,1.12578);
    AlphaTracklets12_->SetBinContent(7,7,2,1.08219);
    AlphaTracklets12_->SetBinContent(7,7,3,0.998864);
    AlphaTracklets12_->SetBinContent(7,7,4,1.06369);
    AlphaTracklets12_->SetBinContent(7,7,5,1.16836);
    AlphaTracklets12_->SetBinContent(7,7,6,1.01264);
    AlphaTracklets12_->SetBinContent(7,7,7,1.03947);
    AlphaTracklets12_->SetBinContent(7,7,8,1.12889);
    AlphaTracklets12_->SetBinContent(7,7,9,1.06652);
    AlphaTracklets12_->SetBinContent(7,7,10,0.996441);
    AlphaTracklets12_->SetBinContent(7,8,1,1.15079);
    AlphaTracklets12_->SetBinContent(7,8,2,1.0917);
    AlphaTracklets12_->SetBinContent(7,8,3,0.994617);
    AlphaTracklets12_->SetBinContent(7,8,4,1.07047);
    AlphaTracklets12_->SetBinContent(7,8,5,1.17173);
    AlphaTracklets12_->SetBinContent(7,8,6,1.01474);
    AlphaTracklets12_->SetBinContent(7,8,7,1.04051);
    AlphaTracklets12_->SetBinContent(7,8,8,1.1378);
    AlphaTracklets12_->SetBinContent(7,8,9,1.06368);
    AlphaTracklets12_->SetBinContent(7,8,10,1.02248);
    AlphaTracklets12_->SetBinContent(7,9,1,1.11047);
    AlphaTracklets12_->SetBinContent(7,9,2,1.0903);
    AlphaTracklets12_->SetBinContent(7,9,3,0.996075);
    AlphaTracklets12_->SetBinContent(7,9,4,1.06587);
    AlphaTracklets12_->SetBinContent(7,9,5,1.16695);
    AlphaTracklets12_->SetBinContent(7,9,6,1.0196);
    AlphaTracklets12_->SetBinContent(7,9,7,1.03631);
    AlphaTracklets12_->SetBinContent(7,9,8,1.1279);
    AlphaTracklets12_->SetBinContent(7,9,9,1.05363);
    AlphaTracklets12_->SetBinContent(7,9,10,1.02012);
    AlphaTracklets12_->SetBinContent(7,10,1,1.14512);
    AlphaTracklets12_->SetBinContent(7,10,2,1.0928);
    AlphaTracklets12_->SetBinContent(7,10,3,1.0161);
    AlphaTracklets12_->SetBinContent(7,10,4,1.06554);
    AlphaTracklets12_->SetBinContent(7,10,5,1.17476);
    AlphaTracklets12_->SetBinContent(7,10,6,1.02508);
    AlphaTracklets12_->SetBinContent(7,10,7,1.05192);
    AlphaTracklets12_->SetBinContent(7,10,8,1.15893);
    AlphaTracklets12_->SetBinContent(7,10,9,1.0691);
    AlphaTracklets12_->SetBinContent(7,10,10,1.02855);
    AlphaTracklets12_->SetBinContent(7,11,1,1.11269);
    AlphaTracklets12_->SetBinContent(7,11,2,1.10422);
    AlphaTracklets12_->SetBinContent(7,11,3,1.01706);
    AlphaTracklets12_->SetBinContent(7,11,4,1.08153);
    AlphaTracklets12_->SetBinContent(7,11,5,1.16621);
    AlphaTracklets12_->SetBinContent(7,11,6,1.02948);
    AlphaTracklets12_->SetBinContent(7,11,7,1.05616);
    AlphaTracklets12_->SetBinContent(7,11,8,1.14133);
    AlphaTracklets12_->SetBinContent(7,11,9,1.08921);
    AlphaTracklets12_->SetBinContent(7,11,10,1.02017);
    AlphaTracklets12_->SetBinContent(7,12,1,1.12451);
    AlphaTracklets12_->SetBinContent(7,12,2,1.17456);
    AlphaTracklets12_->SetBinContent(7,12,3,1.04029);
    AlphaTracklets12_->SetBinContent(7,12,4,1.08417);
    AlphaTracklets12_->SetBinContent(7,12,5,1.17512);
    AlphaTracklets12_->SetBinContent(7,12,6,1.02366);
    AlphaTracklets12_->SetBinContent(7,12,7,1.04426);
    AlphaTracklets12_->SetBinContent(7,12,8,1.17058);
    AlphaTracklets12_->SetBinContent(7,12,9,1.09479);
    AlphaTracklets12_->SetBinContent(7,12,10,1.14286);
    AlphaTracklets12_->SetBinContent(7,13,1,0.971429);
    AlphaTracklets12_->SetBinContent(7,13,2,1.06593);
    AlphaTracklets12_->SetBinContent(7,13,3,1.03448);
    AlphaTracklets12_->SetBinContent(7,13,4,1.0412);
    AlphaTracklets12_->SetBinContent(7,13,5,1.19469);
    AlphaTracklets12_->SetBinContent(7,13,6,1.0628);
    AlphaTracklets12_->SetBinContent(7,13,7,1.03755);
    AlphaTracklets12_->SetBinContent(7,13,8,1.02713);
    AlphaTracklets12_->SetBinContent(7,13,9,0.9375);
    AlphaTracklets12_->SetBinContent(7,13,10,1.06579);
    AlphaTracklets12_->SetBinContent(7,14,1,0);
    AlphaTracklets12_->SetBinContent(7,14,2,0);
    AlphaTracklets12_->SetBinContent(7,14,3,0);
    AlphaTracklets12_->SetBinContent(7,14,4,0);
    AlphaTracklets12_->SetBinContent(7,14,5,0);
    AlphaTracklets12_->SetBinContent(7,14,6,0);
    AlphaTracklets12_->SetBinContent(7,14,7,0);
    AlphaTracklets12_->SetBinContent(7,14,8,0);
    AlphaTracklets12_->SetBinContent(7,14,9,0);
    AlphaTracklets12_->SetBinContent(7,14,10,0);
    AlphaTracklets12_->SetBinContent(8,1,1,1.07111);
    AlphaTracklets12_->SetBinContent(8,1,2,1.02679);
    AlphaTracklets12_->SetBinContent(8,1,3,1.14489);
    AlphaTracklets12_->SetBinContent(8,1,4,1.11852);
    AlphaTracklets12_->SetBinContent(8,1,5,1.04852);
    AlphaTracklets12_->SetBinContent(8,1,6,1.07072);
    AlphaTracklets12_->SetBinContent(8,1,7,1.12016);
    AlphaTracklets12_->SetBinContent(8,1,8,1.04786);
    AlphaTracklets12_->SetBinContent(8,1,9,1.05767);
    AlphaTracklets12_->SetBinContent(8,1,10,1.08907);
    AlphaTracklets12_->SetBinContent(8,2,1,1.08932);
    AlphaTracklets12_->SetBinContent(8,2,2,1.04674);
    AlphaTracklets12_->SetBinContent(8,2,3,1.10682);
    AlphaTracklets12_->SetBinContent(8,2,4,1.08537);
    AlphaTracklets12_->SetBinContent(8,2,5,1.04739);
    AlphaTracklets12_->SetBinContent(8,2,6,1.0629);
    AlphaTracklets12_->SetBinContent(8,2,7,1.10893);
    AlphaTracklets12_->SetBinContent(8,2,8,1.03535);
    AlphaTracklets12_->SetBinContent(8,2,9,1.04295);
    AlphaTracklets12_->SetBinContent(8,2,10,1.11099);
    AlphaTracklets12_->SetBinContent(8,3,1,1.05552);
    AlphaTracklets12_->SetBinContent(8,3,2,1.03556);
    AlphaTracklets12_->SetBinContent(8,3,3,1.10377);
    AlphaTracklets12_->SetBinContent(8,3,4,1.09258);
    AlphaTracklets12_->SetBinContent(8,3,5,1.04426);
    AlphaTracklets12_->SetBinContent(8,3,6,1.05916);
    AlphaTracklets12_->SetBinContent(8,3,7,1.11301);
    AlphaTracklets12_->SetBinContent(8,3,8,1.03306);
    AlphaTracklets12_->SetBinContent(8,3,9,1.05325);
    AlphaTracklets12_->SetBinContent(8,3,10,1.14286);
    AlphaTracklets12_->SetBinContent(8,4,1,1.06189);
    AlphaTracklets12_->SetBinContent(8,4,2,1.06608);
    AlphaTracklets12_->SetBinContent(8,4,3,1.098);
    AlphaTracklets12_->SetBinContent(8,4,4,1.09099);
    AlphaTracklets12_->SetBinContent(8,4,5,1.04039);
    AlphaTracklets12_->SetBinContent(8,4,6,1.06451);
    AlphaTracklets12_->SetBinContent(8,4,7,1.09351);
    AlphaTracklets12_->SetBinContent(8,4,8,1.04439);
    AlphaTracklets12_->SetBinContent(8,4,9,1.04888);
    AlphaTracklets12_->SetBinContent(8,4,10,1.132);
    AlphaTracklets12_->SetBinContent(8,5,1,1.04912);
    AlphaTracklets12_->SetBinContent(8,5,2,1.04578);
    AlphaTracklets12_->SetBinContent(8,5,3,1.10417);
    AlphaTracklets12_->SetBinContent(8,5,4,1.08645);
    AlphaTracklets12_->SetBinContent(8,5,5,1.04464);
    AlphaTracklets12_->SetBinContent(8,5,6,1.06493);
    AlphaTracklets12_->SetBinContent(8,5,7,1.09493);
    AlphaTracklets12_->SetBinContent(8,5,8,1.03987);
    AlphaTracklets12_->SetBinContent(8,5,9,1.04858);
    AlphaTracklets12_->SetBinContent(8,5,10,1.08446);
    AlphaTracklets12_->SetBinContent(8,6,1,1.06225);
    AlphaTracklets12_->SetBinContent(8,6,2,1.04011);
    AlphaTracklets12_->SetBinContent(8,6,3,1.10501);
    AlphaTracklets12_->SetBinContent(8,6,4,1.087);
    AlphaTracklets12_->SetBinContent(8,6,5,1.05169);
    AlphaTracklets12_->SetBinContent(8,6,6,1.0657);
    AlphaTracklets12_->SetBinContent(8,6,7,1.10245);
    AlphaTracklets12_->SetBinContent(8,6,8,1.04443);
    AlphaTracklets12_->SetBinContent(8,6,9,1.04615);
    AlphaTracklets12_->SetBinContent(8,6,10,1.09446);
    AlphaTracklets12_->SetBinContent(8,7,1,1.05135);
    AlphaTracklets12_->SetBinContent(8,7,2,1.0598);
    AlphaTracklets12_->SetBinContent(8,7,3,1.10506);
    AlphaTracklets12_->SetBinContent(8,7,4,1.10063);
    AlphaTracklets12_->SetBinContent(8,7,5,1.0603);
    AlphaTracklets12_->SetBinContent(8,7,6,1.07141);
    AlphaTracklets12_->SetBinContent(8,7,7,1.11754);
    AlphaTracklets12_->SetBinContent(8,7,8,1.05078);
    AlphaTracklets12_->SetBinContent(8,7,9,1.05538);
    AlphaTracklets12_->SetBinContent(8,7,10,1.11628);
    AlphaTracklets12_->SetBinContent(8,8,1,1.05091);
    AlphaTracklets12_->SetBinContent(8,8,2,1.05112);
    AlphaTracklets12_->SetBinContent(8,8,3,1.11001);
    AlphaTracklets12_->SetBinContent(8,8,4,1.10078);
    AlphaTracklets12_->SetBinContent(8,8,5,1.06538);
    AlphaTracklets12_->SetBinContent(8,8,6,1.07917);
    AlphaTracklets12_->SetBinContent(8,8,7,1.11231);
    AlphaTracklets12_->SetBinContent(8,8,8,1.05997);
    AlphaTracklets12_->SetBinContent(8,8,9,1.07276);
    AlphaTracklets12_->SetBinContent(8,8,10,1.14676);
    AlphaTracklets12_->SetBinContent(8,9,1,1.07318);
    AlphaTracklets12_->SetBinContent(8,9,2,1.05799);
    AlphaTracklets12_->SetBinContent(8,9,3,1.12588);
    AlphaTracklets12_->SetBinContent(8,9,4,1.10409);
    AlphaTracklets12_->SetBinContent(8,9,5,1.06051);
    AlphaTracklets12_->SetBinContent(8,9,6,1.08223);
    AlphaTracklets12_->SetBinContent(8,9,7,1.10764);
    AlphaTracklets12_->SetBinContent(8,9,8,1.0574);
    AlphaTracklets12_->SetBinContent(8,9,9,1.05926);
    AlphaTracklets12_->SetBinContent(8,9,10,1.14079);
    AlphaTracklets12_->SetBinContent(8,10,1,1.06195);
    AlphaTracklets12_->SetBinContent(8,10,2,1.07976);
    AlphaTracklets12_->SetBinContent(8,10,3,1.12242);
    AlphaTracklets12_->SetBinContent(8,10,4,1.11209);
    AlphaTracklets12_->SetBinContent(8,10,5,1.07201);
    AlphaTracklets12_->SetBinContent(8,10,6,1.09512);
    AlphaTracklets12_->SetBinContent(8,10,7,1.1293);
    AlphaTracklets12_->SetBinContent(8,10,8,1.07314);
    AlphaTracklets12_->SetBinContent(8,10,9,1.07472);
    AlphaTracklets12_->SetBinContent(8,10,10,1.11424);
    AlphaTracklets12_->SetBinContent(8,11,1,1.11589);
    AlphaTracklets12_->SetBinContent(8,11,2,1.06964);
    AlphaTracklets12_->SetBinContent(8,11,3,1.1517);
    AlphaTracklets12_->SetBinContent(8,11,4,1.12162);
    AlphaTracklets12_->SetBinContent(8,11,5,1.08614);
    AlphaTracklets12_->SetBinContent(8,11,6,1.10086);
    AlphaTracklets12_->SetBinContent(8,11,7,1.13953);
    AlphaTracklets12_->SetBinContent(8,11,8,1.08919);
    AlphaTracklets12_->SetBinContent(8,11,9,1.09997);
    AlphaTracklets12_->SetBinContent(8,11,10,1.15398);
    AlphaTracklets12_->SetBinContent(8,12,1,1.07634);
    AlphaTracklets12_->SetBinContent(8,12,2,1.04455);
    AlphaTracklets12_->SetBinContent(8,12,3,1.15277);
    AlphaTracklets12_->SetBinContent(8,12,4,1.14341);
    AlphaTracklets12_->SetBinContent(8,12,5,1.10195);
    AlphaTracklets12_->SetBinContent(8,12,6,1.0948);
    AlphaTracklets12_->SetBinContent(8,12,7,1.15698);
    AlphaTracklets12_->SetBinContent(8,12,8,1.13819);
    AlphaTracklets12_->SetBinContent(8,12,9,1.15914);
    AlphaTracklets12_->SetBinContent(8,12,10,1.18908);
    AlphaTracklets12_->SetBinContent(8,13,1,1.09677);
    AlphaTracklets12_->SetBinContent(8,13,2,1.13665);
    AlphaTracklets12_->SetBinContent(8,13,3,1.09615);
    AlphaTracklets12_->SetBinContent(8,13,4,1.13582);
    AlphaTracklets12_->SetBinContent(8,13,5,1.16216);
    AlphaTracklets12_->SetBinContent(8,13,6,1.17179);
    AlphaTracklets12_->SetBinContent(8,13,7,1.18692);
    AlphaTracklets12_->SetBinContent(8,13,8,1.02091);
    AlphaTracklets12_->SetBinContent(8,13,9,1.22609);
    AlphaTracklets12_->SetBinContent(8,13,10,1.0625);
    AlphaTracklets12_->SetBinContent(8,14,1,0);
    AlphaTracklets12_->SetBinContent(8,14,2,0);
    AlphaTracklets12_->SetBinContent(8,14,3,0);
    AlphaTracklets12_->SetBinContent(8,14,4,0);
    AlphaTracklets12_->SetBinContent(8,14,5,0);
    AlphaTracklets12_->SetBinContent(8,14,6,0);
    AlphaTracklets12_->SetBinContent(8,14,7,0);
    AlphaTracklets12_->SetBinContent(8,14,8,0);
    AlphaTracklets12_->SetBinContent(8,14,9,0);
    AlphaTracklets12_->SetBinContent(8,14,10,0);
    AlphaTracklets12_->SetBinContent(9,1,1,1.09677);
    AlphaTracklets12_->SetBinContent(9,1,2,1.06882);
    AlphaTracklets12_->SetBinContent(9,1,3,1.05899);
    AlphaTracklets12_->SetBinContent(9,1,4,1.05563);
    AlphaTracklets12_->SetBinContent(9,1,5,1.07225);
    AlphaTracklets12_->SetBinContent(9,1,6,1.06815);
    AlphaTracklets12_->SetBinContent(9,1,7,1.03101);
    AlphaTracklets12_->SetBinContent(9,1,8,1.06253);
    AlphaTracklets12_->SetBinContent(9,1,9,1.06388);
    AlphaTracklets12_->SetBinContent(9,1,10,0.996552);
    AlphaTracklets12_->SetBinContent(9,2,1,1.09692);
    AlphaTracklets12_->SetBinContent(9,2,2,1.1065);
    AlphaTracklets12_->SetBinContent(9,2,3,1.04998);
    AlphaTracklets12_->SetBinContent(9,2,4,1.03862);
    AlphaTracklets12_->SetBinContent(9,2,5,1.06662);
    AlphaTracklets12_->SetBinContent(9,2,6,1.06515);
    AlphaTracklets12_->SetBinContent(9,2,7,1.04241);
    AlphaTracklets12_->SetBinContent(9,2,8,1.06269);
    AlphaTracklets12_->SetBinContent(9,2,9,1.09562);
    AlphaTracklets12_->SetBinContent(9,2,10,1.05959);
    AlphaTracklets12_->SetBinContent(9,3,1,1.09157);
    AlphaTracklets12_->SetBinContent(9,3,2,1.06462);
    AlphaTracklets12_->SetBinContent(9,3,3,1.05445);
    AlphaTracklets12_->SetBinContent(9,3,4,1.0465);
    AlphaTracklets12_->SetBinContent(9,3,5,1.07025);
    AlphaTracklets12_->SetBinContent(9,3,6,1.07483);
    AlphaTracklets12_->SetBinContent(9,3,7,1.04564);
    AlphaTracklets12_->SetBinContent(9,3,8,1.07266);
    AlphaTracklets12_->SetBinContent(9,3,9,1.09906);
    AlphaTracklets12_->SetBinContent(9,3,10,1.08357);
    AlphaTracklets12_->SetBinContent(9,4,1,1.04094);
    AlphaTracklets12_->SetBinContent(9,4,2,1.07246);
    AlphaTracklets12_->SetBinContent(9,4,3,1.06044);
    AlphaTracklets12_->SetBinContent(9,4,4,1.05024);
    AlphaTracklets12_->SetBinContent(9,4,5,1.06945);
    AlphaTracklets12_->SetBinContent(9,4,6,1.07986);
    AlphaTracklets12_->SetBinContent(9,4,7,1.04306);
    AlphaTracklets12_->SetBinContent(9,4,8,1.08384);
    AlphaTracklets12_->SetBinContent(9,4,9,1.09875);
    AlphaTracklets12_->SetBinContent(9,4,10,1.08859);
    AlphaTracklets12_->SetBinContent(9,5,1,1.09335);
    AlphaTracklets12_->SetBinContent(9,5,2,1.08268);
    AlphaTracklets12_->SetBinContent(9,5,3,1.06192);
    AlphaTracklets12_->SetBinContent(9,5,4,1.06211);
    AlphaTracklets12_->SetBinContent(9,5,5,1.07349);
    AlphaTracklets12_->SetBinContent(9,5,6,1.08464);
    AlphaTracklets12_->SetBinContent(9,5,7,1.04966);
    AlphaTracklets12_->SetBinContent(9,5,8,1.09039);
    AlphaTracklets12_->SetBinContent(9,5,9,1.09731);
    AlphaTracklets12_->SetBinContent(9,5,10,1.06698);
    AlphaTracklets12_->SetBinContent(9,6,1,1.10106);
    AlphaTracklets12_->SetBinContent(9,6,2,1.09929);
    AlphaTracklets12_->SetBinContent(9,6,3,1.06754);
    AlphaTracklets12_->SetBinContent(9,6,4,1.06621);
    AlphaTracklets12_->SetBinContent(9,6,5,1.07646);
    AlphaTracklets12_->SetBinContent(9,6,6,1.09456);
    AlphaTracklets12_->SetBinContent(9,6,7,1.0611);
    AlphaTracklets12_->SetBinContent(9,6,8,1.10131);
    AlphaTracklets12_->SetBinContent(9,6,9,1.11028);
    AlphaTracklets12_->SetBinContent(9,6,10,1.04427);
    AlphaTracklets12_->SetBinContent(9,7,1,1.13388);
    AlphaTracklets12_->SetBinContent(9,7,2,1.08032);
    AlphaTracklets12_->SetBinContent(9,7,3,1.08117);
    AlphaTracklets12_->SetBinContent(9,7,4,1.06165);
    AlphaTracklets12_->SetBinContent(9,7,5,1.09317);
    AlphaTracklets12_->SetBinContent(9,7,6,1.09509);
    AlphaTracklets12_->SetBinContent(9,7,7,1.07887);
    AlphaTracklets12_->SetBinContent(9,7,8,1.11043);
    AlphaTracklets12_->SetBinContent(9,7,9,1.12625);
    AlphaTracklets12_->SetBinContent(9,7,10,1.08468);
    AlphaTracklets12_->SetBinContent(9,8,1,1.09064);
    AlphaTracklets12_->SetBinContent(9,8,2,1.1193);
    AlphaTracklets12_->SetBinContent(9,8,3,1.08484);
    AlphaTracklets12_->SetBinContent(9,8,4,1.08011);
    AlphaTracklets12_->SetBinContent(9,8,5,1.10207);
    AlphaTracklets12_->SetBinContent(9,8,6,1.10162);
    AlphaTracklets12_->SetBinContent(9,8,7,1.07818);
    AlphaTracklets12_->SetBinContent(9,8,8,1.10731);
    AlphaTracklets12_->SetBinContent(9,8,9,1.11347);
    AlphaTracklets12_->SetBinContent(9,8,10,1.10546);
    AlphaTracklets12_->SetBinContent(9,9,1,1.13942);
    AlphaTracklets12_->SetBinContent(9,9,2,1.10777);
    AlphaTracklets12_->SetBinContent(9,9,3,1.09614);
    AlphaTracklets12_->SetBinContent(9,9,4,1.08227);
    AlphaTracklets12_->SetBinContent(9,9,5,1.11027);
    AlphaTracklets12_->SetBinContent(9,9,6,1.11234);
    AlphaTracklets12_->SetBinContent(9,9,7,1.07852);
    AlphaTracklets12_->SetBinContent(9,9,8,1.11252);
    AlphaTracklets12_->SetBinContent(9,9,9,1.15565);
    AlphaTracklets12_->SetBinContent(9,9,10,1.10388);
    AlphaTracklets12_->SetBinContent(9,10,1,1.14955);
    AlphaTracklets12_->SetBinContent(9,10,2,1.14494);
    AlphaTracklets12_->SetBinContent(9,10,3,1.10717);
    AlphaTracklets12_->SetBinContent(9,10,4,1.10027);
    AlphaTracklets12_->SetBinContent(9,10,5,1.10968);
    AlphaTracklets12_->SetBinContent(9,10,6,1.13264);
    AlphaTracklets12_->SetBinContent(9,10,7,1.09324);
    AlphaTracklets12_->SetBinContent(9,10,8,1.12641);
    AlphaTracklets12_->SetBinContent(9,10,9,1.16363);
    AlphaTracklets12_->SetBinContent(9,10,10,1.096);
    AlphaTracklets12_->SetBinContent(9,11,1,1.12797);
    AlphaTracklets12_->SetBinContent(9,11,2,1.13494);
    AlphaTracklets12_->SetBinContent(9,11,3,1.13703);
    AlphaTracklets12_->SetBinContent(9,11,4,1.11134);
    AlphaTracklets12_->SetBinContent(9,11,5,1.13534);
    AlphaTracklets12_->SetBinContent(9,11,6,1.15693);
    AlphaTracklets12_->SetBinContent(9,11,7,1.12133);
    AlphaTracklets12_->SetBinContent(9,11,8,1.15065);
    AlphaTracklets12_->SetBinContent(9,11,9,1.16032);
    AlphaTracklets12_->SetBinContent(9,11,10,1.13808);
    AlphaTracklets12_->SetBinContent(9,12,1,1.19481);
    AlphaTracklets12_->SetBinContent(9,12,2,1.24191);
    AlphaTracklets12_->SetBinContent(9,12,3,1.09995);
    AlphaTracklets12_->SetBinContent(9,12,4,1.18381);
    AlphaTracklets12_->SetBinContent(9,12,5,1.13959);
    AlphaTracklets12_->SetBinContent(9,12,6,1.17762);
    AlphaTracklets12_->SetBinContent(9,12,7,1.14472);
    AlphaTracklets12_->SetBinContent(9,12,8,1.14823);
    AlphaTracklets12_->SetBinContent(9,12,9,1.23214);
    AlphaTracklets12_->SetBinContent(9,12,10,1.18723);
    AlphaTracklets12_->SetBinContent(9,13,1,0.931035);
    AlphaTracklets12_->SetBinContent(9,13,2,1.08947);
    AlphaTracklets12_->SetBinContent(9,13,3,1.15759);
    AlphaTracklets12_->SetBinContent(9,13,4,1.24719);
    AlphaTracklets12_->SetBinContent(9,13,5,1.23153);
    AlphaTracklets12_->SetBinContent(9,13,6,1.11553);
    AlphaTracklets12_->SetBinContent(9,13,7,1.12762);
    AlphaTracklets12_->SetBinContent(9,13,8,1.21973);
    AlphaTracklets12_->SetBinContent(9,13,9,1.23214);
    AlphaTracklets12_->SetBinContent(9,13,10,1.24194);
    AlphaTracklets12_->SetBinContent(9,14,1,0);
    AlphaTracklets12_->SetBinContent(9,14,2,0);
    AlphaTracklets12_->SetBinContent(9,14,3,0);
    AlphaTracklets12_->SetBinContent(9,14,4,0);
    AlphaTracklets12_->SetBinContent(9,14,5,0);
    AlphaTracklets12_->SetBinContent(9,14,6,0);
    AlphaTracklets12_->SetBinContent(9,14,7,0);
    AlphaTracklets12_->SetBinContent(9,14,8,0);
    AlphaTracklets12_->SetBinContent(9,14,9,0);
    AlphaTracklets12_->SetBinContent(9,14,10,0);
    AlphaTracklets12_->SetBinContent(10,1,1,1.07718);
    AlphaTracklets12_->SetBinContent(10,1,2,1.03539);
    AlphaTracklets12_->SetBinContent(10,1,3,1.02789);
    AlphaTracklets12_->SetBinContent(10,1,4,1.0483);
    AlphaTracklets12_->SetBinContent(10,1,5,1.04644);
    AlphaTracklets12_->SetBinContent(10,1,6,1.08444);
    AlphaTracklets12_->SetBinContent(10,1,7,1.22524);
    AlphaTracklets12_->SetBinContent(10,1,8,1.44781);
    AlphaTracklets12_->SetBinContent(10,1,9,1.84009);
    AlphaTracklets12_->SetBinContent(10,1,10,2.81553);
    AlphaTracklets12_->SetBinContent(10,2,1,1.08732);
    AlphaTracklets12_->SetBinContent(10,2,2,1.04435);
    AlphaTracklets12_->SetBinContent(10,2,3,1.0601);
    AlphaTracklets12_->SetBinContent(10,2,4,1.06212);
    AlphaTracklets12_->SetBinContent(10,2,5,1.05219);
    AlphaTracklets12_->SetBinContent(10,2,6,1.12516);
    AlphaTracklets12_->SetBinContent(10,2,7,1.355);
    AlphaTracklets12_->SetBinContent(10,2,8,1.65351);
    AlphaTracklets12_->SetBinContent(10,2,9,2.13444);
    AlphaTracklets12_->SetBinContent(10,2,10,4.43874);
    AlphaTracklets12_->SetBinContent(10,3,1,1.0686);
    AlphaTracklets12_->SetBinContent(10,3,2,1.05451);
    AlphaTracklets12_->SetBinContent(10,3,3,1.06589);
    AlphaTracklets12_->SetBinContent(10,3,4,1.06938);
    AlphaTracklets12_->SetBinContent(10,3,5,1.06003);
    AlphaTracklets12_->SetBinContent(10,3,6,1.14364);
    AlphaTracklets12_->SetBinContent(10,3,7,1.34928);
    AlphaTracklets12_->SetBinContent(10,3,8,1.63194);
    AlphaTracklets12_->SetBinContent(10,3,9,2.34997);
    AlphaTracklets12_->SetBinContent(10,3,10,3.92366);
    AlphaTracklets12_->SetBinContent(10,4,1,1.04755);
    AlphaTracklets12_->SetBinContent(10,4,2,1.05349);
    AlphaTracklets12_->SetBinContent(10,4,3,1.07081);
    AlphaTracklets12_->SetBinContent(10,4,4,1.07987);
    AlphaTracklets12_->SetBinContent(10,4,5,1.07844);
    AlphaTracklets12_->SetBinContent(10,4,6,1.1481);
    AlphaTracklets12_->SetBinContent(10,4,7,1.36199);
    AlphaTracklets12_->SetBinContent(10,4,8,1.67211);
    AlphaTracklets12_->SetBinContent(10,4,9,2.29663);
    AlphaTracklets12_->SetBinContent(10,4,10,3.77138);
    AlphaTracklets12_->SetBinContent(10,5,1,1.05589);
    AlphaTracklets12_->SetBinContent(10,5,2,1.05474);
    AlphaTracklets12_->SetBinContent(10,5,3,1.08443);
    AlphaTracklets12_->SetBinContent(10,5,4,1.09101);
    AlphaTracklets12_->SetBinContent(10,5,5,1.08539);
    AlphaTracklets12_->SetBinContent(10,5,6,1.15743);
    AlphaTracklets12_->SetBinContent(10,5,7,1.38345);
    AlphaTracklets12_->SetBinContent(10,5,8,1.69208);
    AlphaTracklets12_->SetBinContent(10,5,9,2.39015);
    AlphaTracklets12_->SetBinContent(10,5,10,4.09405);
    AlphaTracklets12_->SetBinContent(10,6,1,1.09661);
    AlphaTracklets12_->SetBinContent(10,6,2,1.07071);
    AlphaTracklets12_->SetBinContent(10,6,3,1.09161);
    AlphaTracklets12_->SetBinContent(10,6,4,1.10001);
    AlphaTracklets12_->SetBinContent(10,6,5,1.09338);
    AlphaTracklets12_->SetBinContent(10,6,6,1.17736);
    AlphaTracklets12_->SetBinContent(10,6,7,1.39178);
    AlphaTracklets12_->SetBinContent(10,6,8,1.69769);
    AlphaTracklets12_->SetBinContent(10,6,9,2.48067);
    AlphaTracklets12_->SetBinContent(10,6,10,4.232);
    AlphaTracklets12_->SetBinContent(10,7,1,1.02008);
    AlphaTracklets12_->SetBinContent(10,7,2,1.0797);
    AlphaTracklets12_->SetBinContent(10,7,3,1.09628);
    AlphaTracklets12_->SetBinContent(10,7,4,1.10795);
    AlphaTracklets12_->SetBinContent(10,7,5,1.09919);
    AlphaTracklets12_->SetBinContent(10,7,6,1.18089);
    AlphaTracklets12_->SetBinContent(10,7,7,1.40635);
    AlphaTracklets12_->SetBinContent(10,7,8,1.72893);
    AlphaTracklets12_->SetBinContent(10,7,9,2.43341);
    AlphaTracklets12_->SetBinContent(10,7,10,4.33958);
    AlphaTracklets12_->SetBinContent(10,8,1,1.12394);
    AlphaTracklets12_->SetBinContent(10,8,2,1.10403);
    AlphaTracklets12_->SetBinContent(10,8,3,1.12668);
    AlphaTracklets12_->SetBinContent(10,8,4,1.11095);
    AlphaTracklets12_->SetBinContent(10,8,5,1.12251);
    AlphaTracklets12_->SetBinContent(10,8,6,1.20301);
    AlphaTracklets12_->SetBinContent(10,8,7,1.43547);
    AlphaTracklets12_->SetBinContent(10,8,8,1.74655);
    AlphaTracklets12_->SetBinContent(10,8,9,2.3169);
    AlphaTracklets12_->SetBinContent(10,8,10,4.6497);
    AlphaTracklets12_->SetBinContent(10,9,1,1.07753);
    AlphaTracklets12_->SetBinContent(10,9,2,1.09929);
    AlphaTracklets12_->SetBinContent(10,9,3,1.12757);
    AlphaTracklets12_->SetBinContent(10,9,4,1.14538);
    AlphaTracklets12_->SetBinContent(10,9,5,1.14163);
    AlphaTracklets12_->SetBinContent(10,9,6,1.20944);
    AlphaTracklets12_->SetBinContent(10,9,7,1.422);
    AlphaTracklets12_->SetBinContent(10,9,8,1.77238);
    AlphaTracklets12_->SetBinContent(10,9,9,2.31382);
    AlphaTracklets12_->SetBinContent(10,9,10,4.42336);
    AlphaTracklets12_->SetBinContent(10,10,1,1.16989);
    AlphaTracklets12_->SetBinContent(10,10,2,1.14935);
    AlphaTracklets12_->SetBinContent(10,10,3,1.15353);
    AlphaTracklets12_->SetBinContent(10,10,4,1.16217);
    AlphaTracklets12_->SetBinContent(10,10,5,1.16321);
    AlphaTracklets12_->SetBinContent(10,10,6,1.24102);
    AlphaTracklets12_->SetBinContent(10,10,7,1.45288);
    AlphaTracklets12_->SetBinContent(10,10,8,1.75328);
    AlphaTracklets12_->SetBinContent(10,10,9,2.36414);
    AlphaTracklets12_->SetBinContent(10,10,10,4.72258);
    AlphaTracklets12_->SetBinContent(10,11,1,1.1267);
    AlphaTracklets12_->SetBinContent(10,11,2,1.14799);
    AlphaTracklets12_->SetBinContent(10,11,3,1.17601);
    AlphaTracklets12_->SetBinContent(10,11,4,1.19168);
    AlphaTracklets12_->SetBinContent(10,11,5,1.19289);
    AlphaTracklets12_->SetBinContent(10,11,6,1.25131);
    AlphaTracklets12_->SetBinContent(10,11,7,1.49766);
    AlphaTracklets12_->SetBinContent(10,11,8,1.84404);
    AlphaTracklets12_->SetBinContent(10,11,9,2.55432);
    AlphaTracklets12_->SetBinContent(10,11,10,4.02059);
    AlphaTracklets12_->SetBinContent(10,12,1,1.20325);
    AlphaTracklets12_->SetBinContent(10,12,2,1.21282);
    AlphaTracklets12_->SetBinContent(10,12,3,1.19436);
    AlphaTracklets12_->SetBinContent(10,12,4,1.25221);
    AlphaTracklets12_->SetBinContent(10,12,5,1.18397);
    AlphaTracklets12_->SetBinContent(10,12,6,1.30522);
    AlphaTracklets12_->SetBinContent(10,12,7,1.61572);
    AlphaTracklets12_->SetBinContent(10,12,8,1.81381);
    AlphaTracklets12_->SetBinContent(10,12,9,2.46579);
    AlphaTracklets12_->SetBinContent(10,12,10,3.87143);
    AlphaTracklets12_->SetBinContent(10,13,1,1.25);
    AlphaTracklets12_->SetBinContent(10,13,2,1.13095);
    AlphaTracklets12_->SetBinContent(10,13,3,1.25753);
    AlphaTracklets12_->SetBinContent(10,13,4,1.2998);
    AlphaTracklets12_->SetBinContent(10,13,5,1.18151);
    AlphaTracklets12_->SetBinContent(10,13,6,1.39286);
    AlphaTracklets12_->SetBinContent(10,13,7,1.70383);
    AlphaTracklets12_->SetBinContent(10,13,8,1.92308);
    AlphaTracklets12_->SetBinContent(10,13,9,3.4878);
    AlphaTracklets12_->SetBinContent(10,13,10,3.25);
    AlphaTracklets12_->SetBinContent(10,14,1,0);
    AlphaTracklets12_->SetBinContent(10,14,2,0);
    AlphaTracklets12_->SetBinContent(10,14,3,0);
    AlphaTracklets12_->SetBinContent(10,14,4,0);
    AlphaTracklets12_->SetBinContent(10,14,5,0);
    AlphaTracklets12_->SetBinContent(10,14,6,0);
    AlphaTracklets12_->SetBinContent(10,14,7,0);
    AlphaTracklets12_->SetBinContent(10,14,8,0);
    AlphaTracklets12_->SetBinContent(10,14,9,0);
    AlphaTracklets12_->SetBinContent(10,14,10,0);
    AlphaTracklets12_->SetBinContent(11,1,1,1.5561);
    AlphaTracklets12_->SetBinContent(11,1,2,1.73418);
    AlphaTracklets12_->SetBinContent(11,1,3,2.29621);
    AlphaTracklets12_->SetBinContent(11,1,4,3.29634);
    AlphaTracklets12_->SetBinContent(11,2,1,1.95027);
    AlphaTracklets12_->SetBinContent(11,2,2,2.24508);
    AlphaTracklets12_->SetBinContent(11,2,3,2.88022);
    AlphaTracklets12_->SetBinContent(11,2,4,4.63164);
    AlphaTracklets12_->SetBinContent(11,3,1,1.79464);
    AlphaTracklets12_->SetBinContent(11,3,2,2.18855);
    AlphaTracklets12_->SetBinContent(11,3,3,3.00151);
    AlphaTracklets12_->SetBinContent(11,3,4,4.76541);
    AlphaTracklets12_->SetBinContent(11,4,1,1.84064);
    AlphaTracklets12_->SetBinContent(11,4,2,2.34987);
    AlphaTracklets12_->SetBinContent(11,4,3,3.03066);
    AlphaTracklets12_->SetBinContent(11,4,4,4.80677);
    AlphaTracklets12_->SetBinContent(11,5,1,1.88529);
    AlphaTracklets12_->SetBinContent(11,5,2,2.31713);
    AlphaTracklets12_->SetBinContent(11,5,3,3.05414);
    AlphaTracklets12_->SetBinContent(11,5,4,4.74661);
    AlphaTracklets12_->SetBinContent(11,6,1,1.80494);
    AlphaTracklets12_->SetBinContent(11,6,2,2.27132);
    AlphaTracklets12_->SetBinContent(11,6,3,3.16418);
    AlphaTracklets12_->SetBinContent(11,6,4,4.83848);
    AlphaTracklets12_->SetBinContent(11,7,1,1.97297);
    AlphaTracklets12_->SetBinContent(11,7,2,2.38787);
    AlphaTracklets12_->SetBinContent(11,7,3,3.12309);
    AlphaTracklets12_->SetBinContent(11,7,4,4.76581);
    AlphaTracklets12_->SetBinContent(11,8,1,1.91539);
    AlphaTracklets12_->SetBinContent(11,8,2,2.298);
    AlphaTracklets12_->SetBinContent(11,8,3,3.19482);
    AlphaTracklets12_->SetBinContent(11,8,4,4.64911);
    AlphaTracklets12_->SetBinContent(11,9,1,1.94049);
    AlphaTracklets12_->SetBinContent(11,9,2,2.39413);
    AlphaTracklets12_->SetBinContent(11,9,3,3.30116);
    AlphaTracklets12_->SetBinContent(11,9,4,5.11738);
    AlphaTracklets12_->SetBinContent(11,10,1,2.11352);
    AlphaTracklets12_->SetBinContent(11,10,2,2.44885);
    AlphaTracklets12_->SetBinContent(11,10,3,3.36159);
    AlphaTracklets12_->SetBinContent(11,10,4,5.06106);
    AlphaTracklets12_->SetBinContent(11,11,1,2.29703);
    AlphaTracklets12_->SetBinContent(11,11,2,2.60388);
    AlphaTracklets12_->SetBinContent(11,11,3,3.38887);
    AlphaTracklets12_->SetBinContent(11,11,4,5.27353);
    AlphaTracklets12_->SetBinContent(11,12,1,2.19853);
    AlphaTracklets12_->SetBinContent(11,12,2,2.81065);
    AlphaTracklets12_->SetBinContent(11,12,3,3.69888);
    AlphaTracklets12_->SetBinContent(11,12,4,5.13475);
    AlphaTracklets12_->SetBinContent(11,13,1,1.29412);
    AlphaTracklets12_->SetBinContent(11,13,2,4.08929);
    AlphaTracklets12_->SetBinContent(11,13,3,3.92308);
    AlphaTracklets12_->SetBinContent(11,13,4,5.208);
    AlphaTracklets12_->SetBinContent(11,14,1,0);
    AlphaTracklets12_->SetBinContent(11,14,2,0);
    AlphaTracklets12_->SetBinContent(11,14,3,0);
    AlphaTracklets12_->SetBinContent(11,14,4,0);
  }

  if (which>=13) {
    const int nEtaBin = 12;
    const int nHitBin = 14;
    const int nVzBin  = 10;

    double HitBins[nHitBin+1] = {0,5,10,15,20,25,30,35,40,50,60,80,100,200,700};

    double EtaBins[nEtaBin+1];
    for (int i=0;i<=nEtaBin;i++)
      EtaBins[i] = (double)i*6.0/(double)nEtaBin-3.0;
    double VzBins[nVzBin+1];
    for (int i=0;i<=nVzBin;i++)
      VzBins[i] = (double)i*20.0/(double)nVzBin-10.0;

    AlphaTracklets13_ = new TH3F("hAlphaTracklets13",
                                 "Alpha for tracklets13;#eta;#hits;vz [cm]",
                                 nEtaBin, EtaBins, nHitBin, HitBins, nVzBin, VzBins);
    AlphaTracklets13_->SetDirectory(0);

    AlphaTracklets13_->SetBinContent(3,1,5,3.29862);
    AlphaTracklets13_->SetBinContent(3,1,6,2.40246);
    AlphaTracklets13_->SetBinContent(3,1,7,1.92316);
    AlphaTracklets13_->SetBinContent(3,1,8,1.67219);
    AlphaTracklets13_->SetBinContent(3,1,9,1.38176);
    AlphaTracklets13_->SetBinContent(3,1,10,1.14241);
    AlphaTracklets13_->SetBinContent(3,2,5,3.43827);
    AlphaTracklets13_->SetBinContent(3,2,6,2.38749);
    AlphaTracklets13_->SetBinContent(3,2,7,1.95897);
    AlphaTracklets13_->SetBinContent(3,2,8,1.6392);
    AlphaTracklets13_->SetBinContent(3,2,9,1.37689);
    AlphaTracklets13_->SetBinContent(3,2,10,1.1899);
    AlphaTracklets13_->SetBinContent(3,3,5,3.52871);
    AlphaTracklets13_->SetBinContent(3,3,6,2.45737);
    AlphaTracklets13_->SetBinContent(3,3,7,1.99885);
    AlphaTracklets13_->SetBinContent(3,3,8,1.63279);
    AlphaTracklets13_->SetBinContent(3,3,9,1.40344);
    AlphaTracklets13_->SetBinContent(3,3,10,1.28988);
    AlphaTracklets13_->SetBinContent(3,4,5,3.54365);
    AlphaTracklets13_->SetBinContent(3,4,6,2.53683);
    AlphaTracklets13_->SetBinContent(3,4,7,2.02824);
    AlphaTracklets13_->SetBinContent(3,4,8,1.6926);
    AlphaTracklets13_->SetBinContent(3,4,9,1.45197);
    AlphaTracklets13_->SetBinContent(3,4,10,1.30969);
    AlphaTracklets13_->SetBinContent(3,5,5,3.71725);
    AlphaTracklets13_->SetBinContent(3,5,6,2.52003);
    AlphaTracklets13_->SetBinContent(3,5,7,2.09456);
    AlphaTracklets13_->SetBinContent(3,5,8,1.72218);
    AlphaTracklets13_->SetBinContent(3,5,9,1.48957);
    AlphaTracklets13_->SetBinContent(3,5,10,1.30625);
    AlphaTracklets13_->SetBinContent(3,6,5,3.64208);
    AlphaTracklets13_->SetBinContent(3,6,6,2.63515);
    AlphaTracklets13_->SetBinContent(3,6,7,2.08812);
    AlphaTracklets13_->SetBinContent(3,6,8,1.78835);
    AlphaTracklets13_->SetBinContent(3,6,9,1.52661);
    AlphaTracklets13_->SetBinContent(3,6,10,1.34023);
    AlphaTracklets13_->SetBinContent(3,7,5,3.76728);
    AlphaTracklets13_->SetBinContent(3,7,6,2.60854);
    AlphaTracklets13_->SetBinContent(3,7,7,2.15041);
    AlphaTracklets13_->SetBinContent(3,7,8,1.7965);
    AlphaTracklets13_->SetBinContent(3,7,9,1.50416);
    AlphaTracklets13_->SetBinContent(3,7,10,1.33587);
    AlphaTracklets13_->SetBinContent(3,8,5,3.78679);
    AlphaTracklets13_->SetBinContent(3,8,6,2.77262);
    AlphaTracklets13_->SetBinContent(3,8,7,2.16645);
    AlphaTracklets13_->SetBinContent(3,8,8,1.8244);
    AlphaTracklets13_->SetBinContent(3,8,9,1.59099);
    AlphaTracklets13_->SetBinContent(3,8,10,1.3763);
    AlphaTracklets13_->SetBinContent(3,9,5,3.9184);
    AlphaTracklets13_->SetBinContent(3,9,6,2.74861);
    AlphaTracklets13_->SetBinContent(3,9,7,2.2024);
    AlphaTracklets13_->SetBinContent(3,9,8,1.8454);
    AlphaTracklets13_->SetBinContent(3,9,9,1.58966);
    AlphaTracklets13_->SetBinContent(3,9,10,1.41312);
    AlphaTracklets13_->SetBinContent(3,10,5,3.9293);
    AlphaTracklets13_->SetBinContent(3,10,6,2.86661);
    AlphaTracklets13_->SetBinContent(3,10,7,2.2829);
    AlphaTracklets13_->SetBinContent(3,10,8,1.92525);
    AlphaTracklets13_->SetBinContent(3,10,9,1.59415);
    AlphaTracklets13_->SetBinContent(3,10,10,1.44757);
    AlphaTracklets13_->SetBinContent(3,11,5,4.1735);
    AlphaTracklets13_->SetBinContent(3,11,6,2.86711);
    AlphaTracklets13_->SetBinContent(3,11,7,2.37098);
    AlphaTracklets13_->SetBinContent(3,11,8,1.98502);
    AlphaTracklets13_->SetBinContent(3,11,9,1.70666);
    AlphaTracklets13_->SetBinContent(3,11,10,1.50256);
    AlphaTracklets13_->SetBinContent(3,12,5,4.10982);
    AlphaTracklets13_->SetBinContent(3,12,6,2.96355);
    AlphaTracklets13_->SetBinContent(3,12,7,2.40535);
    AlphaTracklets13_->SetBinContent(3,12,8,2.08604);
    AlphaTracklets13_->SetBinContent(3,12,9,1.90506);
    AlphaTracklets13_->SetBinContent(3,12,10,1.64815);
    AlphaTracklets13_->SetBinContent(3,13,5,5.08759);
    AlphaTracklets13_->SetBinContent(3,13,6,2.97039);
    AlphaTracklets13_->SetBinContent(3,13,7,2.44828);
    AlphaTracklets13_->SetBinContent(3,13,8,2.45614);
    AlphaTracklets13_->SetBinContent(3,13,9,2.375);
    AlphaTracklets13_->SetBinContent(3,13,10,1.18966);
    AlphaTracklets13_->SetBinContent(3,14,5,0);
    AlphaTracklets13_->SetBinContent(3,14,6,0);
    AlphaTracklets13_->SetBinContent(3,14,7,0);
    AlphaTracklets13_->SetBinContent(3,14,8,0);
    AlphaTracklets13_->SetBinContent(3,14,9,0);
    AlphaTracklets13_->SetBinContent(3,14,10,0);
    AlphaTracklets13_->SetBinContent(4,1,1,1.67876);
    AlphaTracklets13_->SetBinContent(4,1,2,1.36252);
    AlphaTracklets13_->SetBinContent(4,1,3,1.16882);
    AlphaTracklets13_->SetBinContent(4,1,4,1.0816);
    AlphaTracklets13_->SetBinContent(4,1,5,1.14216);
    AlphaTracklets13_->SetBinContent(4,1,6,1.16257);
    AlphaTracklets13_->SetBinContent(4,1,7,1.09451);
    AlphaTracklets13_->SetBinContent(4,1,8,1.09256);
    AlphaTracklets13_->SetBinContent(4,1,9,1.14326);
    AlphaTracklets13_->SetBinContent(4,1,10,1.1617);
    AlphaTracklets13_->SetBinContent(4,2,1,1.72509);
    AlphaTracklets13_->SetBinContent(4,2,2,1.40058);
    AlphaTracklets13_->SetBinContent(4,2,3,1.16613);
    AlphaTracklets13_->SetBinContent(4,2,4,1.09416);
    AlphaTracklets13_->SetBinContent(4,2,5,1.16571);
    AlphaTracklets13_->SetBinContent(4,2,6,1.16677);
    AlphaTracklets13_->SetBinContent(4,2,7,1.10047);
    AlphaTracklets13_->SetBinContent(4,2,8,1.10909);
    AlphaTracklets13_->SetBinContent(4,2,9,1.17779);
    AlphaTracklets13_->SetBinContent(4,2,10,1.16207);
    AlphaTracklets13_->SetBinContent(4,3,1,1.75489);
    AlphaTracklets13_->SetBinContent(4,3,2,1.43025);
    AlphaTracklets13_->SetBinContent(4,3,3,1.16605);
    AlphaTracklets13_->SetBinContent(4,3,4,1.10137);
    AlphaTracklets13_->SetBinContent(4,3,5,1.16644);
    AlphaTracklets13_->SetBinContent(4,3,6,1.16621);
    AlphaTracklets13_->SetBinContent(4,3,7,1.12365);
    AlphaTracklets13_->SetBinContent(4,3,8,1.11707);
    AlphaTracklets13_->SetBinContent(4,3,9,1.17172);
    AlphaTracklets13_->SetBinContent(4,3,10,1.15789);
    AlphaTracklets13_->SetBinContent(4,4,1,1.68203);
    AlphaTracklets13_->SetBinContent(4,4,2,1.4);
    AlphaTracklets13_->SetBinContent(4,4,3,1.17561);
    AlphaTracklets13_->SetBinContent(4,4,4,1.12521);
    AlphaTracklets13_->SetBinContent(4,4,5,1.18526);
    AlphaTracklets13_->SetBinContent(4,4,6,1.18121);
    AlphaTracklets13_->SetBinContent(4,4,7,1.14095);
    AlphaTracklets13_->SetBinContent(4,4,8,1.13519);
    AlphaTracklets13_->SetBinContent(4,4,9,1.20293);
    AlphaTracklets13_->SetBinContent(4,4,10,1.13745);
    AlphaTracklets13_->SetBinContent(4,5,1,1.74336);
    AlphaTracklets13_->SetBinContent(4,5,2,1.44483);
    AlphaTracklets13_->SetBinContent(4,5,3,1.21029);
    AlphaTracklets13_->SetBinContent(4,5,4,1.14142);
    AlphaTracklets13_->SetBinContent(4,5,5,1.20144);
    AlphaTracklets13_->SetBinContent(4,5,6,1.19785);
    AlphaTracklets13_->SetBinContent(4,5,7,1.14935);
    AlphaTracklets13_->SetBinContent(4,5,8,1.15311);
    AlphaTracklets13_->SetBinContent(4,5,9,1.17249);
    AlphaTracklets13_->SetBinContent(4,5,10,1.13769);
    AlphaTracklets13_->SetBinContent(4,6,1,1.70165);
    AlphaTracklets13_->SetBinContent(4,6,2,1.41489);
    AlphaTracklets13_->SetBinContent(4,6,3,1.23942);
    AlphaTracklets13_->SetBinContent(4,6,4,1.16798);
    AlphaTracklets13_->SetBinContent(4,6,5,1.21249);
    AlphaTracklets13_->SetBinContent(4,6,6,1.21942);
    AlphaTracklets13_->SetBinContent(4,6,7,1.16178);
    AlphaTracklets13_->SetBinContent(4,6,8,1.16173);
    AlphaTracklets13_->SetBinContent(4,6,9,1.21148);
    AlphaTracklets13_->SetBinContent(4,6,10,1.23168);
    AlphaTracklets13_->SetBinContent(4,7,1,1.78947);
    AlphaTracklets13_->SetBinContent(4,7,2,1.45113);
    AlphaTracklets13_->SetBinContent(4,7,3,1.23975);
    AlphaTracklets13_->SetBinContent(4,7,4,1.17634);
    AlphaTracklets13_->SetBinContent(4,7,5,1.23682);
    AlphaTracklets13_->SetBinContent(4,7,6,1.23532);
    AlphaTracklets13_->SetBinContent(4,7,7,1.17839);
    AlphaTracklets13_->SetBinContent(4,7,8,1.17726);
    AlphaTracklets13_->SetBinContent(4,7,9,1.2361);
    AlphaTracklets13_->SetBinContent(4,7,10,1.22043);
    AlphaTracklets13_->SetBinContent(4,8,1,1.83141);
    AlphaTracklets13_->SetBinContent(4,8,2,1.45854);
    AlphaTracklets13_->SetBinContent(4,8,3,1.26022);
    AlphaTracklets13_->SetBinContent(4,8,4,1.20054);
    AlphaTracklets13_->SetBinContent(4,8,5,1.26646);
    AlphaTracklets13_->SetBinContent(4,8,6,1.25187);
    AlphaTracklets13_->SetBinContent(4,8,7,1.20082);
    AlphaTracklets13_->SetBinContent(4,8,8,1.17525);
    AlphaTracklets13_->SetBinContent(4,8,9,1.21659);
    AlphaTracklets13_->SetBinContent(4,8,10,1.20317);
    AlphaTracklets13_->SetBinContent(4,9,1,1.79823);
    AlphaTracklets13_->SetBinContent(4,9,2,1.50106);
    AlphaTracklets13_->SetBinContent(4,9,3,1.2671);
    AlphaTracklets13_->SetBinContent(4,9,4,1.20316);
    AlphaTracklets13_->SetBinContent(4,9,5,1.28459);
    AlphaTracklets13_->SetBinContent(4,9,6,1.2745);
    AlphaTracklets13_->SetBinContent(4,9,7,1.21884);
    AlphaTracklets13_->SetBinContent(4,9,8,1.22976);
    AlphaTracklets13_->SetBinContent(4,9,9,1.25927);
    AlphaTracklets13_->SetBinContent(4,9,10,1.22273);
    AlphaTracklets13_->SetBinContent(4,10,1,1.88005);
    AlphaTracklets13_->SetBinContent(4,10,2,1.57312);
    AlphaTracklets13_->SetBinContent(4,10,3,1.28794);
    AlphaTracklets13_->SetBinContent(4,10,4,1.22969);
    AlphaTracklets13_->SetBinContent(4,10,5,1.2826);
    AlphaTracklets13_->SetBinContent(4,10,6,1.29977);
    AlphaTracklets13_->SetBinContent(4,10,7,1.23641);
    AlphaTracklets13_->SetBinContent(4,10,8,1.25106);
    AlphaTracklets13_->SetBinContent(4,10,9,1.30843);
    AlphaTracklets13_->SetBinContent(4,10,10,1.20542);
    AlphaTracklets13_->SetBinContent(4,11,1,2.02117);
    AlphaTracklets13_->SetBinContent(4,11,2,1.64472);
    AlphaTracklets13_->SetBinContent(4,11,3,1.33357);
    AlphaTracklets13_->SetBinContent(4,11,4,1.27954);
    AlphaTracklets13_->SetBinContent(4,11,5,1.34022);
    AlphaTracklets13_->SetBinContent(4,11,6,1.3308);
    AlphaTracklets13_->SetBinContent(4,11,7,1.26338);
    AlphaTracklets13_->SetBinContent(4,11,8,1.30447);
    AlphaTracklets13_->SetBinContent(4,11,9,1.36376);
    AlphaTracklets13_->SetBinContent(4,11,10,1.24104);
    AlphaTracklets13_->SetBinContent(4,12,1,1.76923);
    AlphaTracklets13_->SetBinContent(4,12,2,1.76458);
    AlphaTracklets13_->SetBinContent(4,12,3,1.48578);
    AlphaTracklets13_->SetBinContent(4,12,4,1.28782);
    AlphaTracklets13_->SetBinContent(4,12,5,1.37776);
    AlphaTracklets13_->SetBinContent(4,12,6,1.33313);
    AlphaTracklets13_->SetBinContent(4,12,7,1.36401);
    AlphaTracklets13_->SetBinContent(4,12,8,1.3033);
    AlphaTracklets13_->SetBinContent(4,12,9,1.47766);
    AlphaTracklets13_->SetBinContent(4,12,10,1.46269);
    AlphaTracklets13_->SetBinContent(4,13,1,1.6);
    AlphaTracklets13_->SetBinContent(4,13,2,1.68504);
    AlphaTracklets13_->SetBinContent(4,13,3,1.62441);
    AlphaTracklets13_->SetBinContent(4,13,4,1.39961);
    AlphaTracklets13_->SetBinContent(4,13,5,1.50429);
    AlphaTracklets13_->SetBinContent(4,13,6,1.44094);
    AlphaTracklets13_->SetBinContent(4,13,7,1.28635);
    AlphaTracklets13_->SetBinContent(4,13,8,1.25701);
    AlphaTracklets13_->SetBinContent(4,13,9,1.71579);
    AlphaTracklets13_->SetBinContent(4,13,10,1.33333);
    AlphaTracklets13_->SetBinContent(4,14,1,0);
    AlphaTracklets13_->SetBinContent(4,14,2,0);
    AlphaTracklets13_->SetBinContent(4,14,3,0);
    AlphaTracklets13_->SetBinContent(4,14,4,0);
    AlphaTracklets13_->SetBinContent(4,14,5,0);
    AlphaTracklets13_->SetBinContent(4,14,6,0);
    AlphaTracklets13_->SetBinContent(4,14,7,0);
    AlphaTracklets13_->SetBinContent(4,14,8,0);
    AlphaTracklets13_->SetBinContent(4,14,9,0);
    AlphaTracklets13_->SetBinContent(4,14,10,0);
    AlphaTracklets13_->SetBinContent(5,1,1,1.08714);
    AlphaTracklets13_->SetBinContent(5,1,2,1.10147);
    AlphaTracklets13_->SetBinContent(5,1,3,1.09865);
    AlphaTracklets13_->SetBinContent(5,1,4,1.16421);
    AlphaTracklets13_->SetBinContent(5,1,5,1.07683);
    AlphaTracklets13_->SetBinContent(5,1,6,1.07687);
    AlphaTracklets13_->SetBinContent(5,1,7,1.15315);
    AlphaTracklets13_->SetBinContent(5,1,8,1.13266);
    AlphaTracklets13_->SetBinContent(5,1,9,1.06426);
    AlphaTracklets13_->SetBinContent(5,1,10,1.14904);
    AlphaTracklets13_->SetBinContent(5,2,1,1.141);
    AlphaTracklets13_->SetBinContent(5,2,2,1.07771);
    AlphaTracklets13_->SetBinContent(5,2,3,1.104);
    AlphaTracklets13_->SetBinContent(5,2,4,1.16985);
    AlphaTracklets13_->SetBinContent(5,2,5,1.09235);
    AlphaTracklets13_->SetBinContent(5,2,6,1.06868);
    AlphaTracklets13_->SetBinContent(5,2,7,1.15589);
    AlphaTracklets13_->SetBinContent(5,2,8,1.10982);
    AlphaTracklets13_->SetBinContent(5,2,9,1.09445);
    AlphaTracklets13_->SetBinContent(5,2,10,1.15674);
    AlphaTracklets13_->SetBinContent(5,3,1,1.17057);
    AlphaTracklets13_->SetBinContent(5,3,2,1.10973);
    AlphaTracklets13_->SetBinContent(5,3,3,1.11988);
    AlphaTracklets13_->SetBinContent(5,3,4,1.1673);
    AlphaTracklets13_->SetBinContent(5,3,5,1.09309);
    AlphaTracklets13_->SetBinContent(5,3,6,1.07688);
    AlphaTracklets13_->SetBinContent(5,3,7,1.16415);
    AlphaTracklets13_->SetBinContent(5,3,8,1.12392);
    AlphaTracklets13_->SetBinContent(5,3,9,1.08627);
    AlphaTracklets13_->SetBinContent(5,3,10,1.16301);
    AlphaTracklets13_->SetBinContent(5,4,1,1.18634);
    AlphaTracklets13_->SetBinContent(5,4,2,1.11459);
    AlphaTracklets13_->SetBinContent(5,4,3,1.12524);
    AlphaTracklets13_->SetBinContent(5,4,4,1.17163);
    AlphaTracklets13_->SetBinContent(5,4,5,1.10151);
    AlphaTracklets13_->SetBinContent(5,4,6,1.09382);
    AlphaTracklets13_->SetBinContent(5,4,7,1.16025);
    AlphaTracklets13_->SetBinContent(5,4,8,1.13634);
    AlphaTracklets13_->SetBinContent(5,4,9,1.09382);
    AlphaTracklets13_->SetBinContent(5,4,10,1.16776);
    AlphaTracklets13_->SetBinContent(5,5,1,1.18713);
    AlphaTracklets13_->SetBinContent(5,5,2,1.12313);
    AlphaTracklets13_->SetBinContent(5,5,3,1.13668);
    AlphaTracklets13_->SetBinContent(5,5,4,1.18504);
    AlphaTracklets13_->SetBinContent(5,5,5,1.10726);
    AlphaTracklets13_->SetBinContent(5,5,6,1.0955);
    AlphaTracklets13_->SetBinContent(5,5,7,1.16678);
    AlphaTracklets13_->SetBinContent(5,5,8,1.13266);
    AlphaTracklets13_->SetBinContent(5,5,9,1.11597);
    AlphaTracklets13_->SetBinContent(5,5,10,1.13077);
    AlphaTracklets13_->SetBinContent(5,6,1,1.18182);
    AlphaTracklets13_->SetBinContent(5,6,2,1.13403);
    AlphaTracklets13_->SetBinContent(5,6,3,1.1558);
    AlphaTracklets13_->SetBinContent(5,6,4,1.19555);
    AlphaTracklets13_->SetBinContent(5,6,5,1.12865);
    AlphaTracklets13_->SetBinContent(5,6,6,1.10945);
    AlphaTracklets13_->SetBinContent(5,6,7,1.18181);
    AlphaTracklets13_->SetBinContent(5,6,8,1.15265);
    AlphaTracklets13_->SetBinContent(5,6,9,1.10909);
    AlphaTracklets13_->SetBinContent(5,6,10,1.17587);
    AlphaTracklets13_->SetBinContent(5,7,1,1.21671);
    AlphaTracklets13_->SetBinContent(5,7,2,1.16261);
    AlphaTracklets13_->SetBinContent(5,7,3,1.18838);
    AlphaTracklets13_->SetBinContent(5,7,4,1.20225);
    AlphaTracklets13_->SetBinContent(5,7,5,1.13673);
    AlphaTracklets13_->SetBinContent(5,7,6,1.11799);
    AlphaTracklets13_->SetBinContent(5,7,7,1.18071);
    AlphaTracklets13_->SetBinContent(5,7,8,1.15251);
    AlphaTracklets13_->SetBinContent(5,7,9,1.12103);
    AlphaTracklets13_->SetBinContent(5,7,10,1.16266);
    AlphaTracklets13_->SetBinContent(5,8,1,1.1961);
    AlphaTracklets13_->SetBinContent(5,8,2,1.13567);
    AlphaTracklets13_->SetBinContent(5,8,3,1.15994);
    AlphaTracklets13_->SetBinContent(5,8,4,1.20614);
    AlphaTracklets13_->SetBinContent(5,8,5,1.13706);
    AlphaTracklets13_->SetBinContent(5,8,6,1.14398);
    AlphaTracklets13_->SetBinContent(5,8,7,1.18274);
    AlphaTracklets13_->SetBinContent(5,8,8,1.16538);
    AlphaTracklets13_->SetBinContent(5,8,9,1.12772);
    AlphaTracklets13_->SetBinContent(5,8,10,1.14127);
    AlphaTracklets13_->SetBinContent(5,9,1,1.19129);
    AlphaTracklets13_->SetBinContent(5,9,2,1.18872);
    AlphaTracklets13_->SetBinContent(5,9,3,1.18964);
    AlphaTracklets13_->SetBinContent(5,9,4,1.24034);
    AlphaTracklets13_->SetBinContent(5,9,5,1.14884);
    AlphaTracklets13_->SetBinContent(5,9,6,1.15101);
    AlphaTracklets13_->SetBinContent(5,9,7,1.20251);
    AlphaTracklets13_->SetBinContent(5,9,8,1.16842);
    AlphaTracklets13_->SetBinContent(5,9,9,1.15277);
    AlphaTracklets13_->SetBinContent(5,9,10,1.18675);
    AlphaTracklets13_->SetBinContent(5,10,1,1.22952);
    AlphaTracklets13_->SetBinContent(5,10,2,1.17517);
    AlphaTracklets13_->SetBinContent(5,10,3,1.19614);
    AlphaTracklets13_->SetBinContent(5,10,4,1.25654);
    AlphaTracklets13_->SetBinContent(5,10,5,1.17996);
    AlphaTracklets13_->SetBinContent(5,10,6,1.15863);
    AlphaTracklets13_->SetBinContent(5,10,7,1.23174);
    AlphaTracklets13_->SetBinContent(5,10,8,1.19652);
    AlphaTracklets13_->SetBinContent(5,10,9,1.17998);
    AlphaTracklets13_->SetBinContent(5,10,10,1.1921);
    AlphaTracklets13_->SetBinContent(5,11,1,1.32435);
    AlphaTracklets13_->SetBinContent(5,11,2,1.24419);
    AlphaTracklets13_->SetBinContent(5,11,3,1.22588);
    AlphaTracklets13_->SetBinContent(5,11,4,1.27198);
    AlphaTracklets13_->SetBinContent(5,11,5,1.20334);
    AlphaTracklets13_->SetBinContent(5,11,6,1.205);
    AlphaTracklets13_->SetBinContent(5,11,7,1.25117);
    AlphaTracklets13_->SetBinContent(5,11,8,1.21265);
    AlphaTracklets13_->SetBinContent(5,11,9,1.18149);
    AlphaTracklets13_->SetBinContent(5,11,10,1.2321);
    AlphaTracklets13_->SetBinContent(5,12,1,1.40741);
    AlphaTracklets13_->SetBinContent(5,12,2,1.30416);
    AlphaTracklets13_->SetBinContent(5,12,3,1.23364);
    AlphaTracklets13_->SetBinContent(5,12,4,1.31407);
    AlphaTracklets13_->SetBinContent(5,12,5,1.21068);
    AlphaTracklets13_->SetBinContent(5,12,6,1.22838);
    AlphaTracklets13_->SetBinContent(5,12,7,1.26783);
    AlphaTracklets13_->SetBinContent(5,12,8,1.24586);
    AlphaTracklets13_->SetBinContent(5,12,9,1.19503);
    AlphaTracklets13_->SetBinContent(5,12,10,1.19824);
    AlphaTracklets13_->SetBinContent(5,13,1,1.32);
    AlphaTracklets13_->SetBinContent(5,13,2,1.29452);
    AlphaTracklets13_->SetBinContent(5,13,3,1.35636);
    AlphaTracklets13_->SetBinContent(5,13,4,1.30545);
    AlphaTracklets13_->SetBinContent(5,13,5,1.32762);
    AlphaTracklets13_->SetBinContent(5,13,6,1.23088);
    AlphaTracklets13_->SetBinContent(5,13,7,1.33148);
    AlphaTracklets13_->SetBinContent(5,13,8,1.47027);
    AlphaTracklets13_->SetBinContent(5,13,9,1.31405);
    AlphaTracklets13_->SetBinContent(5,13,10,1.25926);
    AlphaTracklets13_->SetBinContent(5,14,1,0);
    AlphaTracklets13_->SetBinContent(5,14,2,0);
    AlphaTracklets13_->SetBinContent(5,14,3,0);
    AlphaTracklets13_->SetBinContent(5,14,4,0);
    AlphaTracklets13_->SetBinContent(5,14,5,0);
    AlphaTracklets13_->SetBinContent(5,14,6,0);
    AlphaTracklets13_->SetBinContent(5,14,7,0);
    AlphaTracklets13_->SetBinContent(5,14,8,0);
    AlphaTracklets13_->SetBinContent(5,14,9,0);
    AlphaTracklets13_->SetBinContent(5,14,10,0);
    AlphaTracklets13_->SetBinContent(6,1,1,1.06316);
    AlphaTracklets13_->SetBinContent(6,1,2,1.11067);
    AlphaTracklets13_->SetBinContent(6,1,3,1.22055);
    AlphaTracklets13_->SetBinContent(6,1,4,1.09595);
    AlphaTracklets13_->SetBinContent(6,1,5,1.05498);
    AlphaTracklets13_->SetBinContent(6,1,6,1.19455);
    AlphaTracklets13_->SetBinContent(6,1,7,1.11996);
    AlphaTracklets13_->SetBinContent(6,1,8,1.08129);
    AlphaTracklets13_->SetBinContent(6,1,9,1.16701);
    AlphaTracklets13_->SetBinContent(6,1,10,1.14194);
    AlphaTracklets13_->SetBinContent(6,2,1,1.10093);
    AlphaTracklets13_->SetBinContent(6,2,2,1.11716);
    AlphaTracklets13_->SetBinContent(6,2,3,1.17764);
    AlphaTracklets13_->SetBinContent(6,2,4,1.07862);
    AlphaTracklets13_->SetBinContent(6,2,5,1.06114);
    AlphaTracklets13_->SetBinContent(6,2,6,1.20356);
    AlphaTracklets13_->SetBinContent(6,2,7,1.10371);
    AlphaTracklets13_->SetBinContent(6,2,8,1.06791);
    AlphaTracklets13_->SetBinContent(6,2,9,1.12116);
    AlphaTracklets13_->SetBinContent(6,2,10,1.14248);
    AlphaTracklets13_->SetBinContent(6,3,1,1.10254);
    AlphaTracklets13_->SetBinContent(6,3,2,1.11102);
    AlphaTracklets13_->SetBinContent(6,3,3,1.16813);
    AlphaTracklets13_->SetBinContent(6,3,4,1.07826);
    AlphaTracklets13_->SetBinContent(6,3,5,1.05228);
    AlphaTracklets13_->SetBinContent(6,3,6,1.19042);
    AlphaTracklets13_->SetBinContent(6,3,7,1.09946);
    AlphaTracklets13_->SetBinContent(6,3,8,1.07499);
    AlphaTracklets13_->SetBinContent(6,3,9,1.11196);
    AlphaTracklets13_->SetBinContent(6,3,10,1.15217);
    AlphaTracklets13_->SetBinContent(6,4,1,1.07842);
    AlphaTracklets13_->SetBinContent(6,4,2,1.11872);
    AlphaTracklets13_->SetBinContent(6,4,3,1.18124);
    AlphaTracklets13_->SetBinContent(6,4,4,1.08033);
    AlphaTracklets13_->SetBinContent(6,4,5,1.05832);
    AlphaTracklets13_->SetBinContent(6,4,6,1.19292);
    AlphaTracklets13_->SetBinContent(6,4,7,1.10122);
    AlphaTracklets13_->SetBinContent(6,4,8,1.06853);
    AlphaTracklets13_->SetBinContent(6,4,9,1.13206);
    AlphaTracklets13_->SetBinContent(6,4,10,1.16031);
    AlphaTracklets13_->SetBinContent(6,5,1,1.0784);
    AlphaTracklets13_->SetBinContent(6,5,2,1.09777);
    AlphaTracklets13_->SetBinContent(6,5,3,1.19186);
    AlphaTracklets13_->SetBinContent(6,5,4,1.08408);
    AlphaTracklets13_->SetBinContent(6,5,5,1.05602);
    AlphaTracklets13_->SetBinContent(6,5,6,1.19784);
    AlphaTracklets13_->SetBinContent(6,5,7,1.0996);
    AlphaTracklets13_->SetBinContent(6,5,8,1.07263);
    AlphaTracklets13_->SetBinContent(6,5,9,1.12072);
    AlphaTracklets13_->SetBinContent(6,5,10,1.15709);
    AlphaTracklets13_->SetBinContent(6,6,1,1.10085);
    AlphaTracklets13_->SetBinContent(6,6,2,1.11276);
    AlphaTracklets13_->SetBinContent(6,6,3,1.17746);
    AlphaTracklets13_->SetBinContent(6,6,4,1.08725);
    AlphaTracklets13_->SetBinContent(6,6,5,1.05983);
    AlphaTracklets13_->SetBinContent(6,6,6,1.20599);
    AlphaTracklets13_->SetBinContent(6,6,7,1.11392);
    AlphaTracklets13_->SetBinContent(6,6,8,1.07463);
    AlphaTracklets13_->SetBinContent(6,6,9,1.11397);
    AlphaTracklets13_->SetBinContent(6,6,10,1.18056);
    AlphaTracklets13_->SetBinContent(6,7,1,1.13551);
    AlphaTracklets13_->SetBinContent(6,7,2,1.128);
    AlphaTracklets13_->SetBinContent(6,7,3,1.17499);
    AlphaTracklets13_->SetBinContent(6,7,4,1.09453);
    AlphaTracklets13_->SetBinContent(6,7,5,1.07152);
    AlphaTracklets13_->SetBinContent(6,7,6,1.20163);
    AlphaTracklets13_->SetBinContent(6,7,7,1.09988);
    AlphaTracklets13_->SetBinContent(6,7,8,1.06398);
    AlphaTracklets13_->SetBinContent(6,7,9,1.13988);
    AlphaTracklets13_->SetBinContent(6,7,10,1.15017);
    AlphaTracklets13_->SetBinContent(6,8,1,1.11586);
    AlphaTracklets13_->SetBinContent(6,8,2,1.11285);
    AlphaTracklets13_->SetBinContent(6,8,3,1.19557);
    AlphaTracklets13_->SetBinContent(6,8,4,1.11133);
    AlphaTracklets13_->SetBinContent(6,8,5,1.06904);
    AlphaTracklets13_->SetBinContent(6,8,6,1.21462);
    AlphaTracklets13_->SetBinContent(6,8,7,1.11884);
    AlphaTracklets13_->SetBinContent(6,8,8,1.08969);
    AlphaTracklets13_->SetBinContent(6,8,9,1.14363);
    AlphaTracklets13_->SetBinContent(6,8,10,1.19816);
    AlphaTracklets13_->SetBinContent(6,9,1,1.11095);
    AlphaTracklets13_->SetBinContent(6,9,2,1.16091);
    AlphaTracklets13_->SetBinContent(6,9,3,1.17439);
    AlphaTracklets13_->SetBinContent(6,9,4,1.10918);
    AlphaTracklets13_->SetBinContent(6,9,5,1.08097);
    AlphaTracklets13_->SetBinContent(6,9,6,1.21504);
    AlphaTracklets13_->SetBinContent(6,9,7,1.11338);
    AlphaTracklets13_->SetBinContent(6,9,8,1.09825);
    AlphaTracklets13_->SetBinContent(6,9,9,1.15824);
    AlphaTracklets13_->SetBinContent(6,9,10,1.18458);
    AlphaTracklets13_->SetBinContent(6,10,1,1.08971);
    AlphaTracklets13_->SetBinContent(6,10,2,1.15604);
    AlphaTracklets13_->SetBinContent(6,10,3,1.18915);
    AlphaTracklets13_->SetBinContent(6,10,4,1.10636);
    AlphaTracklets13_->SetBinContent(6,10,5,1.08489);
    AlphaTracklets13_->SetBinContent(6,10,6,1.23463);
    AlphaTracklets13_->SetBinContent(6,10,7,1.14198);
    AlphaTracklets13_->SetBinContent(6,10,8,1.10959);
    AlphaTracklets13_->SetBinContent(6,10,9,1.16876);
    AlphaTracklets13_->SetBinContent(6,10,10,1.20914);
    AlphaTracklets13_->SetBinContent(6,11,1,1.13206);
    AlphaTracklets13_->SetBinContent(6,11,2,1.18382);
    AlphaTracklets13_->SetBinContent(6,11,3,1.21928);
    AlphaTracklets13_->SetBinContent(6,11,4,1.13857);
    AlphaTracklets13_->SetBinContent(6,11,5,1.105);
    AlphaTracklets13_->SetBinContent(6,11,6,1.25069);
    AlphaTracklets13_->SetBinContent(6,11,7,1.1342);
    AlphaTracklets13_->SetBinContent(6,11,8,1.12825);
    AlphaTracklets13_->SetBinContent(6,11,9,1.17995);
    AlphaTracklets13_->SetBinContent(6,11,10,1.25238);
    AlphaTracklets13_->SetBinContent(6,12,1,1.27556);
    AlphaTracklets13_->SetBinContent(6,12,2,1.22318);
    AlphaTracklets13_->SetBinContent(6,12,3,1.2359);
    AlphaTracklets13_->SetBinContent(6,12,4,1.13796);
    AlphaTracklets13_->SetBinContent(6,12,5,1.12411);
    AlphaTracklets13_->SetBinContent(6,12,6,1.2484);
    AlphaTracklets13_->SetBinContent(6,12,7,1.15697);
    AlphaTracklets13_->SetBinContent(6,12,8,1.1349);
    AlphaTracklets13_->SetBinContent(6,12,9,1.2);
    AlphaTracklets13_->SetBinContent(6,12,10,1.13502);
    AlphaTracklets13_->SetBinContent(6,13,1,1.25);
    AlphaTracklets13_->SetBinContent(6,13,2,1.11538);
    AlphaTracklets13_->SetBinContent(6,13,3,1.18729);
    AlphaTracklets13_->SetBinContent(6,13,4,1.18182);
    AlphaTracklets13_->SetBinContent(6,13,5,1.17485);
    AlphaTracklets13_->SetBinContent(6,13,6,1.27542);
    AlphaTracklets13_->SetBinContent(6,13,7,1.23059);
    AlphaTracklets13_->SetBinContent(6,13,8,1.13043);
    AlphaTracklets13_->SetBinContent(6,13,9,1.21831);
    AlphaTracklets13_->SetBinContent(6,13,10,1.2449);
    AlphaTracklets13_->SetBinContent(6,14,1,0);
    AlphaTracklets13_->SetBinContent(6,14,2,0);
    AlphaTracklets13_->SetBinContent(6,14,3,0);
    AlphaTracklets13_->SetBinContent(6,14,4,0);
    AlphaTracklets13_->SetBinContent(6,14,5,0);
    AlphaTracklets13_->SetBinContent(6,14,6,0);
    AlphaTracklets13_->SetBinContent(6,14,7,0);
    AlphaTracklets13_->SetBinContent(6,14,8,0);
    AlphaTracklets13_->SetBinContent(6,14,9,0);
    AlphaTracklets13_->SetBinContent(6,14,10,0);
    AlphaTracklets13_->SetBinContent(7,1,1,1.13218);
    AlphaTracklets13_->SetBinContent(7,1,2,1.18476);
    AlphaTracklets13_->SetBinContent(7,1,3,1.05882);
    AlphaTracklets13_->SetBinContent(7,1,4,1.14318);
    AlphaTracklets13_->SetBinContent(7,1,5,1.22005);
    AlphaTracklets13_->SetBinContent(7,1,6,1.061);
    AlphaTracklets13_->SetBinContent(7,1,7,1.10841);
    AlphaTracklets13_->SetBinContent(7,1,8,1.23014);
    AlphaTracklets13_->SetBinContent(7,1,9,1.09358);
    AlphaTracklets13_->SetBinContent(7,1,10,1.09314);
    AlphaTracklets13_->SetBinContent(7,2,1,1.09262);
    AlphaTracklets13_->SetBinContent(7,2,2,1.1099);
    AlphaTracklets13_->SetBinContent(7,2,3,1.06228);
    AlphaTracklets13_->SetBinContent(7,2,4,1.10578);
    AlphaTracklets13_->SetBinContent(7,2,5,1.19657);
    AlphaTracklets13_->SetBinContent(7,2,6,1.05193);
    AlphaTracklets13_->SetBinContent(7,2,7,1.08959);
    AlphaTracklets13_->SetBinContent(7,2,8,1.19168);
    AlphaTracklets13_->SetBinContent(7,2,9,1.11591);
    AlphaTracklets13_->SetBinContent(7,2,10,1.13579);
    AlphaTracklets13_->SetBinContent(7,3,1,1.15529);
    AlphaTracklets13_->SetBinContent(7,3,2,1.12035);
    AlphaTracklets13_->SetBinContent(7,3,3,1.06284);
    AlphaTracklets13_->SetBinContent(7,3,4,1.10075);
    AlphaTracklets13_->SetBinContent(7,3,5,1.1916);
    AlphaTracklets13_->SetBinContent(7,3,6,1.04848);
    AlphaTracklets13_->SetBinContent(7,3,7,1.0905);
    AlphaTracklets13_->SetBinContent(7,3,8,1.19926);
    AlphaTracklets13_->SetBinContent(7,3,9,1.12392);
    AlphaTracklets13_->SetBinContent(7,3,10,1.11502);
    AlphaTracklets13_->SetBinContent(7,4,1,1.12353);
    AlphaTracklets13_->SetBinContent(7,4,2,1.1139);
    AlphaTracklets13_->SetBinContent(7,4,3,1.0645);
    AlphaTracklets13_->SetBinContent(7,4,4,1.09434);
    AlphaTracklets13_->SetBinContent(7,4,5,1.19858);
    AlphaTracklets13_->SetBinContent(7,4,6,1.05251);
    AlphaTracklets13_->SetBinContent(7,4,7,1.09308);
    AlphaTracklets13_->SetBinContent(7,4,8,1.18226);
    AlphaTracklets13_->SetBinContent(7,4,9,1.15234);
    AlphaTracklets13_->SetBinContent(7,4,10,1.08247);
    AlphaTracklets13_->SetBinContent(7,5,1,1.15231);
    AlphaTracklets13_->SetBinContent(7,5,2,1.12521);
    AlphaTracklets13_->SetBinContent(7,5,3,1.06854);
    AlphaTracklets13_->SetBinContent(7,5,4,1.10037);
    AlphaTracklets13_->SetBinContent(7,5,5,1.19866);
    AlphaTracklets13_->SetBinContent(7,5,6,1.05499);
    AlphaTracklets13_->SetBinContent(7,5,7,1.08766);
    AlphaTracklets13_->SetBinContent(7,5,8,1.19932);
    AlphaTracklets13_->SetBinContent(7,5,9,1.12945);
    AlphaTracklets13_->SetBinContent(7,5,10,1.12061);
    AlphaTracklets13_->SetBinContent(7,6,1,1.15227);
    AlphaTracklets13_->SetBinContent(7,6,2,1.13378);
    AlphaTracklets13_->SetBinContent(7,6,3,1.06652);
    AlphaTracklets13_->SetBinContent(7,6,4,1.09952);
    AlphaTracklets13_->SetBinContent(7,6,5,1.20622);
    AlphaTracklets13_->SetBinContent(7,6,6,1.0545);
    AlphaTracklets13_->SetBinContent(7,6,7,1.09439);
    AlphaTracklets13_->SetBinContent(7,6,8,1.17784);
    AlphaTracklets13_->SetBinContent(7,6,9,1.14341);
    AlphaTracklets13_->SetBinContent(7,6,10,1.11466);
    AlphaTracklets13_->SetBinContent(7,7,1,1.15797);
    AlphaTracklets13_->SetBinContent(7,7,2,1.14119);
    AlphaTracklets13_->SetBinContent(7,7,3,1.075);
    AlphaTracklets13_->SetBinContent(7,7,4,1.11101);
    AlphaTracklets13_->SetBinContent(7,7,5,1.20616);
    AlphaTracklets13_->SetBinContent(7,7,6,1.06839);
    AlphaTracklets13_->SetBinContent(7,7,7,1.09694);
    AlphaTracklets13_->SetBinContent(7,7,8,1.1952);
    AlphaTracklets13_->SetBinContent(7,7,9,1.13465);
    AlphaTracklets13_->SetBinContent(7,7,10,1.11111);
    AlphaTracklets13_->SetBinContent(7,8,1,1.18112);
    AlphaTracklets13_->SetBinContent(7,8,2,1.1523);
    AlphaTracklets13_->SetBinContent(7,8,3,1.08592);
    AlphaTracklets13_->SetBinContent(7,8,4,1.11138);
    AlphaTracklets13_->SetBinContent(7,8,5,1.22069);
    AlphaTracklets13_->SetBinContent(7,8,6,1.08008);
    AlphaTracklets13_->SetBinContent(7,8,7,1.09911);
    AlphaTracklets13_->SetBinContent(7,8,8,1.21837);
    AlphaTracklets13_->SetBinContent(7,8,9,1.15834);
    AlphaTracklets13_->SetBinContent(7,8,10,1.141);
    AlphaTracklets13_->SetBinContent(7,9,1,1.15948);
    AlphaTracklets13_->SetBinContent(7,9,2,1.15789);
    AlphaTracklets13_->SetBinContent(7,9,3,1.09369);
    AlphaTracklets13_->SetBinContent(7,9,4,1.11896);
    AlphaTracklets13_->SetBinContent(7,9,5,1.21251);
    AlphaTracklets13_->SetBinContent(7,9,6,1.08071);
    AlphaTracklets13_->SetBinContent(7,9,7,1.12048);
    AlphaTracklets13_->SetBinContent(7,9,8,1.20132);
    AlphaTracklets13_->SetBinContent(7,9,9,1.15822);
    AlphaTracklets13_->SetBinContent(7,9,10,1.15059);
    AlphaTracklets13_->SetBinContent(7,10,1,1.23171);
    AlphaTracklets13_->SetBinContent(7,10,2,1.15651);
    AlphaTracklets13_->SetBinContent(7,10,3,1.1176);
    AlphaTracklets13_->SetBinContent(7,10,4,1.13589);
    AlphaTracklets13_->SetBinContent(7,10,5,1.22916);
    AlphaTracklets13_->SetBinContent(7,10,6,1.09357);
    AlphaTracklets13_->SetBinContent(7,10,7,1.12445);
    AlphaTracklets13_->SetBinContent(7,10,8,1.20941);
    AlphaTracklets13_->SetBinContent(7,10,9,1.16201);
    AlphaTracklets13_->SetBinContent(7,10,10,1.1661);
    AlphaTracklets13_->SetBinContent(7,11,1,1.22821);
    AlphaTracklets13_->SetBinContent(7,11,2,1.2008);
    AlphaTracklets13_->SetBinContent(7,11,3,1.12899);
    AlphaTracklets13_->SetBinContent(7,11,4,1.16528);
    AlphaTracklets13_->SetBinContent(7,11,5,1.24038);
    AlphaTracklets13_->SetBinContent(7,11,6,1.1088);
    AlphaTracklets13_->SetBinContent(7,11,7,1.1345);
    AlphaTracklets13_->SetBinContent(7,11,8,1.21419);
    AlphaTracklets13_->SetBinContent(7,11,9,1.18209);
    AlphaTracklets13_->SetBinContent(7,11,10,1.12106);
    AlphaTracklets13_->SetBinContent(7,12,1,1.20921);
    AlphaTracklets13_->SetBinContent(7,12,2,1.31198);
    AlphaTracklets13_->SetBinContent(7,12,3,1.18004);
    AlphaTracklets13_->SetBinContent(7,12,4,1.16825);
    AlphaTracklets13_->SetBinContent(7,12,5,1.28494);
    AlphaTracklets13_->SetBinContent(7,12,6,1.10158);
    AlphaTracklets13_->SetBinContent(7,12,7,1.15821);
    AlphaTracklets13_->SetBinContent(7,12,8,1.26099);
    AlphaTracklets13_->SetBinContent(7,12,9,1.26402);
    AlphaTracklets13_->SetBinContent(7,12,10,1.23077);
    AlphaTracklets13_->SetBinContent(7,13,1,1.0303);
    AlphaTracklets13_->SetBinContent(7,13,2,1.35664);
    AlphaTracklets13_->SetBinContent(7,13,3,1.11455);
    AlphaTracklets13_->SetBinContent(7,13,4,1.13276);
    AlphaTracklets13_->SetBinContent(7,13,5,1.33663);
    AlphaTracklets13_->SetBinContent(7,13,6,1.14883);
    AlphaTracklets13_->SetBinContent(7,13,7,1.23239);
    AlphaTracklets13_->SetBinContent(7,13,8,1.13734);
    AlphaTracklets13_->SetBinContent(7,13,9,1.16379);
    AlphaTracklets13_->SetBinContent(7,13,10,1.37288);
    AlphaTracklets13_->SetBinContent(7,14,1,0);
    AlphaTracklets13_->SetBinContent(7,14,2,0);
    AlphaTracklets13_->SetBinContent(7,14,3,0);
    AlphaTracklets13_->SetBinContent(7,14,4,0);
    AlphaTracklets13_->SetBinContent(7,14,5,0);
    AlphaTracklets13_->SetBinContent(7,14,6,0);
    AlphaTracklets13_->SetBinContent(7,14,7,0);
    AlphaTracklets13_->SetBinContent(7,14,8,0);
    AlphaTracklets13_->SetBinContent(7,14,9,0);
    AlphaTracklets13_->SetBinContent(7,14,10,0);
    AlphaTracklets13_->SetBinContent(8,1,1,1.12617);
    AlphaTracklets13_->SetBinContent(8,1,2,1.06877);
    AlphaTracklets13_->SetBinContent(8,1,3,1.14404);
    AlphaTracklets13_->SetBinContent(8,1,4,1.17971);
    AlphaTracklets13_->SetBinContent(8,1,5,1.10337);
    AlphaTracklets13_->SetBinContent(8,1,6,1.11499);
    AlphaTracklets13_->SetBinContent(8,1,7,1.17785);
    AlphaTracklets13_->SetBinContent(8,1,8,1.11549);
    AlphaTracklets13_->SetBinContent(8,1,9,1.08046);
    AlphaTracklets13_->SetBinContent(8,1,10,1.11157);
    AlphaTracklets13_->SetBinContent(8,2,1,1.13895);
    AlphaTracklets13_->SetBinContent(8,2,2,1.0814);
    AlphaTracklets13_->SetBinContent(8,2,3,1.11499);
    AlphaTracklets13_->SetBinContent(8,2,4,1.14939);
    AlphaTracklets13_->SetBinContent(8,2,5,1.08972);
    AlphaTracklets13_->SetBinContent(8,2,6,1.11336);
    AlphaTracklets13_->SetBinContent(8,2,7,1.18176);
    AlphaTracklets13_->SetBinContent(8,2,8,1.11903);
    AlphaTracklets13_->SetBinContent(8,2,9,1.0863);
    AlphaTracklets13_->SetBinContent(8,2,10,1.14302);
    AlphaTracklets13_->SetBinContent(8,3,1,1.14824);
    AlphaTracklets13_->SetBinContent(8,3,2,1.0757);
    AlphaTracklets13_->SetBinContent(8,3,3,1.11691);
    AlphaTracklets13_->SetBinContent(8,3,4,1.1665);
    AlphaTracklets13_->SetBinContent(8,3,5,1.10409);
    AlphaTracklets13_->SetBinContent(8,3,6,1.10856);
    AlphaTracklets13_->SetBinContent(8,3,7,1.193);
    AlphaTracklets13_->SetBinContent(8,3,8,1.10819);
    AlphaTracklets13_->SetBinContent(8,3,9,1.12167);
    AlphaTracklets13_->SetBinContent(8,3,10,1.17042);
    AlphaTracklets13_->SetBinContent(8,4,1,1.14119);
    AlphaTracklets13_->SetBinContent(8,4,2,1.10466);
    AlphaTracklets13_->SetBinContent(8,4,3,1.13203);
    AlphaTracklets13_->SetBinContent(8,4,4,1.17175);
    AlphaTracklets13_->SetBinContent(8,4,5,1.10742);
    AlphaTracklets13_->SetBinContent(8,4,6,1.13477);
    AlphaTracklets13_->SetBinContent(8,4,7,1.18786);
    AlphaTracklets13_->SetBinContent(8,4,8,1.13026);
    AlphaTracklets13_->SetBinContent(8,4,9,1.11287);
    AlphaTracklets13_->SetBinContent(8,4,10,1.13071);
    AlphaTracklets13_->SetBinContent(8,5,1,1.17321);
    AlphaTracklets13_->SetBinContent(8,5,2,1.08858);
    AlphaTracklets13_->SetBinContent(8,5,3,1.1415);
    AlphaTracklets13_->SetBinContent(8,5,4,1.17187);
    AlphaTracklets13_->SetBinContent(8,5,5,1.11641);
    AlphaTracklets13_->SetBinContent(8,5,6,1.12977);
    AlphaTracklets13_->SetBinContent(8,5,7,1.18387);
    AlphaTracklets13_->SetBinContent(8,5,8,1.13696);
    AlphaTracklets13_->SetBinContent(8,5,9,1.10478);
    AlphaTracklets13_->SetBinContent(8,5,10,1.12434);
    AlphaTracklets13_->SetBinContent(8,6,1,1.15863);
    AlphaTracklets13_->SetBinContent(8,6,2,1.11125);
    AlphaTracklets13_->SetBinContent(8,6,3,1.14671);
    AlphaTracklets13_->SetBinContent(8,6,4,1.18781);
    AlphaTracklets13_->SetBinContent(8,6,5,1.12272);
    AlphaTracklets13_->SetBinContent(8,6,6,1.14967);
    AlphaTracklets13_->SetBinContent(8,6,7,1.19992);
    AlphaTracklets13_->SetBinContent(8,6,8,1.14556);
    AlphaTracklets13_->SetBinContent(8,6,9,1.10067);
    AlphaTracklets13_->SetBinContent(8,6,10,1.18117);
    AlphaTracklets13_->SetBinContent(8,7,1,1.16418);
    AlphaTracklets13_->SetBinContent(8,7,2,1.11264);
    AlphaTracklets13_->SetBinContent(8,7,3,1.15987);
    AlphaTracklets13_->SetBinContent(8,7,4,1.19661);
    AlphaTracklets13_->SetBinContent(8,7,5,1.13919);
    AlphaTracklets13_->SetBinContent(8,7,6,1.15317);
    AlphaTracklets13_->SetBinContent(8,7,7,1.2062);
    AlphaTracklets13_->SetBinContent(8,7,8,1.15788);
    AlphaTracklets13_->SetBinContent(8,7,9,1.13511);
    AlphaTracklets13_->SetBinContent(8,7,10,1.21519);
    AlphaTracklets13_->SetBinContent(8,8,1,1.14592);
    AlphaTracklets13_->SetBinContent(8,8,2,1.12622);
    AlphaTracklets13_->SetBinContent(8,8,3,1.15198);
    AlphaTracklets13_->SetBinContent(8,8,4,1.20281);
    AlphaTracklets13_->SetBinContent(8,8,5,1.1491);
    AlphaTracklets13_->SetBinContent(8,8,6,1.17113);
    AlphaTracklets13_->SetBinContent(8,8,7,1.21557);
    AlphaTracklets13_->SetBinContent(8,8,8,1.16772);
    AlphaTracklets13_->SetBinContent(8,8,9,1.12955);
    AlphaTracklets13_->SetBinContent(8,8,10,1.19723);
    AlphaTracklets13_->SetBinContent(8,9,1,1.14279);
    AlphaTracklets13_->SetBinContent(8,9,2,1.16892);
    AlphaTracklets13_->SetBinContent(8,9,3,1.18036);
    AlphaTracklets13_->SetBinContent(8,9,4,1.22542);
    AlphaTracklets13_->SetBinContent(8,9,5,1.16223);
    AlphaTracklets13_->SetBinContent(8,9,6,1.17592);
    AlphaTracklets13_->SetBinContent(8,9,7,1.23718);
    AlphaTracklets13_->SetBinContent(8,9,8,1.17686);
    AlphaTracklets13_->SetBinContent(8,9,9,1.13913);
    AlphaTracklets13_->SetBinContent(8,9,10,1.18542);
    AlphaTracklets13_->SetBinContent(8,10,1,1.18227);
    AlphaTracklets13_->SetBinContent(8,10,2,1.12817);
    AlphaTracklets13_->SetBinContent(8,10,3,1.19334);
    AlphaTracklets13_->SetBinContent(8,10,4,1.23377);
    AlphaTracklets13_->SetBinContent(8,10,5,1.17867);
    AlphaTracklets13_->SetBinContent(8,10,6,1.19644);
    AlphaTracklets13_->SetBinContent(8,10,7,1.25137);
    AlphaTracklets13_->SetBinContent(8,10,8,1.19674);
    AlphaTracklets13_->SetBinContent(8,10,9,1.16542);
    AlphaTracklets13_->SetBinContent(8,10,10,1.1722);
    AlphaTracklets13_->SetBinContent(8,11,1,1.2304);
    AlphaTracklets13_->SetBinContent(8,11,2,1.17566);
    AlphaTracklets13_->SetBinContent(8,11,3,1.22086);
    AlphaTracklets13_->SetBinContent(8,11,4,1.2624);
    AlphaTracklets13_->SetBinContent(8,11,5,1.19263);
    AlphaTracklets13_->SetBinContent(8,11,6,1.20722);
    AlphaTracklets13_->SetBinContent(8,11,7,1.28102);
    AlphaTracklets13_->SetBinContent(8,11,8,1.22259);
    AlphaTracklets13_->SetBinContent(8,11,9,1.2047);
    AlphaTracklets13_->SetBinContent(8,11,10,1.23216);
    AlphaTracklets13_->SetBinContent(8,12,1,1.175);
    AlphaTracklets13_->SetBinContent(8,12,2,1.17877);
    AlphaTracklets13_->SetBinContent(8,12,3,1.25127);
    AlphaTracklets13_->SetBinContent(8,12,4,1.34195);
    AlphaTracklets13_->SetBinContent(8,12,5,1.23145);
    AlphaTracklets13_->SetBinContent(8,12,6,1.25984);
    AlphaTracklets13_->SetBinContent(8,12,7,1.34368);
    AlphaTracklets13_->SetBinContent(8,12,8,1.26589);
    AlphaTracklets13_->SetBinContent(8,12,9,1.22994);
    AlphaTracklets13_->SetBinContent(8,12,10,1.31019);
    AlphaTracklets13_->SetBinContent(8,13,1,1.61905);
    AlphaTracklets13_->SetBinContent(8,13,2,1.10241);
    AlphaTracklets13_->SetBinContent(8,13,3,1.29057);
    AlphaTracklets13_->SetBinContent(8,13,4,1.34879);
    AlphaTracklets13_->SetBinContent(8,13,5,1.27574);
    AlphaTracklets13_->SetBinContent(8,13,6,1.31918);
    AlphaTracklets13_->SetBinContent(8,13,7,1.29592);
    AlphaTracklets13_->SetBinContent(8,13,8,1.20082);
    AlphaTracklets13_->SetBinContent(8,13,9,1.30556);
    AlphaTracklets13_->SetBinContent(8,13,10,1.17241);
    AlphaTracklets13_->SetBinContent(8,14,1,0);
    AlphaTracklets13_->SetBinContent(8,14,2,0);
    AlphaTracklets13_->SetBinContent(8,14,3,0);
    AlphaTracklets13_->SetBinContent(8,14,4,0);
    AlphaTracklets13_->SetBinContent(8,14,5,0);
    AlphaTracklets13_->SetBinContent(8,14,6,0);
    AlphaTracklets13_->SetBinContent(8,14,7,0);
    AlphaTracklets13_->SetBinContent(8,14,8,0);
    AlphaTracklets13_->SetBinContent(8,14,9,0);
    AlphaTracklets13_->SetBinContent(8,14,10,0);
    AlphaTracklets13_->SetBinContent(9,1,1,1.10121);
    AlphaTracklets13_->SetBinContent(9,1,2,1.12242);
    AlphaTracklets13_->SetBinContent(9,1,3,1.11463);
    AlphaTracklets13_->SetBinContent(9,1,4,1.10519);
    AlphaTracklets13_->SetBinContent(9,1,5,1.13033);
    AlphaTracklets13_->SetBinContent(9,1,6,1.11687);
    AlphaTracklets13_->SetBinContent(9,1,7,1.06007);
    AlphaTracklets13_->SetBinContent(9,1,8,1.11901);
    AlphaTracklets13_->SetBinContent(9,1,9,1.36809);
    AlphaTracklets13_->SetBinContent(9,1,10,1.67052);
    AlphaTracklets13_->SetBinContent(9,2,1,1.21315);
    AlphaTracklets13_->SetBinContent(9,2,2,1.16989);
    AlphaTracklets13_->SetBinContent(9,2,3,1.11643);
    AlphaTracklets13_->SetBinContent(9,2,4,1.09963);
    AlphaTracklets13_->SetBinContent(9,2,5,1.14862);
    AlphaTracklets13_->SetBinContent(9,2,6,1.13469);
    AlphaTracklets13_->SetBinContent(9,2,7,1.08193);
    AlphaTracklets13_->SetBinContent(9,2,8,1.14377);
    AlphaTracklets13_->SetBinContent(9,2,9,1.42342);
    AlphaTracklets13_->SetBinContent(9,2,10,1.71386);
    AlphaTracklets13_->SetBinContent(9,3,1,1.16887);
    AlphaTracklets13_->SetBinContent(9,3,2,1.16889);
    AlphaTracklets13_->SetBinContent(9,3,3,1.12825);
    AlphaTracklets13_->SetBinContent(9,3,4,1.12092);
    AlphaTracklets13_->SetBinContent(9,3,5,1.17222);
    AlphaTracklets13_->SetBinContent(9,3,6,1.15845);
    AlphaTracklets13_->SetBinContent(9,3,7,1.10523);
    AlphaTracklets13_->SetBinContent(9,3,8,1.16612);
    AlphaTracklets13_->SetBinContent(9,3,9,1.40569);
    AlphaTracklets13_->SetBinContent(9,3,10,1.61383);
    AlphaTracklets13_->SetBinContent(9,4,1,1.1267);
    AlphaTracklets13_->SetBinContent(9,4,2,1.18256);
    AlphaTracklets13_->SetBinContent(9,4,3,1.14513);
    AlphaTracklets13_->SetBinContent(9,4,4,1.13743);
    AlphaTracklets13_->SetBinContent(9,4,5,1.18066);
    AlphaTracklets13_->SetBinContent(9,4,6,1.17613);
    AlphaTracklets13_->SetBinContent(9,4,7,1.11583);
    AlphaTracklets13_->SetBinContent(9,4,8,1.18128);
    AlphaTracklets13_->SetBinContent(9,4,9,1.43845);
    AlphaTracklets13_->SetBinContent(9,4,10,1.75252);
    AlphaTracklets13_->SetBinContent(9,5,1,1.2204);
    AlphaTracklets13_->SetBinContent(9,5,2,1.18959);
    AlphaTracklets13_->SetBinContent(9,5,3,1.17165);
    AlphaTracklets13_->SetBinContent(9,5,4,1.1555);
    AlphaTracklets13_->SetBinContent(9,5,5,1.19029);
    AlphaTracklets13_->SetBinContent(9,5,6,1.18661);
    AlphaTracklets13_->SetBinContent(9,5,7,1.14057);
    AlphaTracklets13_->SetBinContent(9,5,8,1.19565);
    AlphaTracklets13_->SetBinContent(9,5,9,1.42999);
    AlphaTracklets13_->SetBinContent(9,5,10,1.70257);
    AlphaTracklets13_->SetBinContent(9,6,1,1.22196);
    AlphaTracklets13_->SetBinContent(9,6,2,1.25932);
    AlphaTracklets13_->SetBinContent(9,6,3,1.17128);
    AlphaTracklets13_->SetBinContent(9,6,4,1.16974);
    AlphaTracklets13_->SetBinContent(9,6,5,1.19975);
    AlphaTracklets13_->SetBinContent(9,6,6,1.20857);
    AlphaTracklets13_->SetBinContent(9,6,7,1.16142);
    AlphaTracklets13_->SetBinContent(9,6,8,1.23016);
    AlphaTracklets13_->SetBinContent(9,6,9,1.46978);
    AlphaTracklets13_->SetBinContent(9,6,10,1.80957);
    AlphaTracklets13_->SetBinContent(9,7,1,1.22305);
    AlphaTracklets13_->SetBinContent(9,7,2,1.2363);
    AlphaTracklets13_->SetBinContent(9,7,3,1.20949);
    AlphaTracklets13_->SetBinContent(9,7,4,1.17043);
    AlphaTracklets13_->SetBinContent(9,7,5,1.23028);
    AlphaTracklets13_->SetBinContent(9,7,6,1.21562);
    AlphaTracklets13_->SetBinContent(9,7,7,1.18396);
    AlphaTracklets13_->SetBinContent(9,7,8,1.22272);
    AlphaTracklets13_->SetBinContent(9,7,9,1.50539);
    AlphaTracklets13_->SetBinContent(9,7,10,1.81508);
    AlphaTracklets13_->SetBinContent(9,8,1,1.20323);
    AlphaTracklets13_->SetBinContent(9,8,2,1.25697);
    AlphaTracklets13_->SetBinContent(9,8,3,1.22259);
    AlphaTracklets13_->SetBinContent(9,8,4,1.19169);
    AlphaTracklets13_->SetBinContent(9,8,5,1.23404);
    AlphaTracklets13_->SetBinContent(9,8,6,1.24409);
    AlphaTracklets13_->SetBinContent(9,8,7,1.17263);
    AlphaTracklets13_->SetBinContent(9,8,8,1.24116);
    AlphaTracklets13_->SetBinContent(9,8,9,1.46887);
    AlphaTracklets13_->SetBinContent(9,8,10,1.83831);
    AlphaTracklets13_->SetBinContent(9,9,1,1.3095);
    AlphaTracklets13_->SetBinContent(9,9,2,1.26329);
    AlphaTracklets13_->SetBinContent(9,9,3,1.23293);
    AlphaTracklets13_->SetBinContent(9,9,4,1.2149);
    AlphaTracklets13_->SetBinContent(9,9,5,1.25155);
    AlphaTracklets13_->SetBinContent(9,9,6,1.25575);
    AlphaTracklets13_->SetBinContent(9,9,7,1.1952);
    AlphaTracklets13_->SetBinContent(9,9,8,1.27905);
    AlphaTracklets13_->SetBinContent(9,9,9,1.55203);
    AlphaTracklets13_->SetBinContent(9,9,10,1.87669);
    AlphaTracklets13_->SetBinContent(9,10,1,1.21372);
    AlphaTracklets13_->SetBinContent(9,10,2,1.34199);
    AlphaTracklets13_->SetBinContent(9,10,3,1.2497);
    AlphaTracklets13_->SetBinContent(9,10,4,1.24864);
    AlphaTracklets13_->SetBinContent(9,10,5,1.28144);
    AlphaTracklets13_->SetBinContent(9,10,6,1.28908);
    AlphaTracklets13_->SetBinContent(9,10,7,1.22289);
    AlphaTracklets13_->SetBinContent(9,10,8,1.26333);
    AlphaTracklets13_->SetBinContent(9,10,9,1.55702);
    AlphaTracklets13_->SetBinContent(9,10,10,1.91877);
    AlphaTracklets13_->SetBinContent(9,11,1,1.29098);
    AlphaTracklets13_->SetBinContent(9,11,2,1.32848);
    AlphaTracklets13_->SetBinContent(9,11,3,1.29292);
    AlphaTracklets13_->SetBinContent(9,11,4,1.26426);
    AlphaTracklets13_->SetBinContent(9,11,5,1.30746);
    AlphaTracklets13_->SetBinContent(9,11,6,1.33277);
    AlphaTracklets13_->SetBinContent(9,11,7,1.27772);
    AlphaTracklets13_->SetBinContent(9,11,8,1.33713);
    AlphaTracklets13_->SetBinContent(9,11,9,1.52621);
    AlphaTracklets13_->SetBinContent(9,11,10,1.98951);
    AlphaTracklets13_->SetBinContent(9,12,1,1.63314);
    AlphaTracklets13_->SetBinContent(9,12,2,1.36947);
    AlphaTracklets13_->SetBinContent(9,12,3,1.35923);
    AlphaTracklets13_->SetBinContent(9,12,4,1.33833);
    AlphaTracklets13_->SetBinContent(9,12,5,1.33516);
    AlphaTracklets13_->SetBinContent(9,12,6,1.40493);
    AlphaTracklets13_->SetBinContent(9,12,7,1.30873);
    AlphaTracklets13_->SetBinContent(9,12,8,1.39798);
    AlphaTracklets13_->SetBinContent(9,12,9,1.70209);
    AlphaTracklets13_->SetBinContent(9,12,10,1.9375);
    AlphaTracklets13_->SetBinContent(9,13,1,1.42105);
    AlphaTracklets13_->SetBinContent(9,13,2,1.40816);
    AlphaTracklets13_->SetBinContent(9,13,3,1.43772);
    AlphaTracklets13_->SetBinContent(9,13,4,1.32669);
    AlphaTracklets13_->SetBinContent(9,13,5,1.54321);
    AlphaTracklets13_->SetBinContent(9,13,6,1.39658);
    AlphaTracklets13_->SetBinContent(9,13,7,1.27123);
    AlphaTracklets13_->SetBinContent(9,13,8,1.3399);
    AlphaTracklets13_->SetBinContent(9,13,9,1.4375);
    AlphaTracklets13_->SetBinContent(9,13,10,1.54);
    AlphaTracklets13_->SetBinContent(9,14,1,0);
    AlphaTracklets13_->SetBinContent(9,14,2,0);
    AlphaTracklets13_->SetBinContent(9,14,3,0);
    AlphaTracklets13_->SetBinContent(9,14,4,0);
    AlphaTracklets13_->SetBinContent(9,14,5,0);
    AlphaTracklets13_->SetBinContent(9,14,6,0);
    AlphaTracklets13_->SetBinContent(9,14,7,0);
    AlphaTracklets13_->SetBinContent(9,14,8,0);
    AlphaTracklets13_->SetBinContent(9,14,9,0);
    AlphaTracklets13_->SetBinContent(9,14,10,0);
    AlphaTracklets13_->SetBinContent(10,1,1,1.26378);
    AlphaTracklets13_->SetBinContent(10,1,2,1.3477);
    AlphaTracklets13_->SetBinContent(10,1,3,1.58208);
    AlphaTracklets13_->SetBinContent(10,1,4,1.96004);
    AlphaTracklets13_->SetBinContent(10,1,5,2.52438);
    AlphaTracklets13_->SetBinContent(10,1,6,3.56489);
    AlphaTracklets13_->SetBinContent(10,2,1,1.26176);
    AlphaTracklets13_->SetBinContent(10,2,2,1.37037);
    AlphaTracklets13_->SetBinContent(10,2,3,1.61734);
    AlphaTracklets13_->SetBinContent(10,2,4,1.98407);
    AlphaTracklets13_->SetBinContent(10,2,5,2.47862);
    AlphaTracklets13_->SetBinContent(10,2,6,3.52909);
    AlphaTracklets13_->SetBinContent(10,3,1,1.2607);
    AlphaTracklets13_->SetBinContent(10,3,2,1.39274);
    AlphaTracklets13_->SetBinContent(10,3,3,1.67065);
    AlphaTracklets13_->SetBinContent(10,3,4,1.97691);
    AlphaTracklets13_->SetBinContent(10,3,5,2.51316);
    AlphaTracklets13_->SetBinContent(10,3,6,3.61732);
    AlphaTracklets13_->SetBinContent(10,4,1,1.24389);
    AlphaTracklets13_->SetBinContent(10,4,2,1.43128);
    AlphaTracklets13_->SetBinContent(10,4,3,1.71576);
    AlphaTracklets13_->SetBinContent(10,4,4,2.05526);
    AlphaTracklets13_->SetBinContent(10,4,5,2.55025);
    AlphaTracklets13_->SetBinContent(10,4,6,3.61829);
    AlphaTracklets13_->SetBinContent(10,5,1,1.34209);
    AlphaTracklets13_->SetBinContent(10,5,2,1.44629);
    AlphaTracklets13_->SetBinContent(10,5,3,1.73771);
    AlphaTracklets13_->SetBinContent(10,5,4,2.1319);
    AlphaTracklets13_->SetBinContent(10,5,5,2.6342);
    AlphaTracklets13_->SetBinContent(10,5,6,3.69833);
    AlphaTracklets13_->SetBinContent(10,6,1,1.32409);
    AlphaTracklets13_->SetBinContent(10,6,2,1.4826);
    AlphaTracklets13_->SetBinContent(10,6,3,1.7541);
    AlphaTracklets13_->SetBinContent(10,6,4,2.12247);
    AlphaTracklets13_->SetBinContent(10,6,5,2.61113);
    AlphaTracklets13_->SetBinContent(10,6,6,3.84614);
    AlphaTracklets13_->SetBinContent(10,7,1,1.28261);
    AlphaTracklets13_->SetBinContent(10,7,2,1.52668);
    AlphaTracklets13_->SetBinContent(10,7,3,1.8008);
    AlphaTracklets13_->SetBinContent(10,7,4,2.19908);
    AlphaTracklets13_->SetBinContent(10,7,5,2.66566);
    AlphaTracklets13_->SetBinContent(10,7,6,3.8218);
    AlphaTracklets13_->SetBinContent(10,8,1,1.35641);
    AlphaTracklets13_->SetBinContent(10,8,2,1.55592);
    AlphaTracklets13_->SetBinContent(10,8,3,1.84557);
    AlphaTracklets13_->SetBinContent(10,8,4,2.20245);
    AlphaTracklets13_->SetBinContent(10,8,5,2.73373);
    AlphaTracklets13_->SetBinContent(10,8,6,3.79148);
    AlphaTracklets13_->SetBinContent(10,9,1,1.38532);
    AlphaTracklets13_->SetBinContent(10,9,2,1.52899);
    AlphaTracklets13_->SetBinContent(10,9,3,1.85795);
    AlphaTracklets13_->SetBinContent(10,9,4,2.26173);
    AlphaTracklets13_->SetBinContent(10,9,5,2.77377);
    AlphaTracklets13_->SetBinContent(10,9,6,3.86538);
    AlphaTracklets13_->SetBinContent(10,10,1,1.45584);
    AlphaTracklets13_->SetBinContent(10,10,2,1.61802);
    AlphaTracklets13_->SetBinContent(10,10,3,1.89146);
    AlphaTracklets13_->SetBinContent(10,10,4,2.30873);
    AlphaTracklets13_->SetBinContent(10,10,5,2.91823);
    AlphaTracklets13_->SetBinContent(10,10,6,3.98149);
    AlphaTracklets13_->SetBinContent(10,11,1,1.49549);
    AlphaTracklets13_->SetBinContent(10,11,2,1.64919);
    AlphaTracklets13_->SetBinContent(10,11,3,1.96845);
    AlphaTracklets13_->SetBinContent(10,11,4,2.41492);
    AlphaTracklets13_->SetBinContent(10,11,5,2.92894);
    AlphaTracklets13_->SetBinContent(10,11,6,4.12848);
    AlphaTracklets13_->SetBinContent(10,12,1,1.63536);
    AlphaTracklets13_->SetBinContent(10,12,2,1.67105);
    AlphaTracklets13_->SetBinContent(10,12,3,2.13208);
    AlphaTracklets13_->SetBinContent(10,12,4,2.5402);
    AlphaTracklets13_->SetBinContent(10,12,5,2.9162);
    AlphaTracklets13_->SetBinContent(10,12,6,4.42421);
    AlphaTracklets13_->SetBinContent(10,13,1,1.36364);
    AlphaTracklets13_->SetBinContent(10,13,2,2.06522);
    AlphaTracklets13_->SetBinContent(10,13,3,2.17341);
    AlphaTracklets13_->SetBinContent(10,13,4,2.25685);
    AlphaTracklets13_->SetBinContent(10,13,5,3.39614);
    AlphaTracklets13_->SetBinContent(10,13,6,4.90052);
    AlphaTracklets13_->SetBinContent(10,14,1,0);
    AlphaTracklets13_->SetBinContent(10,14,2,0);
    AlphaTracklets13_->SetBinContent(10,14,3,0);
    AlphaTracklets13_->SetBinContent(10,14,4,0);
    AlphaTracklets13_->SetBinContent(10,14,5,0);
    AlphaTracklets13_->SetBinContent(10,14,6,0);
  }

  if (which>=23) {

    const int nEtaBin = 12;
    const int nHitBin = 14;
    const int nVzBin  = 10;

    double HitBins[nHitBin+1] = {0,5,10,15,20,25,30,35,40,50,60,80,100,200,700};

    double EtaBins[nEtaBin+1];
    for (int i=0;i<=nEtaBin;i++)
      EtaBins[i] = (double)i*6.0/(double)nEtaBin-3.0;
    double VzBins[nVzBin+1];
    for (int i=0;i<=nVzBin;i++)
      VzBins[i] = (double)i*20.0/(double)nVzBin-10.0;

    AlphaTracklets23_ = new TH3F("hAlphaTracklets23",
                                 "Alpha for tracklets23;#eta;#hits;vz [cm]",
                                 nEtaBin, EtaBins, nHitBin, HitBins, nVzBin, VzBins);
    AlphaTracklets23_->SetDirectory(0);

    AlphaTracklets23_->SetBinContent(3,1,5,3.38308);
    AlphaTracklets23_->SetBinContent(3,1,6,2.34772);
    AlphaTracklets23_->SetBinContent(3,1,7,1.88945);
    AlphaTracklets23_->SetBinContent(3,1,8,1.54388);
    AlphaTracklets23_->SetBinContent(3,1,9,1.27955);
    AlphaTracklets23_->SetBinContent(3,1,10,1.10816);
    AlphaTracklets23_->SetBinContent(3,2,5,3.27884);
    AlphaTracklets23_->SetBinContent(3,2,6,2.22303);
    AlphaTracklets23_->SetBinContent(3,2,7,1.81699);
    AlphaTracklets23_->SetBinContent(3,2,8,1.49227);
    AlphaTracklets23_->SetBinContent(3,2,9,1.25156);
    AlphaTracklets23_->SetBinContent(3,2,10,1.12269);
    AlphaTracklets23_->SetBinContent(3,3,5,3.20933);
    AlphaTracklets23_->SetBinContent(3,3,6,2.24795);
    AlphaTracklets23_->SetBinContent(3,3,7,1.79453);
    AlphaTracklets23_->SetBinContent(3,3,8,1.49351);
    AlphaTracklets23_->SetBinContent(3,3,9,1.25269);
    AlphaTracklets23_->SetBinContent(3,3,10,1.16437);
    AlphaTracklets23_->SetBinContent(3,4,5,3.22097);
    AlphaTracklets23_->SetBinContent(3,4,6,2.24832);
    AlphaTracklets23_->SetBinContent(3,4,7,1.81552);
    AlphaTracklets23_->SetBinContent(3,4,8,1.48839);
    AlphaTracklets23_->SetBinContent(3,4,9,1.27005);
    AlphaTracklets23_->SetBinContent(3,4,10,1.14292);
    AlphaTracklets23_->SetBinContent(3,5,5,3.26176);
    AlphaTracklets23_->SetBinContent(3,5,6,2.29334);
    AlphaTracklets23_->SetBinContent(3,5,7,1.80412);
    AlphaTracklets23_->SetBinContent(3,5,8,1.51012);
    AlphaTracklets23_->SetBinContent(3,5,9,1.27943);
    AlphaTracklets23_->SetBinContent(3,5,10,1.16142);
    AlphaTracklets23_->SetBinContent(3,6,5,3.22703);
    AlphaTracklets23_->SetBinContent(3,6,6,2.27187);
    AlphaTracklets23_->SetBinContent(3,6,7,1.81449);
    AlphaTracklets23_->SetBinContent(3,6,8,1.50404);
    AlphaTracklets23_->SetBinContent(3,6,9,1.2876);
    AlphaTracklets23_->SetBinContent(3,6,10,1.1662);
    AlphaTracklets23_->SetBinContent(3,7,5,3.21015);
    AlphaTracklets23_->SetBinContent(3,7,6,2.29223);
    AlphaTracklets23_->SetBinContent(3,7,7,1.84044);
    AlphaTracklets23_->SetBinContent(3,7,8,1.5401);
    AlphaTracklets23_->SetBinContent(3,7,9,1.28395);
    AlphaTracklets23_->SetBinContent(3,7,10,1.17095);
    AlphaTracklets23_->SetBinContent(3,8,5,3.33452);
    AlphaTracklets23_->SetBinContent(3,8,6,2.37071);
    AlphaTracklets23_->SetBinContent(3,8,7,1.8849);
    AlphaTracklets23_->SetBinContent(3,8,8,1.55687);
    AlphaTracklets23_->SetBinContent(3,8,9,1.33145);
    AlphaTracklets23_->SetBinContent(3,8,10,1.19818);
    AlphaTracklets23_->SetBinContent(3,9,5,3.32477);
    AlphaTracklets23_->SetBinContent(3,9,6,2.38881);
    AlphaTracklets23_->SetBinContent(3,9,7,1.8645);
    AlphaTracklets23_->SetBinContent(3,9,8,1.54367);
    AlphaTracklets23_->SetBinContent(3,9,9,1.38023);
    AlphaTracklets23_->SetBinContent(3,9,10,1.17575);
    AlphaTracklets23_->SetBinContent(3,10,5,3.27017);
    AlphaTracklets23_->SetBinContent(3,10,6,2.37089);
    AlphaTracklets23_->SetBinContent(3,10,7,1.88249);
    AlphaTracklets23_->SetBinContent(3,10,8,1.56901);
    AlphaTracklets23_->SetBinContent(3,10,9,1.30542);
    AlphaTracklets23_->SetBinContent(3,10,10,1.18853);
    AlphaTracklets23_->SetBinContent(3,11,5,3.30431);
    AlphaTracklets23_->SetBinContent(3,11,6,2.4801);
    AlphaTracklets23_->SetBinContent(3,11,7,1.91068);
    AlphaTracklets23_->SetBinContent(3,11,8,1.53834);
    AlphaTracklets23_->SetBinContent(3,11,9,1.3843);
    AlphaTracklets23_->SetBinContent(3,11,10,1.22313);
    AlphaTracklets23_->SetBinContent(3,12,5,3.50197);
    AlphaTracklets23_->SetBinContent(3,12,6,2.41644);
    AlphaTracklets23_->SetBinContent(3,12,7,1.89029);
    AlphaTracklets23_->SetBinContent(3,12,8,1.52287);
    AlphaTracklets23_->SetBinContent(3,12,9,1.79154);
    AlphaTracklets23_->SetBinContent(3,12,10,1.07229);
    AlphaTracklets23_->SetBinContent(3,13,5,3.75194);
    AlphaTracklets23_->SetBinContent(3,13,6,2.26267);
    AlphaTracklets23_->SetBinContent(3,13,7,1.92417);
    AlphaTracklets23_->SetBinContent(3,13,8,1.47682);
    AlphaTracklets23_->SetBinContent(3,13,9,1.25556);
    AlphaTracklets23_->SetBinContent(3,13,10,2.93333);
    AlphaTracklets23_->SetBinContent(3,14,5,0);
    AlphaTracklets23_->SetBinContent(3,14,6,0);
    AlphaTracklets23_->SetBinContent(3,14,7,0);
    AlphaTracklets23_->SetBinContent(3,14,8,0);
    AlphaTracklets23_->SetBinContent(3,14,9,0);
    AlphaTracklets23_->SetBinContent(3,14,10,0);
    AlphaTracklets23_->SetBinContent(4,1,1,1.66234);
    AlphaTracklets23_->SetBinContent(4,1,2,1.26118);
    AlphaTracklets23_->SetBinContent(4,1,3,1.10768);
    AlphaTracklets23_->SetBinContent(4,1,4,1.07616);
    AlphaTracklets23_->SetBinContent(4,1,5,1.07452);
    AlphaTracklets23_->SetBinContent(4,1,6,1.08175);
    AlphaTracklets23_->SetBinContent(4,1,7,1.07495);
    AlphaTracklets23_->SetBinContent(4,1,8,1.07746);
    AlphaTracklets23_->SetBinContent(4,1,9,1.08099);
    AlphaTracklets23_->SetBinContent(4,1,10,1.06863);
    AlphaTracklets23_->SetBinContent(4,2,1,1.59117);
    AlphaTracklets23_->SetBinContent(4,2,2,1.28544);
    AlphaTracklets23_->SetBinContent(4,2,3,1.07882);
    AlphaTracklets23_->SetBinContent(4,2,4,1.06364);
    AlphaTracklets23_->SetBinContent(4,2,5,1.07065);
    AlphaTracklets23_->SetBinContent(4,2,6,1.05421);
    AlphaTracklets23_->SetBinContent(4,2,7,1.05212);
    AlphaTracklets23_->SetBinContent(4,2,8,1.02743);
    AlphaTracklets23_->SetBinContent(4,2,9,1.0502);
    AlphaTracklets23_->SetBinContent(4,2,10,1.07346);
    AlphaTracklets23_->SetBinContent(4,3,1,1.58881);
    AlphaTracklets23_->SetBinContent(4,3,2,1.2617);
    AlphaTracklets23_->SetBinContent(4,3,3,1.08782);
    AlphaTracklets23_->SetBinContent(4,3,4,1.04729);
    AlphaTracklets23_->SetBinContent(4,3,5,1.06077);
    AlphaTracklets23_->SetBinContent(4,3,6,1.05053);
    AlphaTracklets23_->SetBinContent(4,3,7,1.04896);
    AlphaTracklets23_->SetBinContent(4,3,8,1.04622);
    AlphaTracklets23_->SetBinContent(4,3,9,1.0552);
    AlphaTracklets23_->SetBinContent(4,3,10,1.06174);
    AlphaTracklets23_->SetBinContent(4,4,1,1.55942);
    AlphaTracklets23_->SetBinContent(4,4,2,1.24904);
    AlphaTracklets23_->SetBinContent(4,4,3,1.07903);
    AlphaTracklets23_->SetBinContent(4,4,4,1.05812);
    AlphaTracklets23_->SetBinContent(4,4,5,1.07259);
    AlphaTracklets23_->SetBinContent(4,4,6,1.06213);
    AlphaTracklets23_->SetBinContent(4,4,7,1.05845);
    AlphaTracklets23_->SetBinContent(4,4,8,1.05147);
    AlphaTracklets23_->SetBinContent(4,4,9,1.04473);
    AlphaTracklets23_->SetBinContent(4,4,10,1.03457);
    AlphaTracklets23_->SetBinContent(4,5,1,1.62918);
    AlphaTracklets23_->SetBinContent(4,5,2,1.26643);
    AlphaTracklets23_->SetBinContent(4,5,3,1.08997);
    AlphaTracklets23_->SetBinContent(4,5,4,1.07264);
    AlphaTracklets23_->SetBinContent(4,5,5,1.07627);
    AlphaTracklets23_->SetBinContent(4,5,6,1.06224);
    AlphaTracklets23_->SetBinContent(4,5,7,1.06504);
    AlphaTracklets23_->SetBinContent(4,5,8,1.05417);
    AlphaTracklets23_->SetBinContent(4,5,9,1.06618);
    AlphaTracklets23_->SetBinContent(4,5,10,1.02044);
    AlphaTracklets23_->SetBinContent(4,6,1,1.59361);
    AlphaTracklets23_->SetBinContent(4,6,2,1.23262);
    AlphaTracklets23_->SetBinContent(4,6,3,1.09599);
    AlphaTracklets23_->SetBinContent(4,6,4,1.06615);
    AlphaTracklets23_->SetBinContent(4,6,5,1.07959);
    AlphaTracklets23_->SetBinContent(4,6,6,1.06501);
    AlphaTracklets23_->SetBinContent(4,6,7,1.05996);
    AlphaTracklets23_->SetBinContent(4,6,8,1.0459);
    AlphaTracklets23_->SetBinContent(4,6,9,1.06233);
    AlphaTracklets23_->SetBinContent(4,6,10,1.07382);
    AlphaTracklets23_->SetBinContent(4,7,1,1.59904);
    AlphaTracklets23_->SetBinContent(4,7,2,1.30261);
    AlphaTracklets23_->SetBinContent(4,7,3,1.10598);
    AlphaTracklets23_->SetBinContent(4,7,4,1.0789);
    AlphaTracklets23_->SetBinContent(4,7,5,1.09548);
    AlphaTracklets23_->SetBinContent(4,7,6,1.06718);
    AlphaTracklets23_->SetBinContent(4,7,7,1.06925);
    AlphaTracklets23_->SetBinContent(4,7,8,1.05837);
    AlphaTracklets23_->SetBinContent(4,7,9,1.07298);
    AlphaTracklets23_->SetBinContent(4,7,10,1.08098);
    AlphaTracklets23_->SetBinContent(4,8,1,1.52607);
    AlphaTracklets23_->SetBinContent(4,8,2,1.33542);
    AlphaTracklets23_->SetBinContent(4,8,3,1.09118);
    AlphaTracklets23_->SetBinContent(4,8,4,1.09021);
    AlphaTracklets23_->SetBinContent(4,8,5,1.08725);
    AlphaTracklets23_->SetBinContent(4,8,6,1.08714);
    AlphaTracklets23_->SetBinContent(4,8,7,1.07986);
    AlphaTracklets23_->SetBinContent(4,8,8,1.05784);
    AlphaTracklets23_->SetBinContent(4,8,9,1.07345);
    AlphaTracklets23_->SetBinContent(4,8,10,1.12906);
    AlphaTracklets23_->SetBinContent(4,9,1,1.62233);
    AlphaTracklets23_->SetBinContent(4,9,2,1.301);
    AlphaTracklets23_->SetBinContent(4,9,3,1.09737);
    AlphaTracklets23_->SetBinContent(4,9,4,1.08925);
    AlphaTracklets23_->SetBinContent(4,9,5,1.09619);
    AlphaTracklets23_->SetBinContent(4,9,6,1.08347);
    AlphaTracklets23_->SetBinContent(4,9,7,1.08154);
    AlphaTracklets23_->SetBinContent(4,9,8,1.07991);
    AlphaTracklets23_->SetBinContent(4,9,9,1.06076);
    AlphaTracklets23_->SetBinContent(4,9,10,1.06918);
    AlphaTracklets23_->SetBinContent(4,10,1,1.66667);
    AlphaTracklets23_->SetBinContent(4,10,2,1.28708);
    AlphaTracklets23_->SetBinContent(4,10,3,1.11371);
    AlphaTracklets23_->SetBinContent(4,10,4,1.10948);
    AlphaTracklets23_->SetBinContent(4,10,5,1.112);
    AlphaTracklets23_->SetBinContent(4,10,6,1.09294);
    AlphaTracklets23_->SetBinContent(4,10,7,1.09422);
    AlphaTracklets23_->SetBinContent(4,10,8,1.0633);
    AlphaTracklets23_->SetBinContent(4,10,9,1.07525);
    AlphaTracklets23_->SetBinContent(4,10,10,1.0898);
    AlphaTracklets23_->SetBinContent(4,11,1,1.71024);
    AlphaTracklets23_->SetBinContent(4,11,2,1.31985);
    AlphaTracklets23_->SetBinContent(4,11,3,1.13869);
    AlphaTracklets23_->SetBinContent(4,11,4,1.10237);
    AlphaTracklets23_->SetBinContent(4,11,5,1.11925);
    AlphaTracklets23_->SetBinContent(4,11,6,1.0967);
    AlphaTracklets23_->SetBinContent(4,11,7,1.09463);
    AlphaTracklets23_->SetBinContent(4,11,8,1.07578);
    AlphaTracklets23_->SetBinContent(4,11,9,1.06029);
    AlphaTracklets23_->SetBinContent(4,11,10,1.05017);
    AlphaTracklets23_->SetBinContent(4,12,1,1.58824);
    AlphaTracklets23_->SetBinContent(4,12,2,1.25057);
    AlphaTracklets23_->SetBinContent(4,12,3,1.12118);
    AlphaTracklets23_->SetBinContent(4,12,4,1.12959);
    AlphaTracklets23_->SetBinContent(4,12,5,1.12079);
    AlphaTracklets23_->SetBinContent(4,12,6,1.13405);
    AlphaTracklets23_->SetBinContent(4,12,7,1.08382);
    AlphaTracklets23_->SetBinContent(4,12,8,1.06474);
    AlphaTracklets23_->SetBinContent(4,12,9,1.15842);
    AlphaTracklets23_->SetBinContent(4,12,10,1.19745);
    AlphaTracklets23_->SetBinContent(4,13,1,2.36842);
    AlphaTracklets23_->SetBinContent(4,13,2,1.32353);
    AlphaTracklets23_->SetBinContent(4,13,3,1.30108);
    AlphaTracklets23_->SetBinContent(4,13,4,1.13613);
    AlphaTracklets23_->SetBinContent(4,13,5,1.11489);
    AlphaTracklets23_->SetBinContent(4,13,6,1.08842);
    AlphaTracklets23_->SetBinContent(4,13,7,1.2153);
    AlphaTracklets23_->SetBinContent(4,13,8,1.02222);
    AlphaTracklets23_->SetBinContent(4,13,9,1.15842);
    AlphaTracklets23_->SetBinContent(4,13,10,2.55556);
    AlphaTracklets23_->SetBinContent(4,14,1,0);
    AlphaTracklets23_->SetBinContent(4,14,2,0);
    AlphaTracklets23_->SetBinContent(4,14,3,0);
    AlphaTracklets23_->SetBinContent(4,14,4,0);
    AlphaTracklets23_->SetBinContent(4,14,5,0);
    AlphaTracklets23_->SetBinContent(4,14,6,0);
    AlphaTracklets23_->SetBinContent(4,14,7,0);
    AlphaTracklets23_->SetBinContent(4,14,8,0);
    AlphaTracklets23_->SetBinContent(4,14,9,0);
    AlphaTracklets23_->SetBinContent(4,14,10,0);
    AlphaTracklets23_->SetBinContent(5,1,1,1.07519);
    AlphaTracklets23_->SetBinContent(5,1,2,1.13624);
    AlphaTracklets23_->SetBinContent(5,1,3,1.06278);
    AlphaTracklets23_->SetBinContent(5,1,4,1.07285);
    AlphaTracklets23_->SetBinContent(5,1,5,1.1018);
    AlphaTracklets23_->SetBinContent(5,1,6,1.09199);
    AlphaTracklets23_->SetBinContent(5,1,7,1.05803);
    AlphaTracklets23_->SetBinContent(5,1,8,1.10688);
    AlphaTracklets23_->SetBinContent(5,1,9,1.12112);
    AlphaTracklets23_->SetBinContent(5,1,10,1.12885);
    AlphaTracklets23_->SetBinContent(5,2,1,1.08115);
    AlphaTracklets23_->SetBinContent(5,2,2,1.06308);
    AlphaTracklets23_->SetBinContent(5,2,3,1.03966);
    AlphaTracklets23_->SetBinContent(5,2,4,1.04387);
    AlphaTracklets23_->SetBinContent(5,2,5,1.0662);
    AlphaTracklets23_->SetBinContent(5,2,6,1.06068);
    AlphaTracklets23_->SetBinContent(5,2,7,1.02691);
    AlphaTracklets23_->SetBinContent(5,2,8,1.08253);
    AlphaTracklets23_->SetBinContent(5,2,9,1.08882);
    AlphaTracklets23_->SetBinContent(5,2,10,1.0939);
    AlphaTracklets23_->SetBinContent(5,3,1,1.10936);
    AlphaTracklets23_->SetBinContent(5,3,2,1.08249);
    AlphaTracklets23_->SetBinContent(5,3,3,1.04052);
    AlphaTracklets23_->SetBinContent(5,3,4,1.04158);
    AlphaTracklets23_->SetBinContent(5,3,5,1.05562);
    AlphaTracklets23_->SetBinContent(5,3,6,1.04468);
    AlphaTracklets23_->SetBinContent(5,3,7,1.01801);
    AlphaTracklets23_->SetBinContent(5,3,8,1.06763);
    AlphaTracklets23_->SetBinContent(5,3,9,1.06207);
    AlphaTracklets23_->SetBinContent(5,3,10,1.06836);
    AlphaTracklets23_->SetBinContent(5,4,1,1.08569);
    AlphaTracklets23_->SetBinContent(5,4,2,1.07918);
    AlphaTracklets23_->SetBinContent(5,4,3,1.03785);
    AlphaTracklets23_->SetBinContent(5,4,4,1.03212);
    AlphaTracklets23_->SetBinContent(5,4,5,1.06335);
    AlphaTracklets23_->SetBinContent(5,4,6,1.05043);
    AlphaTracklets23_->SetBinContent(5,4,7,1.01704);
    AlphaTracklets23_->SetBinContent(5,4,8,1.05117);
    AlphaTracklets23_->SetBinContent(5,4,9,1.05976);
    AlphaTracklets23_->SetBinContent(5,4,10,1.07752);
    AlphaTracklets23_->SetBinContent(5,5,1,1.06866);
    AlphaTracklets23_->SetBinContent(5,5,2,1.0878);
    AlphaTracklets23_->SetBinContent(5,5,3,1.04593);
    AlphaTracklets23_->SetBinContent(5,5,4,1.03797);
    AlphaTracklets23_->SetBinContent(5,5,5,1.05119);
    AlphaTracklets23_->SetBinContent(5,5,6,1.04013);
    AlphaTracklets23_->SetBinContent(5,5,7,1.00307);
    AlphaTracklets23_->SetBinContent(5,5,8,1.05596);
    AlphaTracklets23_->SetBinContent(5,5,9,1.06156);
    AlphaTracklets23_->SetBinContent(5,5,10,1.03935);
    AlphaTracklets23_->SetBinContent(5,6,1,1.05679);
    AlphaTracklets23_->SetBinContent(5,6,2,1.10414);
    AlphaTracklets23_->SetBinContent(5,6,3,1.04278);
    AlphaTracklets23_->SetBinContent(5,6,4,1.0353);
    AlphaTracklets23_->SetBinContent(5,6,5,1.06145);
    AlphaTracklets23_->SetBinContent(5,6,6,1.05219);
    AlphaTracklets23_->SetBinContent(5,6,7,1.01212);
    AlphaTracklets23_->SetBinContent(5,6,8,1.05899);
    AlphaTracklets23_->SetBinContent(5,6,9,1.06845);
    AlphaTracklets23_->SetBinContent(5,6,10,1.05764);
    AlphaTracklets23_->SetBinContent(5,7,1,1.05552);
    AlphaTracklets23_->SetBinContent(5,7,2,1.07301);
    AlphaTracklets23_->SetBinContent(5,7,3,1.06149);
    AlphaTracklets23_->SetBinContent(5,7,4,1.03864);
    AlphaTracklets23_->SetBinContent(5,7,5,1.05961);
    AlphaTracklets23_->SetBinContent(5,7,6,1.05192);
    AlphaTracklets23_->SetBinContent(5,7,7,1.02022);
    AlphaTracklets23_->SetBinContent(5,7,8,1.05235);
    AlphaTracklets23_->SetBinContent(5,7,9,1.04868);
    AlphaTracklets23_->SetBinContent(5,7,10,1.03455);
    AlphaTracklets23_->SetBinContent(5,8,1,1.03896);
    AlphaTracklets23_->SetBinContent(5,8,2,1.09401);
    AlphaTracklets23_->SetBinContent(5,8,3,1.04782);
    AlphaTracklets23_->SetBinContent(5,8,4,1.03441);
    AlphaTracklets23_->SetBinContent(5,8,5,1.06046);
    AlphaTracklets23_->SetBinContent(5,8,6,1.05002);
    AlphaTracklets23_->SetBinContent(5,8,7,1.0051);
    AlphaTracklets23_->SetBinContent(5,8,8,1.05984);
    AlphaTracklets23_->SetBinContent(5,8,9,1.05247);
    AlphaTracklets23_->SetBinContent(5,8,10,1.08816);
    AlphaTracklets23_->SetBinContent(5,9,1,1.13531);
    AlphaTracklets23_->SetBinContent(5,9,2,1.08398);
    AlphaTracklets23_->SetBinContent(5,9,3,1.03833);
    AlphaTracklets23_->SetBinContent(5,9,4,1.0406);
    AlphaTracklets23_->SetBinContent(5,9,5,1.06394);
    AlphaTracklets23_->SetBinContent(5,9,6,1.05222);
    AlphaTracklets23_->SetBinContent(5,9,7,1.01707);
    AlphaTracklets23_->SetBinContent(5,9,8,1.05628);
    AlphaTracklets23_->SetBinContent(5,9,9,1.06462);
    AlphaTracklets23_->SetBinContent(5,9,10,1.02194);
    AlphaTracklets23_->SetBinContent(5,10,1,1.13355);
    AlphaTracklets23_->SetBinContent(5,10,2,1.10368);
    AlphaTracklets23_->SetBinContent(5,10,3,1.034);
    AlphaTracklets23_->SetBinContent(5,10,4,1.04582);
    AlphaTracklets23_->SetBinContent(5,10,5,1.06188);
    AlphaTracklets23_->SetBinContent(5,10,6,1.04183);
    AlphaTracklets23_->SetBinContent(5,10,7,1.00837);
    AlphaTracklets23_->SetBinContent(5,10,8,1.05649);
    AlphaTracklets23_->SetBinContent(5,10,9,1.07756);
    AlphaTracklets23_->SetBinContent(5,10,10,1.04647);
    AlphaTracklets23_->SetBinContent(5,11,1,1.02366);
    AlphaTracklets23_->SetBinContent(5,11,2,1.12954);
    AlphaTracklets23_->SetBinContent(5,11,3,1.02578);
    AlphaTracklets23_->SetBinContent(5,11,4,1.03935);
    AlphaTracklets23_->SetBinContent(5,11,5,1.04506);
    AlphaTracklets23_->SetBinContent(5,11,6,1.0557);
    AlphaTracklets23_->SetBinContent(5,11,7,0.999213);
    AlphaTracklets23_->SetBinContent(5,11,8,1.06034);
    AlphaTracklets23_->SetBinContent(5,11,9,1.02358);
    AlphaTracklets23_->SetBinContent(5,11,10,1.01854);
    AlphaTracklets23_->SetBinContent(5,12,1,1.16667);
    AlphaTracklets23_->SetBinContent(5,12,2,1.075);
    AlphaTracklets23_->SetBinContent(5,12,3,1.08243);
    AlphaTracklets23_->SetBinContent(5,12,4,1.0187);
    AlphaTracklets23_->SetBinContent(5,12,5,1.06535);
    AlphaTracklets23_->SetBinContent(5,12,6,1.0175);
    AlphaTracklets23_->SetBinContent(5,12,7,1.05553);
    AlphaTracklets23_->SetBinContent(5,12,8,1.07045);
    AlphaTracklets23_->SetBinContent(5,12,9,1.00704);
    AlphaTracklets23_->SetBinContent(5,12,10,1.06494);
    AlphaTracklets23_->SetBinContent(5,13,1,0.945455);
    AlphaTracklets23_->SetBinContent(5,13,2,0.964646);
    AlphaTracklets23_->SetBinContent(5,13,3,0.960474);
    AlphaTracklets23_->SetBinContent(5,13,4,1.15196);
    AlphaTracklets23_->SetBinContent(5,13,5,1.09109);
    AlphaTracklets23_->SetBinContent(5,13,6,0.981557);
    AlphaTracklets23_->SetBinContent(5,13,7,0.972028);
    AlphaTracklets23_->SetBinContent(5,13,8,1.16098);
    AlphaTracklets23_->SetBinContent(5,13,9,1.104);
    AlphaTracklets23_->SetBinContent(5,13,10,1.01923);
    AlphaTracklets23_->SetBinContent(5,14,1,0);
    AlphaTracklets23_->SetBinContent(5,14,2,0);
    AlphaTracklets23_->SetBinContent(5,14,3,0);
    AlphaTracklets23_->SetBinContent(5,14,4,0);
    AlphaTracklets23_->SetBinContent(5,14,5,0);
    AlphaTracklets23_->SetBinContent(5,14,6,0);
    AlphaTracklets23_->SetBinContent(5,14,7,0);
    AlphaTracklets23_->SetBinContent(5,14,8,0);
    AlphaTracklets23_->SetBinContent(5,14,9,0);
    AlphaTracklets23_->SetBinContent(5,14,10,0);
    AlphaTracklets23_->SetBinContent(6,1,1,1.05793);
    AlphaTracklets23_->SetBinContent(6,1,2,1.07134);
    AlphaTracklets23_->SetBinContent(6,1,3,1.16732);
    AlphaTracklets23_->SetBinContent(6,1,4,1.14014);
    AlphaTracklets23_->SetBinContent(6,1,5,1.04884);
    AlphaTracklets23_->SetBinContent(6,1,6,1.13906);
    AlphaTracklets23_->SetBinContent(6,1,7,1.18691);
    AlphaTracklets23_->SetBinContent(6,1,8,1.08443);
    AlphaTracklets23_->SetBinContent(6,1,9,1.13503);
    AlphaTracklets23_->SetBinContent(6,1,10,1.17293);
    AlphaTracklets23_->SetBinContent(6,2,1,1.04901);
    AlphaTracklets23_->SetBinContent(6,2,2,1.03686);
    AlphaTracklets23_->SetBinContent(6,2,3,1.09848);
    AlphaTracklets23_->SetBinContent(6,2,4,1.06087);
    AlphaTracklets23_->SetBinContent(6,2,5,1.00599);
    AlphaTracklets23_->SetBinContent(6,2,6,1.09695);
    AlphaTracklets23_->SetBinContent(6,2,7,1.10906);
    AlphaTracklets23_->SetBinContent(6,2,8,1.0311);
    AlphaTracklets23_->SetBinContent(6,2,9,1.05796);
    AlphaTracklets23_->SetBinContent(6,2,10,1.04862);
    AlphaTracklets23_->SetBinContent(6,3,1,1.01846);
    AlphaTracklets23_->SetBinContent(6,3,2,1.01445);
    AlphaTracklets23_->SetBinContent(6,3,3,1.08326);
    AlphaTracklets23_->SetBinContent(6,3,4,1.03821);
    AlphaTracklets23_->SetBinContent(6,3,5,0.98838);
    AlphaTracklets23_->SetBinContent(6,3,6,1.0744);
    AlphaTracklets23_->SetBinContent(6,3,7,1.08359);
    AlphaTracklets23_->SetBinContent(6,3,8,1.02623);
    AlphaTracklets23_->SetBinContent(6,3,9,1.02682);
    AlphaTracklets23_->SetBinContent(6,3,10,1.0638);
    AlphaTracklets23_->SetBinContent(6,4,1,1.01264);
    AlphaTracklets23_->SetBinContent(6,4,2,1.00226);
    AlphaTracklets23_->SetBinContent(6,4,3,1.0643);
    AlphaTracklets23_->SetBinContent(6,4,4,1.04177);
    AlphaTracklets23_->SetBinContent(6,4,5,0.978268);
    AlphaTracklets23_->SetBinContent(6,4,6,1.06026);
    AlphaTracklets23_->SetBinContent(6,4,7,1.07406);
    AlphaTracklets23_->SetBinContent(6,4,8,1.01174);
    AlphaTracklets23_->SetBinContent(6,4,9,1.03664);
    AlphaTracklets23_->SetBinContent(6,4,10,1.10756);
    AlphaTracklets23_->SetBinContent(6,5,1,1.00293);
    AlphaTracklets23_->SetBinContent(6,5,2,1.02175);
    AlphaTracklets23_->SetBinContent(6,5,3,1.05971);
    AlphaTracklets23_->SetBinContent(6,5,4,1.02092);
    AlphaTracklets23_->SetBinContent(6,5,5,0.969199);
    AlphaTracklets23_->SetBinContent(6,5,6,1.06256);
    AlphaTracklets23_->SetBinContent(6,5,7,1.0566);
    AlphaTracklets23_->SetBinContent(6,5,8,0.996387);
    AlphaTracklets23_->SetBinContent(6,5,9,1.02389);
    AlphaTracklets23_->SetBinContent(6,5,10,1.07895);
    AlphaTracklets23_->SetBinContent(6,6,1,0.981318);
    AlphaTracklets23_->SetBinContent(6,6,2,0.990927);
    AlphaTracklets23_->SetBinContent(6,6,3,1.05136);
    AlphaTracklets23_->SetBinContent(6,6,4,1.01575);
    AlphaTracklets23_->SetBinContent(6,6,5,0.964321);
    AlphaTracklets23_->SetBinContent(6,6,6,1.05229);
    AlphaTracklets23_->SetBinContent(6,6,7,1.0521);
    AlphaTracklets23_->SetBinContent(6,6,8,0.997108);
    AlphaTracklets23_->SetBinContent(6,6,9,1.03548);
    AlphaTracklets23_->SetBinContent(6,6,10,1.10448);
    AlphaTracklets23_->SetBinContent(6,7,1,1.02195);
    AlphaTracklets23_->SetBinContent(6,7,2,0.98838);
    AlphaTracklets23_->SetBinContent(6,7,3,1.04718);
    AlphaTracklets23_->SetBinContent(6,7,4,1.01828);
    AlphaTracklets23_->SetBinContent(6,7,5,0.959574);
    AlphaTracklets23_->SetBinContent(6,7,6,1.04442);
    AlphaTracklets23_->SetBinContent(6,7,7,1.0427);
    AlphaTracklets23_->SetBinContent(6,7,8,0.997814);
    AlphaTracklets23_->SetBinContent(6,7,9,1.01313);
    AlphaTracklets23_->SetBinContent(6,7,10,1.05772);
    AlphaTracklets23_->SetBinContent(6,8,1,0.957558);
    AlphaTracklets23_->SetBinContent(6,8,2,0.996883);
    AlphaTracklets23_->SetBinContent(6,8,3,1.05222);
    AlphaTracklets23_->SetBinContent(6,8,4,1.00471);
    AlphaTracklets23_->SetBinContent(6,8,5,0.953844);
    AlphaTracklets23_->SetBinContent(6,8,6,1.04381);
    AlphaTracklets23_->SetBinContent(6,8,7,1.05088);
    AlphaTracklets23_->SetBinContent(6,8,8,1.00057);
    AlphaTracklets23_->SetBinContent(6,8,9,1.02437);
    AlphaTracklets23_->SetBinContent(6,8,10,1.063);
    AlphaTracklets23_->SetBinContent(6,9,1,0.985324);
    AlphaTracklets23_->SetBinContent(6,9,2,0.969361);
    AlphaTracklets23_->SetBinContent(6,9,3,1.02279);
    AlphaTracklets23_->SetBinContent(6,9,4,0.991316);
    AlphaTracklets23_->SetBinContent(6,9,5,0.950503);
    AlphaTracklets23_->SetBinContent(6,9,6,1.03299);
    AlphaTracklets23_->SetBinContent(6,9,7,1.03936);
    AlphaTracklets23_->SetBinContent(6,9,8,0.981863);
    AlphaTracklets23_->SetBinContent(6,9,9,0.99771);
    AlphaTracklets23_->SetBinContent(6,9,10,1.01309);
    AlphaTracklets23_->SetBinContent(6,10,1,0.958188);
    AlphaTracklets23_->SetBinContent(6,10,2,0.955747);
    AlphaTracklets23_->SetBinContent(6,10,3,1.01795);
    AlphaTracklets23_->SetBinContent(6,10,4,0.986532);
    AlphaTracklets23_->SetBinContent(6,10,5,0.93472);
    AlphaTracklets23_->SetBinContent(6,10,6,1.0301);
    AlphaTracklets23_->SetBinContent(6,10,7,1.0076);
    AlphaTracklets23_->SetBinContent(6,10,8,0.967766);
    AlphaTracklets23_->SetBinContent(6,10,9,0.978737);
    AlphaTracklets23_->SetBinContent(6,10,10,1.03578);
    AlphaTracklets23_->SetBinContent(6,11,1,0.902519);
    AlphaTracklets23_->SetBinContent(6,11,2,0.943473);
    AlphaTracklets23_->SetBinContent(6,11,3,1.00067);
    AlphaTracklets23_->SetBinContent(6,11,4,0.953671);
    AlphaTracklets23_->SetBinContent(6,11,5,0.918645);
    AlphaTracklets23_->SetBinContent(6,11,6,0.999714);
    AlphaTracklets23_->SetBinContent(6,11,7,0.988108);
    AlphaTracklets23_->SetBinContent(6,11,8,0.933061);
    AlphaTracklets23_->SetBinContent(6,11,9,0.955997);
    AlphaTracklets23_->SetBinContent(6,11,10,0.974166);
    AlphaTracklets23_->SetBinContent(6,12,1,0.849624);
    AlphaTracklets23_->SetBinContent(6,12,2,0.839525);
    AlphaTracklets23_->SetBinContent(6,12,3,0.945101);
    AlphaTracklets23_->SetBinContent(6,12,4,0.937946);
    AlphaTracklets23_->SetBinContent(6,12,5,0.892371);
    AlphaTracklets23_->SetBinContent(6,12,6,0.946417);
    AlphaTracklets23_->SetBinContent(6,12,7,0.947462);
    AlphaTracklets23_->SetBinContent(6,12,8,0.90504);
    AlphaTracklets23_->SetBinContent(6,12,9,0.933442);
    AlphaTracklets23_->SetBinContent(6,12,10,1.03743);
    AlphaTracklets23_->SetBinContent(6,13,1,1.35135);
    AlphaTracklets23_->SetBinContent(6,13,2,0.971698);
    AlphaTracklets23_->SetBinContent(6,13,3,0.987234);
    AlphaTracklets23_->SetBinContent(6,13,4,0.95914);
    AlphaTracklets23_->SetBinContent(6,13,5,0.804408);
    AlphaTracklets23_->SetBinContent(6,13,6,0.960714);
    AlphaTracklets23_->SetBinContent(6,13,7,0.897579);
    AlphaTracklets23_->SetBinContent(6,13,8,0.753731);
    AlphaTracklets23_->SetBinContent(6,13,9,0.745856);
    AlphaTracklets23_->SetBinContent(6,13,10,1.25455);
    AlphaTracklets23_->SetBinContent(6,14,1,0);
    AlphaTracklets23_->SetBinContent(6,14,2,0);
    AlphaTracklets23_->SetBinContent(6,14,3,0);
    AlphaTracklets23_->SetBinContent(6,14,4,0);
    AlphaTracklets23_->SetBinContent(6,14,5,0);
    AlphaTracklets23_->SetBinContent(6,14,6,0);
    AlphaTracklets23_->SetBinContent(6,14,7,0);
    AlphaTracklets23_->SetBinContent(6,14,8,0);
    AlphaTracklets23_->SetBinContent(6,14,9,0);
    AlphaTracklets23_->SetBinContent(6,14,10,0);
    AlphaTracklets23_->SetBinContent(7,1,1,1.13415);
    AlphaTracklets23_->SetBinContent(7,1,2,1.06045);
    AlphaTracklets23_->SetBinContent(7,1,3,1.04545);
    AlphaTracklets23_->SetBinContent(7,1,4,1.17947);
    AlphaTracklets23_->SetBinContent(7,1,5,1.15181);
    AlphaTracklets23_->SetBinContent(7,1,6,1.06228);
    AlphaTracklets23_->SetBinContent(7,1,7,1.17431);
    AlphaTracklets23_->SetBinContent(7,1,8,1.19191);
    AlphaTracklets23_->SetBinContent(7,1,9,1.04888);
    AlphaTracklets23_->SetBinContent(7,1,10,1.08683);
    AlphaTracklets23_->SetBinContent(7,2,1,1.07349);
    AlphaTracklets23_->SetBinContent(7,2,2,1.02679);
    AlphaTracklets23_->SetBinContent(7,2,3,1.03049);
    AlphaTracklets23_->SetBinContent(7,2,4,1.08613);
    AlphaTracklets23_->SetBinContent(7,2,5,1.10469);
    AlphaTracklets23_->SetBinContent(7,2,6,1.01049);
    AlphaTracklets23_->SetBinContent(7,2,7,1.0841);
    AlphaTracklets23_->SetBinContent(7,2,8,1.12319);
    AlphaTracklets23_->SetBinContent(7,2,9,1.02864);
    AlphaTracklets23_->SetBinContent(7,2,10,1.03357);
    AlphaTracklets23_->SetBinContent(7,3,1,1.05322);
    AlphaTracklets23_->SetBinContent(7,3,2,1.03087);
    AlphaTracklets23_->SetBinContent(7,3,3,0.998505);
    AlphaTracklets23_->SetBinContent(7,3,4,1.0879);
    AlphaTracklets23_->SetBinContent(7,3,5,1.08596);
    AlphaTracklets23_->SetBinContent(7,3,6,0.997965);
    AlphaTracklets23_->SetBinContent(7,3,7,1.06613);
    AlphaTracklets23_->SetBinContent(7,3,8,1.11186);
    AlphaTracklets23_->SetBinContent(7,3,9,1.03065);
    AlphaTracklets23_->SetBinContent(7,3,10,1.03672);
    AlphaTracklets23_->SetBinContent(7,4,1,1.03264);
    AlphaTracklets23_->SetBinContent(7,4,2,1.04757);
    AlphaTracklets23_->SetBinContent(7,4,3,0.992544);
    AlphaTracklets23_->SetBinContent(7,4,4,1.06214);
    AlphaTracklets23_->SetBinContent(7,4,5,1.07973);
    AlphaTracklets23_->SetBinContent(7,4,6,0.989021);
    AlphaTracklets23_->SetBinContent(7,4,7,1.04617);
    AlphaTracklets23_->SetBinContent(7,4,8,1.09763);
    AlphaTracklets23_->SetBinContent(7,4,9,1.03598);
    AlphaTracklets23_->SetBinContent(7,4,10,1.02352);
    AlphaTracklets23_->SetBinContent(7,5,1,1.06841);
    AlphaTracklets23_->SetBinContent(7,5,2,1.0188);
    AlphaTracklets23_->SetBinContent(7,5,3,0.972586);
    AlphaTracklets23_->SetBinContent(7,5,4,1.0527);
    AlphaTracklets23_->SetBinContent(7,5,5,1.06595);
    AlphaTracklets23_->SetBinContent(7,5,6,0.980791);
    AlphaTracklets23_->SetBinContent(7,5,7,1.05507);
    AlphaTracklets23_->SetBinContent(7,5,8,1.08196);
    AlphaTracklets23_->SetBinContent(7,5,9,1.01868);
    AlphaTracklets23_->SetBinContent(7,5,10,1.01039);
    AlphaTracklets23_->SetBinContent(7,6,1,1.06121);
    AlphaTracklets23_->SetBinContent(7,6,2,1.01897);
    AlphaTracklets23_->SetBinContent(7,6,3,0.983852);
    AlphaTracklets23_->SetBinContent(7,6,4,1.04063);
    AlphaTracklets23_->SetBinContent(7,6,5,1.06564);
    AlphaTracklets23_->SetBinContent(7,6,6,0.972101);
    AlphaTracklets23_->SetBinContent(7,6,7,1.02408);
    AlphaTracklets23_->SetBinContent(7,6,8,1.06281);
    AlphaTracklets23_->SetBinContent(7,6,9,1.0174);
    AlphaTracklets23_->SetBinContent(7,6,10,1.00376);
    AlphaTracklets23_->SetBinContent(7,7,1,1.06343);
    AlphaTracklets23_->SetBinContent(7,7,2,0.997513);
    AlphaTracklets23_->SetBinContent(7,7,3,0.976437);
    AlphaTracklets23_->SetBinContent(7,7,4,1.02419);
    AlphaTracklets23_->SetBinContent(7,7,5,1.05392);
    AlphaTracklets23_->SetBinContent(7,7,6,0.980181);
    AlphaTracklets23_->SetBinContent(7,7,7,1.03226);
    AlphaTracklets23_->SetBinContent(7,7,8,1.06363);
    AlphaTracklets23_->SetBinContent(7,7,9,1.01859);
    AlphaTracklets23_->SetBinContent(7,7,10,1);
    AlphaTracklets23_->SetBinContent(7,8,1,1.0282);
    AlphaTracklets23_->SetBinContent(7,8,2,0.992917);
    AlphaTracklets23_->SetBinContent(7,8,3,0.963372);
    AlphaTracklets23_->SetBinContent(7,8,4,1.03674);
    AlphaTracklets23_->SetBinContent(7,8,5,1.04881);
    AlphaTracklets23_->SetBinContent(7,8,6,0.968316);
    AlphaTracklets23_->SetBinContent(7,8,7,1.02425);
    AlphaTracklets23_->SetBinContent(7,8,8,1.0667);
    AlphaTracklets23_->SetBinContent(7,8,9,1.00677);
    AlphaTracklets23_->SetBinContent(7,8,10,1.0291);
    AlphaTracklets23_->SetBinContent(7,9,1,1.04355);
    AlphaTracklets23_->SetBinContent(7,9,2,0.984473);
    AlphaTracklets23_->SetBinContent(7,9,3,0.951608);
    AlphaTracklets23_->SetBinContent(7,9,4,1.02244);
    AlphaTracklets23_->SetBinContent(7,9,5,1.04274);
    AlphaTracklets23_->SetBinContent(7,9,6,0.957455);
    AlphaTracklets23_->SetBinContent(7,9,7,1.01653);
    AlphaTracklets23_->SetBinContent(7,9,8,1.0575);
    AlphaTracklets23_->SetBinContent(7,9,9,0.999289);
    AlphaTracklets23_->SetBinContent(7,9,10,0.969013);
    AlphaTracklets23_->SetBinContent(7,10,1,1.03116);
    AlphaTracklets23_->SetBinContent(7,10,2,0.961697);
    AlphaTracklets23_->SetBinContent(7,10,3,0.937408);
    AlphaTracklets23_->SetBinContent(7,10,4,0.995431);
    AlphaTracklets23_->SetBinContent(7,10,5,1.03613);
    AlphaTracklets23_->SetBinContent(7,10,6,0.952762);
    AlphaTracklets23_->SetBinContent(7,10,7,1.0065);
    AlphaTracklets23_->SetBinContent(7,10,8,1.03212);
    AlphaTracklets23_->SetBinContent(7,10,9,0.97096);
    AlphaTracklets23_->SetBinContent(7,10,10,0.981361);
    AlphaTracklets23_->SetBinContent(7,11,1,0.982538);
    AlphaTracklets23_->SetBinContent(7,11,2,0.933681);
    AlphaTracklets23_->SetBinContent(7,11,3,0.947947);
    AlphaTracklets23_->SetBinContent(7,11,4,0.982873);
    AlphaTracklets23_->SetBinContent(7,11,5,1.01769);
    AlphaTracklets23_->SetBinContent(7,11,6,0.942783);
    AlphaTracklets23_->SetBinContent(7,11,7,0.976579);
    AlphaTracklets23_->SetBinContent(7,11,8,1.00596);
    AlphaTracklets23_->SetBinContent(7,11,9,0.96826);
    AlphaTracklets23_->SetBinContent(7,11,10,0.983246);
    AlphaTracklets23_->SetBinContent(7,12,1,1.07874);
    AlphaTracklets23_->SetBinContent(7,12,2,0.897727);
    AlphaTracklets23_->SetBinContent(7,12,3,0.898093);
    AlphaTracklets23_->SetBinContent(7,12,4,0.965778);
    AlphaTracklets23_->SetBinContent(7,12,5,0.934842);
    AlphaTracklets23_->SetBinContent(7,12,6,0.877445);
    AlphaTracklets23_->SetBinContent(7,12,7,0.947559);
    AlphaTracklets23_->SetBinContent(7,12,8,1.03818);
    AlphaTracklets23_->SetBinContent(7,12,9,1.00876);
    AlphaTracklets23_->SetBinContent(7,12,10,0.907609);
    AlphaTracklets23_->SetBinContent(7,13,1,0.913043);
    AlphaTracklets23_->SetBinContent(7,13,2,0.865801);
    AlphaTracklets23_->SetBinContent(7,13,3,0.802083);
    AlphaTracklets23_->SetBinContent(7,13,4,0.90989);
    AlphaTracklets23_->SetBinContent(7,13,5,0.972171);
    AlphaTracklets23_->SetBinContent(7,13,6,0.839233);
    AlphaTracklets23_->SetBinContent(7,13,7,0.907298);
    AlphaTracklets23_->SetBinContent(7,13,8,0.792);
    AlphaTracklets23_->SetBinContent(7,13,9,0.922535);
    AlphaTracklets23_->SetBinContent(7,13,10,1.02703);
    AlphaTracklets23_->SetBinContent(7,14,1,0);
    AlphaTracklets23_->SetBinContent(7,14,2,0);
    AlphaTracklets23_->SetBinContent(7,14,3,0);
    AlphaTracklets23_->SetBinContent(7,14,4,0);
    AlphaTracklets23_->SetBinContent(7,14,5,0);
    AlphaTracklets23_->SetBinContent(7,14,6,0);
    AlphaTracklets23_->SetBinContent(7,14,7,0);
    AlphaTracklets23_->SetBinContent(7,14,8,0);
    AlphaTracklets23_->SetBinContent(7,14,9,0);
    AlphaTracklets23_->SetBinContent(7,14,10,0);
    AlphaTracklets23_->SetBinContent(8,1,1,1.11053);
    AlphaTracklets23_->SetBinContent(8,1,2,1.09345);
    AlphaTracklets23_->SetBinContent(8,1,3,1.14014);
    AlphaTracklets23_->SetBinContent(8,1,4,1.08744);
    AlphaTracklets23_->SetBinContent(8,1,5,1.13989);
    AlphaTracklets23_->SetBinContent(8,1,6,1.12118);
    AlphaTracklets23_->SetBinContent(8,1,7,1.1049);
    AlphaTracklets23_->SetBinContent(8,1,8,1.0878);
    AlphaTracklets23_->SetBinContent(8,1,9,1.11072);
    AlphaTracklets23_->SetBinContent(8,1,10,1.09611);
    AlphaTracklets23_->SetBinContent(8,2,1,1.06245);
    AlphaTracklets23_->SetBinContent(8,2,2,1.07566);
    AlphaTracklets23_->SetBinContent(8,2,3,1.07833);
    AlphaTracklets23_->SetBinContent(8,2,4,1.04518);
    AlphaTracklets23_->SetBinContent(8,2,5,1.08441);
    AlphaTracklets23_->SetBinContent(8,2,6,1.09067);
    AlphaTracklets23_->SetBinContent(8,2,7,1.05759);
    AlphaTracklets23_->SetBinContent(8,2,8,1.03343);
    AlphaTracklets23_->SetBinContent(8,2,9,1.09141);
    AlphaTracklets23_->SetBinContent(8,2,10,1.10711);
    AlphaTracklets23_->SetBinContent(8,3,1,1.02721);
    AlphaTracklets23_->SetBinContent(8,3,2,1.05781);
    AlphaTracklets23_->SetBinContent(8,3,3,1.07387);
    AlphaTracklets23_->SetBinContent(8,3,4,1.04367);
    AlphaTracklets23_->SetBinContent(8,3,5,1.07033);
    AlphaTracklets23_->SetBinContent(8,3,6,1.08919);
    AlphaTracklets23_->SetBinContent(8,3,7,1.05414);
    AlphaTracklets23_->SetBinContent(8,3,8,1.03528);
    AlphaTracklets23_->SetBinContent(8,3,9,1.07813);
    AlphaTracklets23_->SetBinContent(8,3,10,1.07778);
    AlphaTracklets23_->SetBinContent(8,4,1,1.03981);
    AlphaTracklets23_->SetBinContent(8,4,2,1.0549);
    AlphaTracklets23_->SetBinContent(8,4,3,1.06551);
    AlphaTracklets23_->SetBinContent(8,4,4,1.04113);
    AlphaTracklets23_->SetBinContent(8,4,5,1.07094);
    AlphaTracklets23_->SetBinContent(8,4,6,1.08373);
    AlphaTracklets23_->SetBinContent(8,4,7,1.04396);
    AlphaTracklets23_->SetBinContent(8,4,8,1.04561);
    AlphaTracklets23_->SetBinContent(8,4,9,1.06875);
    AlphaTracklets23_->SetBinContent(8,4,10,1.06513);
    AlphaTracklets23_->SetBinContent(8,5,1,1.02975);
    AlphaTracklets23_->SetBinContent(8,5,2,1.04784);
    AlphaTracklets23_->SetBinContent(8,5,3,1.07631);
    AlphaTracklets23_->SetBinContent(8,5,4,1.03921);
    AlphaTracklets23_->SetBinContent(8,5,5,1.06648);
    AlphaTracklets23_->SetBinContent(8,5,6,1.07781);
    AlphaTracklets23_->SetBinContent(8,5,7,1.04656);
    AlphaTracklets23_->SetBinContent(8,5,8,1.04198);
    AlphaTracklets23_->SetBinContent(8,5,9,1.05783);
    AlphaTracklets23_->SetBinContent(8,5,10,1.043);
    AlphaTracklets23_->SetBinContent(8,6,1,1.03566);
    AlphaTracklets23_->SetBinContent(8,6,2,1.05173);
    AlphaTracklets23_->SetBinContent(8,6,3,1.07994);
    AlphaTracklets23_->SetBinContent(8,6,4,1.03992);
    AlphaTracklets23_->SetBinContent(8,6,5,1.07805);
    AlphaTracklets23_->SetBinContent(8,6,6,1.08172);
    AlphaTracklets23_->SetBinContent(8,6,7,1.04866);
    AlphaTracklets23_->SetBinContent(8,6,8,1.05721);
    AlphaTracklets23_->SetBinContent(8,6,9,1.05577);
    AlphaTracklets23_->SetBinContent(8,6,10,1.08449);
    AlphaTracklets23_->SetBinContent(8,7,1,0.997317);
    AlphaTracklets23_->SetBinContent(8,7,2,1.06557);
    AlphaTracklets23_->SetBinContent(8,7,3,1.0713);
    AlphaTracklets23_->SetBinContent(8,7,4,1.03845);
    AlphaTracklets23_->SetBinContent(8,7,5,1.07346);
    AlphaTracklets23_->SetBinContent(8,7,6,1.08731);
    AlphaTracklets23_->SetBinContent(8,7,7,1.04524);
    AlphaTracklets23_->SetBinContent(8,7,8,1.03714);
    AlphaTracklets23_->SetBinContent(8,7,9,1.07793);
    AlphaTracklets23_->SetBinContent(8,7,10,1.07671);
    AlphaTracklets23_->SetBinContent(8,8,1,1.04988);
    AlphaTracklets23_->SetBinContent(8,8,2,1.08845);
    AlphaTracklets23_->SetBinContent(8,8,3,1.08331);
    AlphaTracklets23_->SetBinContent(8,8,4,1.01773);
    AlphaTracklets23_->SetBinContent(8,8,5,1.08263);
    AlphaTracklets23_->SetBinContent(8,8,6,1.0787);
    AlphaTracklets23_->SetBinContent(8,8,7,1.05765);
    AlphaTracklets23_->SetBinContent(8,8,8,1.04193);
    AlphaTracklets23_->SetBinContent(8,8,9,1.04586);
    AlphaTracklets23_->SetBinContent(8,8,10,1.04375);
    AlphaTracklets23_->SetBinContent(8,9,1,1.03929);
    AlphaTracklets23_->SetBinContent(8,9,2,1.06553);
    AlphaTracklets23_->SetBinContent(8,9,3,1.05662);
    AlphaTracklets23_->SetBinContent(8,9,4,1.03631);
    AlphaTracklets23_->SetBinContent(8,9,5,1.08167);
    AlphaTracklets23_->SetBinContent(8,9,6,1.0931);
    AlphaTracklets23_->SetBinContent(8,9,7,1.04744);
    AlphaTracklets23_->SetBinContent(8,9,8,1.04854);
    AlphaTracklets23_->SetBinContent(8,9,9,1.09039);
    AlphaTracklets23_->SetBinContent(8,9,10,1.06678);
    AlphaTracklets23_->SetBinContent(8,10,1,1.0317);
    AlphaTracklets23_->SetBinContent(8,10,2,1.06491);
    AlphaTracklets23_->SetBinContent(8,10,3,1.08594);
    AlphaTracklets23_->SetBinContent(8,10,4,1.03803);
    AlphaTracklets23_->SetBinContent(8,10,5,1.0588);
    AlphaTracklets23_->SetBinContent(8,10,6,1.08632);
    AlphaTracklets23_->SetBinContent(8,10,7,1.06475);
    AlphaTracklets23_->SetBinContent(8,10,8,1.03379);
    AlphaTracklets23_->SetBinContent(8,10,9,1.10892);
    AlphaTracklets23_->SetBinContent(8,10,10,1.0484);
    AlphaTracklets23_->SetBinContent(8,11,1,1.01183);
    AlphaTracklets23_->SetBinContent(8,11,2,1.04401);
    AlphaTracklets23_->SetBinContent(8,11,3,1.07333);
    AlphaTracklets23_->SetBinContent(8,11,4,1.04597);
    AlphaTracklets23_->SetBinContent(8,11,5,1.07236);
    AlphaTracklets23_->SetBinContent(8,11,6,1.08729);
    AlphaTracklets23_->SetBinContent(8,11,7,1.04684);
    AlphaTracklets23_->SetBinContent(8,11,8,1.04652);
    AlphaTracklets23_->SetBinContent(8,11,9,1.07335);
    AlphaTracklets23_->SetBinContent(8,11,10,1.04774);
    AlphaTracklets23_->SetBinContent(8,12,1,1.1097);
    AlphaTracklets23_->SetBinContent(8,12,2,1.00868);
    AlphaTracklets23_->SetBinContent(8,12,3,1.09869);
    AlphaTracklets23_->SetBinContent(8,12,4,1.06316);
    AlphaTracklets23_->SetBinContent(8,12,5,1.08347);
    AlphaTracklets23_->SetBinContent(8,12,6,1.10182);
    AlphaTracklets23_->SetBinContent(8,12,7,1.03371);
    AlphaTracklets23_->SetBinContent(8,12,8,1.12076);
    AlphaTracklets23_->SetBinContent(8,12,9,1.06193);
    AlphaTracklets23_->SetBinContent(8,12,10,0.993902);
    AlphaTracklets23_->SetBinContent(8,13,1,1.51515);
    AlphaTracklets23_->SetBinContent(8,13,2,0.968254);
    AlphaTracklets23_->SetBinContent(8,13,3,1.15385);
    AlphaTracklets23_->SetBinContent(8,13,4,1.14758);
    AlphaTracklets23_->SetBinContent(8,13,5,0.992439);
    AlphaTracklets23_->SetBinContent(8,13,6,1.08263);
    AlphaTracklets23_->SetBinContent(8,13,7,1.12251);
    AlphaTracklets23_->SetBinContent(8,13,8,0.946939);
    AlphaTracklets23_->SetBinContent(8,13,9,1.1413);
    AlphaTracklets23_->SetBinContent(8,13,10,0.953125);
    AlphaTracklets23_->SetBinContent(8,14,1,0);
    AlphaTracklets23_->SetBinContent(8,14,2,0);
    AlphaTracklets23_->SetBinContent(8,14,3,0);
    AlphaTracklets23_->SetBinContent(8,14,4,0);
    AlphaTracklets23_->SetBinContent(8,14,5,0);
    AlphaTracklets23_->SetBinContent(8,14,6,0);
    AlphaTracklets23_->SetBinContent(8,14,7,0);
    AlphaTracklets23_->SetBinContent(8,14,8,0);
    AlphaTracklets23_->SetBinContent(8,14,9,0);
    AlphaTracklets23_->SetBinContent(8,14,10,0);
    AlphaTracklets23_->SetBinContent(9,1,1,1.10459);
    AlphaTracklets23_->SetBinContent(9,1,2,1.08201);
    AlphaTracklets23_->SetBinContent(9,1,3,1.08711);
    AlphaTracklets23_->SetBinContent(9,1,4,1.08685);
    AlphaTracklets23_->SetBinContent(9,1,5,1.06564);
    AlphaTracklets23_->SetBinContent(9,1,6,1.07428);
    AlphaTracklets23_->SetBinContent(9,1,7,1.05868);
    AlphaTracklets23_->SetBinContent(9,1,8,1.09019);
    AlphaTracklets23_->SetBinContent(9,1,9,1.3088);
    AlphaTracklets23_->SetBinContent(9,1,10,1.5538);
    AlphaTracklets23_->SetBinContent(9,2,1,1.07561);
    AlphaTracklets23_->SetBinContent(9,2,2,1.08378);
    AlphaTracklets23_->SetBinContent(9,2,3,1.06025);
    AlphaTracklets23_->SetBinContent(9,2,4,1.05414);
    AlphaTracklets23_->SetBinContent(9,2,5,1.04947);
    AlphaTracklets23_->SetBinContent(9,2,6,1.03919);
    AlphaTracklets23_->SetBinContent(9,2,7,1.04822);
    AlphaTracklets23_->SetBinContent(9,2,8,1.08045);
    AlphaTracklets23_->SetBinContent(9,2,9,1.27209);
    AlphaTracklets23_->SetBinContent(9,2,10,1.66087);
    AlphaTracklets23_->SetBinContent(9,3,1,1.0942);
    AlphaTracklets23_->SetBinContent(9,3,2,1.06329);
    AlphaTracklets23_->SetBinContent(9,3,3,1.05392);
    AlphaTracklets23_->SetBinContent(9,3,4,1.0491);
    AlphaTracklets23_->SetBinContent(9,3,5,1.04659);
    AlphaTracklets23_->SetBinContent(9,3,6,1.05053);
    AlphaTracklets23_->SetBinContent(9,3,7,1.04021);
    AlphaTracklets23_->SetBinContent(9,3,8,1.07366);
    AlphaTracklets23_->SetBinContent(9,3,9,1.25207);
    AlphaTracklets23_->SetBinContent(9,3,10,1.65933);
    AlphaTracklets23_->SetBinContent(9,4,1,1.0689);
    AlphaTracklets23_->SetBinContent(9,4,2,1.05071);
    AlphaTracklets23_->SetBinContent(9,4,3,1.06843);
    AlphaTracklets23_->SetBinContent(9,4,4,1.05256);
    AlphaTracklets23_->SetBinContent(9,4,5,1.04442);
    AlphaTracklets23_->SetBinContent(9,4,6,1.05079);
    AlphaTracklets23_->SetBinContent(9,4,7,1.04786);
    AlphaTracklets23_->SetBinContent(9,4,8,1.07731);
    AlphaTracklets23_->SetBinContent(9,4,9,1.25222);
    AlphaTracklets23_->SetBinContent(9,4,10,1.63253);
    AlphaTracklets23_->SetBinContent(9,5,1,1.11958);
    AlphaTracklets23_->SetBinContent(9,5,2,1.08218);
    AlphaTracklets23_->SetBinContent(9,5,3,1.08567);
    AlphaTracklets23_->SetBinContent(9,5,4,1.05492);
    AlphaTracklets23_->SetBinContent(9,5,5,1.05433);
    AlphaTracklets23_->SetBinContent(9,5,6,1.05297);
    AlphaTracklets23_->SetBinContent(9,5,7,1.05631);
    AlphaTracklets23_->SetBinContent(9,5,8,1.09158);
    AlphaTracklets23_->SetBinContent(9,5,9,1.25436);
    AlphaTracklets23_->SetBinContent(9,5,10,1.63394);
    AlphaTracklets23_->SetBinContent(9,6,1,1.10027);
    AlphaTracklets23_->SetBinContent(9,6,2,1.10219);
    AlphaTracklets23_->SetBinContent(9,6,3,1.0724);
    AlphaTracklets23_->SetBinContent(9,6,4,1.05975);
    AlphaTracklets23_->SetBinContent(9,6,5,1.06624);
    AlphaTracklets23_->SetBinContent(9,6,6,1.06157);
    AlphaTracklets23_->SetBinContent(9,6,7,1.05932);
    AlphaTracklets23_->SetBinContent(9,6,8,1.08435);
    AlphaTracklets23_->SetBinContent(9,6,9,1.25721);
    AlphaTracklets23_->SetBinContent(9,6,10,1.56176);
    AlphaTracklets23_->SetBinContent(9,7,1,1.12346);
    AlphaTracklets23_->SetBinContent(9,7,2,1.078);
    AlphaTracklets23_->SetBinContent(9,7,3,1.06965);
    AlphaTracklets23_->SetBinContent(9,7,4,1.07981);
    AlphaTracklets23_->SetBinContent(9,7,5,1.06496);
    AlphaTracklets23_->SetBinContent(9,7,6,1.07841);
    AlphaTracklets23_->SetBinContent(9,7,7,1.05604);
    AlphaTracklets23_->SetBinContent(9,7,8,1.09847);
    AlphaTracklets23_->SetBinContent(9,7,9,1.29382);
    AlphaTracklets23_->SetBinContent(9,7,10,1.66565);
    AlphaTracklets23_->SetBinContent(9,8,1,1.14415);
    AlphaTracklets23_->SetBinContent(9,8,2,1.09215);
    AlphaTracklets23_->SetBinContent(9,8,3,1.06401);
    AlphaTracklets23_->SetBinContent(9,8,4,1.07372);
    AlphaTracklets23_->SetBinContent(9,8,5,1.06017);
    AlphaTracklets23_->SetBinContent(9,8,6,1.06595);
    AlphaTracklets23_->SetBinContent(9,8,7,1.08356);
    AlphaTracklets23_->SetBinContent(9,8,8,1.09783);
    AlphaTracklets23_->SetBinContent(9,8,9,1.32129);
    AlphaTracklets23_->SetBinContent(9,8,10,1.6);
    AlphaTracklets23_->SetBinContent(9,9,1,1.11333);
    AlphaTracklets23_->SetBinContent(9,9,2,1.12701);
    AlphaTracklets23_->SetBinContent(9,9,3,1.08394);
    AlphaTracklets23_->SetBinContent(9,9,4,1.08193);
    AlphaTracklets23_->SetBinContent(9,9,5,1.07442);
    AlphaTracklets23_->SetBinContent(9,9,6,1.08508);
    AlphaTracklets23_->SetBinContent(9,9,7,1.07219);
    AlphaTracklets23_->SetBinContent(9,9,8,1.12935);
    AlphaTracklets23_->SetBinContent(9,9,9,1.29767);
    AlphaTracklets23_->SetBinContent(9,9,10,1.61313);
    AlphaTracklets23_->SetBinContent(9,10,1,1.13921);
    AlphaTracklets23_->SetBinContent(9,10,2,1.10603);
    AlphaTracklets23_->SetBinContent(9,10,3,1.09715);
    AlphaTracklets23_->SetBinContent(9,10,4,1.07944);
    AlphaTracklets23_->SetBinContent(9,10,5,1.08802);
    AlphaTracklets23_->SetBinContent(9,10,6,1.09525);
    AlphaTracklets23_->SetBinContent(9,10,7,1.0824);
    AlphaTracklets23_->SetBinContent(9,10,8,1.12829);
    AlphaTracklets23_->SetBinContent(9,10,9,1.30355);
    AlphaTracklets23_->SetBinContent(9,10,10,1.63511);
    AlphaTracklets23_->SetBinContent(9,11,1,1.13072);
    AlphaTracklets23_->SetBinContent(9,11,2,1.0782);
    AlphaTracklets23_->SetBinContent(9,11,3,1.09676);
    AlphaTracklets23_->SetBinContent(9,11,4,1.10631);
    AlphaTracklets23_->SetBinContent(9,11,5,1.0811);
    AlphaTracklets23_->SetBinContent(9,11,6,1.12452);
    AlphaTracklets23_->SetBinContent(9,11,7,1.09055);
    AlphaTracklets23_->SetBinContent(9,11,8,1.1282);
    AlphaTracklets23_->SetBinContent(9,11,9,1.28359);
    AlphaTracklets23_->SetBinContent(9,11,10,1.53646);
    AlphaTracklets23_->SetBinContent(9,12,1,1.18848);
    AlphaTracklets23_->SetBinContent(9,12,2,1.17336);
    AlphaTracklets23_->SetBinContent(9,12,3,1.10313);
    AlphaTracklets23_->SetBinContent(9,12,4,1.06951);
    AlphaTracklets23_->SetBinContent(9,12,5,1.12765);
    AlphaTracklets23_->SetBinContent(9,12,6,1.12747);
    AlphaTracklets23_->SetBinContent(9,12,7,1.11054);
    AlphaTracklets23_->SetBinContent(9,12,8,1.1011);
    AlphaTracklets23_->SetBinContent(9,12,9,1.25059);
    AlphaTracklets23_->SetBinContent(9,12,10,1.76087);
    AlphaTracklets23_->SetBinContent(9,13,1,1.15909);
    AlphaTracklets23_->SetBinContent(9,13,2,1.07143);
    AlphaTracklets23_->SetBinContent(9,13,3,1.2268);
    AlphaTracklets23_->SetBinContent(9,13,4,1.21194);
    AlphaTracklets23_->SetBinContent(9,13,5,1.10596);
    AlphaTracklets23_->SetBinContent(9,13,6,1.09402);
    AlphaTracklets23_->SetBinContent(9,13,7,1.09896);
    AlphaTracklets23_->SetBinContent(9,13,8,1.24157);
    AlphaTracklets23_->SetBinContent(9,13,9,1.26168);
    AlphaTracklets23_->SetBinContent(9,13,10,1.34146);
    AlphaTracklets23_->SetBinContent(9,14,1,0);
    AlphaTracklets23_->SetBinContent(9,14,2,0);
    AlphaTracklets23_->SetBinContent(9,14,3,0);
    AlphaTracklets23_->SetBinContent(9,14,4,0);
    AlphaTracklets23_->SetBinContent(9,14,5,0);
    AlphaTracklets23_->SetBinContent(9,14,6,0);
    AlphaTracklets23_->SetBinContent(9,14,7,0);
    AlphaTracklets23_->SetBinContent(9,14,8,0);
    AlphaTracklets23_->SetBinContent(9,14,9,0);
    AlphaTracklets23_->SetBinContent(9,14,10,0);
    AlphaTracklets23_->SetBinContent(10,1,1,1.15939);
    AlphaTracklets23_->SetBinContent(10,1,2,1.30515);
    AlphaTracklets23_->SetBinContent(10,1,3,1.53527);
    AlphaTracklets23_->SetBinContent(10,1,4,1.88039);
    AlphaTracklets23_->SetBinContent(10,1,5,2.4574);
    AlphaTracklets23_->SetBinContent(10,1,6,3.58944);
    AlphaTracklets23_->SetBinContent(10,2,1,1.19107);
    AlphaTracklets23_->SetBinContent(10,2,2,1.24384);
    AlphaTracklets23_->SetBinContent(10,2,3,1.45303);
    AlphaTracklets23_->SetBinContent(10,2,4,1.80667);
    AlphaTracklets23_->SetBinContent(10,2,5,2.32301);
    AlphaTracklets23_->SetBinContent(10,2,6,3.29887);
    AlphaTracklets23_->SetBinContent(10,3,1,1.12);
    AlphaTracklets23_->SetBinContent(10,3,2,1.24448);
    AlphaTracklets23_->SetBinContent(10,3,3,1.46877);
    AlphaTracklets23_->SetBinContent(10,3,4,1.80857);
    AlphaTracklets23_->SetBinContent(10,3,5,2.28703);
    AlphaTracklets23_->SetBinContent(10,3,6,3.27329);
    AlphaTracklets23_->SetBinContent(10,4,1,1.15253);
    AlphaTracklets23_->SetBinContent(10,4,2,1.25123);
    AlphaTracklets23_->SetBinContent(10,4,3,1.47782);
    AlphaTracklets23_->SetBinContent(10,4,4,1.82943);
    AlphaTracklets23_->SetBinContent(10,4,5,2.29368);
    AlphaTracklets23_->SetBinContent(10,4,6,3.30806);
    AlphaTracklets23_->SetBinContent(10,5,1,1.15616);
    AlphaTracklets23_->SetBinContent(10,5,2,1.24261);
    AlphaTracklets23_->SetBinContent(10,5,3,1.49619);
    AlphaTracklets23_->SetBinContent(10,5,4,1.8355);
    AlphaTracklets23_->SetBinContent(10,5,5,2.303);
    AlphaTracklets23_->SetBinContent(10,5,6,3.28833);
    AlphaTracklets23_->SetBinContent(10,6,1,1.15028);
    AlphaTracklets23_->SetBinContent(10,6,2,1.29768);
    AlphaTracklets23_->SetBinContent(10,6,3,1.52033);
    AlphaTracklets23_->SetBinContent(10,6,4,1.82623);
    AlphaTracklets23_->SetBinContent(10,6,5,2.32335);
    AlphaTracklets23_->SetBinContent(10,6,6,3.28328);
    AlphaTracklets23_->SetBinContent(10,7,1,1.12117);
    AlphaTracklets23_->SetBinContent(10,7,2,1.28376);
    AlphaTracklets23_->SetBinContent(10,7,3,1.51336);
    AlphaTracklets23_->SetBinContent(10,7,4,1.86121);
    AlphaTracklets23_->SetBinContent(10,7,5,2.37364);
    AlphaTracklets23_->SetBinContent(10,7,6,3.34052);
    AlphaTracklets23_->SetBinContent(10,8,1,1.14573);
    AlphaTracklets23_->SetBinContent(10,8,2,1.27265);
    AlphaTracklets23_->SetBinContent(10,8,3,1.4854);
    AlphaTracklets23_->SetBinContent(10,8,4,1.876);
    AlphaTracklets23_->SetBinContent(10,8,5,2.35308);
    AlphaTracklets23_->SetBinContent(10,8,6,3.21314);
    AlphaTracklets23_->SetBinContent(10,9,1,1.13791);
    AlphaTracklets23_->SetBinContent(10,9,2,1.30473);
    AlphaTracklets23_->SetBinContent(10,9,3,1.52305);
    AlphaTracklets23_->SetBinContent(10,9,4,1.86833);
    AlphaTracklets23_->SetBinContent(10,9,5,2.33449);
    AlphaTracklets23_->SetBinContent(10,9,6,3.29282);
    AlphaTracklets23_->SetBinContent(10,10,1,1.23143);
    AlphaTracklets23_->SetBinContent(10,10,2,1.26319);
    AlphaTracklets23_->SetBinContent(10,10,3,1.52489);
    AlphaTracklets23_->SetBinContent(10,10,4,1.9115);
    AlphaTracklets23_->SetBinContent(10,10,5,2.4267);
    AlphaTracklets23_->SetBinContent(10,10,6,3.42027);
    AlphaTracklets23_->SetBinContent(10,11,1,1.29009);
    AlphaTracklets23_->SetBinContent(10,11,2,1.36554);
    AlphaTracklets23_->SetBinContent(10,11,3,1.62809);
    AlphaTracklets23_->SetBinContent(10,11,4,1.90269);
    AlphaTracklets23_->SetBinContent(10,11,5,2.39841);
    AlphaTracklets23_->SetBinContent(10,11,6,3.46853);
    AlphaTracklets23_->SetBinContent(10,12,1,1.67376);
    AlphaTracklets23_->SetBinContent(10,12,2,1.40967);
    AlphaTracklets23_->SetBinContent(10,12,3,1.54868);
    AlphaTracklets23_->SetBinContent(10,12,4,2.13179);
    AlphaTracklets23_->SetBinContent(10,12,5,2.38468);
    AlphaTracklets23_->SetBinContent(10,12,6,3.40736);
    AlphaTracklets23_->SetBinContent(10,13,1,1.13333);
    AlphaTracklets23_->SetBinContent(10,13,2,1.2973);
    AlphaTracklets23_->SetBinContent(10,13,3,1.43605);
    AlphaTracklets23_->SetBinContent(10,13,4,1.59592);
    AlphaTracklets23_->SetBinContent(10,13,5,2.27602);
    AlphaTracklets23_->SetBinContent(10,13,6,3.17419);
    AlphaTracklets23_->SetBinContent(10,14,1,0);
    AlphaTracklets23_->SetBinContent(10,14,2,0);
    AlphaTracklets23_->SetBinContent(10,14,3,0);
    AlphaTracklets23_->SetBinContent(10,14,4,0);
    AlphaTracklets23_->SetBinContent(10,14,5,0);
    AlphaTracklets23_->SetBinContent(10,14,6,0);
  }
}
