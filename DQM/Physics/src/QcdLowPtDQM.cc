// $Id: QcdLowPtDQM.cc,v 1.1 2009/11/06 16:28:16 loizides Exp $

#include "DQM/Physics/src/QcdLowPtDQM.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include <TString.h>
#include <TMath.h>
#include <TH3F.h>

using namespace std;
using namespace edm;
//using namespace reco;

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
  hltProcName_(parameters.getUntrackedParameter<string>("hltProcName","HLT")),
  pixelName_(parameters.getUntrackedParameter<string>("pixelRecHits","siPixelRecHits")),
  ZVCut_(parameters.getUntrackedParameter<double>("ZVertexCut",10)),
  dPhiVc_(parameters.getUntrackedParameter<double>("dPhiVertexCut",0.08)),
  dZVc_(parameters.getUntrackedParameter<double>("dZVertexCut",0.25)),
  verbose_(parameters.getUntrackedParameter<int>("verbose",2)),
  pixLayers_(parameters.getUntrackedParameter<int>("pixLayerCombinations",12)),
  usePixelQ_(parameters.getUntrackedParameter<int>("usePixelQualityWord",0)),
  AlphaTracklets12_(0),
  AlphaTracklets13_(0),
  AlphaTracklets23_(0),
  tgeo_(0),
  theDbe_(0),
  h2TrigCorr_(0)
{
  // Constructor.

  if (parameters.exists("hltTrgNames"))
    hltTrgNames_ = parameters.getUntrackedParameter<vector<string> >("hltTrgNames");
  hltTrgNames_.insert(hltTrgNames_.begin(),"Any");

  if (hltResName_.find(':')==string::npos)
    hltResName_ += "::";
  else 
    hltResName_ += ":";
  hltResName_ += hltProcName_;

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

  std::for_each(NrawTracklets12_.begin(),  NrawTracklets12_.end(),  deleter());
  std::for_each(NsigTracklets12_.begin(),  NsigTracklets12_.end(),  deleter());
  std::for_each(NbkgTracklets12_.begin(),  NbkgTracklets12_.end(),  deleter());
  deleter()(AlphaTracklets12_);
  std::for_each(NrawTracklets13_.begin(),  NrawTracklets13_.end(),  deleter());
  std::for_each(NsigTracklets13_.begin(),  NsigTracklets13_.end(),  deleter());
  std::for_each(NbkgTracklets13_.begin(),  NbkgTracklets13_.end(),  deleter());
  deleter()(AlphaTracklets13_);
  std::for_each(NrawTracklets23_.begin(),  NrawTracklets23_.end(),  deleter());
  std::for_each(NsigTracklets23_.begin(),  NsigTracklets23_.end(),  deleter());
  std::for_each(NbkgTracklets23_.begin(),  NbkgTracklets23_.end(),  deleter());
  deleter()(AlphaTracklets23_);
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::analyze(const Event &iEvent, const EventSetup &iSetup) 
{
  // Analyze the given event.

  fillHltBits(iEvent);
  fillPixels(iEvent);
  trackletVertexUnbinned(iEvent, pixLayers_);
  fillTracklets(iEvent, pixLayers_);
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::beginJob() 
{
  // Begin job and setup the DQM store.

  theDbe_ = Service<DQMStore>().operator->();
  if (!theDbe_)
    print(3,"Could not obtain pointer to DQMStore");
  theDbe_->setCurrentFolder("Physics/QcdLowPt");
  yieldAlphaHistogram();
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::beginLuminosityBlock(const LuminosityBlock &l, 
                                       const EventSetup &iSetup)
{
  // At the moment, nothing needed to be done.
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::beginRun(const edm::Run &, const edm::EventSetup &iSetup)
{
  // Begin run, get or create needed structures.

  // get tracker geometry
  edm::ESHandle<TrackerGeometry> trackerHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  tgeo_ = trackerHandle.product();
  if (!tgeo_)
    print(3,"Could not obtain pointer to TrackerGeometry");

  // get trigger bits
  HLTConfigProvider hltConfig;
  if (!hltConfig.init(hltProcName_))
    print(3,Form("Could not obtain HLT config for process name %s", hltProcName_.c_str()));

  // setup "Any" bit
  hltTrgBits_.clear();
  hltTrgBits_.push_back(0);
  hltTrgDeci_.clear();
  hltTrgDeci_.push_back(true);

  // figure out relation of trigger name to trigger bit
  for(size_t i=1;i<hltTrgNames_.size();++i) {
    const string &n1(hltTrgNames_.at(i));
    bool found = 0;
    for(size_t j=0;j<hltConfig.size();++j) {
      const string &n2(hltConfig.triggerName(j));
      if(0) print(0,Form("Checking trigger name %s for %s", n2.c_str(), n1.c_str()));
      if (n2==n1) {
        hltTrgBits_.push_back(j);
        found = 1;
        break;
      }
    }      
    if (!found) {
      CP(2) print(2, Form("Could not find trigger bit for %s", n1.c_str()));
      hltTrgBits_.push_back(-1);
    }
    hltTrgDeci_.push_back(false);
  }

  // ensure that trigger collections are of same size
  if (hltTrgBits_.size()!=hltTrgNames_.size())
    print(3,Form("Size of trigger bits not equal names: %d %d",
                 hltTrgBits_.size(), hltTrgNames_.size()));
  if (hltTrgDeci_.size()!=hltTrgNames_.size())
    print(3,Form("Size of decision bits not equal names: %d %d",
                 hltTrgDeci_.size(), hltTrgNames_.size()));

  // setup correction histograms
  if (AlphaTracklets12_) {
    for(size_t i=0;i<hltTrgNames_.size();++i) {
      TH3F *h1 = (TH3F*)AlphaTracklets12_->Clone(Form("NrawTracklets12-%s",
                                                      hltTrgNames_.at(i).c_str()));
      NrawTracklets12_.push_back(h1);
      TH3F *h2 = (TH3F*)AlphaTracklets12_->Clone(Form("NsigTracklets12-%s",
                                                      hltTrgNames_.at(i).c_str()));
      NsigTracklets12_.push_back(h2);
      TH3F *h3 = (TH3F*)AlphaTracklets12_->Clone(Form("NbkgTracklets12-%s",
                                                      hltTrgNames_.at(i).c_str()));
      NbkgTracklets12_.push_back(h3);
    }
  }

  // book monitoring histograms
  bookHistos();
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::book1D(std::vector<MonitorElement*> &mes, 
                         const std::string &name, const std::string &title, 
                         int nx, double x1, double x2, bool sumw2, bool sbox)
{
  // Book 1D histos.

  for(size_t i=0;i<hltTrgNames_.size();++i) {
    MonitorElement *e = theDbe_->book1D(Form("%s_%s",name.c_str(),hltTrgNames_.at(i).c_str()),
                                        Form("%s: %s",hltTrgNames_.at(i).c_str(), title.c_str()), 
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

  for(size_t i=0;i<hltTrgNames_.size();++i) {
    MonitorElement *e = theDbe_->book2D(Form("%s_%s",name.c_str(),hltTrgNames_.at(i).c_str()),
                                        Form("%s: %s",hltTrgNames_.at(i).c_str(), title.c_str()), 
                                        nx, x1, x2, ny, y1, y2);
    TH1 *h1 = e->getTH1();
    if (sumw2)
      h1->Sumw2();
    h1->SetStats(sbox);
    mes.push_back(e);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::bookHistos()
{
  // Book histograms if needed.

  if (hNhitsL1_.size())
    return; // histograms already booked

  if (1) {
    const int Nx = 30;
    const double x1 = 0;
    const double x2 = 150;
    book1D(hNhitsL1_,"hNhitsLayer1","number of hits on layer 1;#hits;#",Nx,x1,x2);
    book1D(hNhitsL2_,"hNhitsLayer2","number of hits on layer 2;#hits;#",Nx,x1,x2);
    book1D(hNhitsL3_,"hNhitsLayer3","number of hits on layer 3;#hits;#",Nx,x1,x2);
  }
  if (1) {
    const int Nx = 60;
    const double x1 = -3;
    const double x2 = +3;
    book1D(hdNdEtaHitsL1_,"hdNdEtaHitsLayer1","dN/d#eta hits on layer 1;#eta;dN/d#eta",Nx,x1,x2);
    book1D(hdNdEtaHitsL2_,"hdNdEtaHitsLayer2","dN/d#eta hits on layer 2;#eta;dN/d#eta",Nx,x1,x2);
    book1D(hdNdEtaHitsL3_,"hdNdEtaHitsLayer3","dN/d#eta hits on layer 3;#eta;dN/d#eta",Nx,x1,x2);
  }
  if (1) {
    const int Nx = 64;
    const double x1 = -3.2;
    const double x2 = +3.2;
    book1D(hdNdPhiHitsL1_,"hdNdPhiHitsLayer1","dN/d#phi hits on layer 1;#phi;dN/d#phi",Nx,x1,x2);
    book1D(hdNdPhiHitsL2_,"hdNdPhiHitsLayer2","dN/d#phi hits on layer 2;#phi;dN/d#phi",Nx,x1,x2);
    book1D(hdNdPhiHitsL3_,"hdNdPhiHitsLayer3","dN/d#phi hits on layer 3;#phi;dN/d#phi",Nx,x1,x2);
  }
  if (1) {
    const int Nx = 100;
    const double x1 = -25;
    const double x2 = +25;
    if (pixLayers_>=12)
      book1D(hTrkVtxZ12_,"hTrackletVtxZ12","z vertex from tracklets12;z [cm];#",Nx,x1,x2);
    if (pixLayers_>=13)
      book1D(hTrkVtxZ13_,"hTrackletVtxZ13","z vertex from tracklets13;z [cm];#",Nx,x1,x2);
    if (pixLayers_>=23)
      book1D(hTrkVtxZ23_,"hTrackletVtxZ23","z vertex from tracklets23;z [cm];#",Nx,x1,x2);
  }

  if (0) {
    const int Nx = 100;
    const double x1 = -5;
    const double x2 = +5;
    const int Ny = 64;
    const double y1 = -3.2;
    const double y2 = +3.2;
    if (pixLayers_>=12)
      book2D(hTrkRawDetaDphi12_,"hTracklet12RawDetaDphi",
             "tracklet12 raw #Delta#eta vs #Delta#phi;#Delta#Eta;#Delta#Phi",Nx,x1,x2,Ny,y1,y2); 
    if (pixLayers_>=13)
      book2D(hTrkRawDetaDphi13_,"hTracklet13RawDetaDphi",
             "tracklet13 raw #Delta#eta vs #Delta#phi;#Delta#Eta;#Delta#Phi",Nx,x1,x2,Ny,y1,y2); 
    if (pixLayers_>=23)
      book2D(hTrkRawDetaDphi23_,"hTracklet23RawDetaDphi",
             "tracklet12 raw #Delta#eta vs #Delta#phi;#Delta#Eta;#Delta#Phi",Nx,x1,x2,Ny,y1,y2); 
  }

  if (0) {
    const int Nx = 100;
    const double x1 = -5;
    const double x2 = +5;
    if (pixLayers_>=12)
      book1D(hTrkRawDeta12_,"hTracklet12RawDeta",
             "tracklet12 raw dN/#Delta#eta;#Delta#eta;dN/#Delta#eta",Nx,x1,x2); 
    if (pixLayers_>=13)
      book1D(hTrkRawDeta13_,"hTracklet13RawDeta",
             "tracklet13 raw dN/#Delta#eta;#Delta#eta;dN/#Delta#eta",Nx,x1,x2); 
    if (pixLayers_>=23)
      book1D(hTrkRawDeta23_,"hTracklet23RawDeta",
             "tracklet23 raw dN/#Delta#eta;#Delta#eta;dN/#Delta#eta",Nx,x1,x2); 
  }

  if (0) {
    const int Nx = 64;
    const double x1 = -3.2;
    const double x2 = +3.2;
    if (pixLayers_>=12)
      book1D(hTrkRawDphi12_,"hTracklet12RawDphi",
             "tracklet12 raw dN/#Delta#phi;#Delta#phi;dN/#Delta#phi",Nx,x1,x2); 
    if (pixLayers_>=13)
      book1D(hTrkRawDphi13_,"hTracklet13RawDphi",
             "tracklet13 raw dN/#Delta#phi;#Delta#phi;dN/#Delta#phi",Nx,x1,x2); 
    if (pixLayers_>=23)
      book1D(hTrkRawDphi23_,"hTracklet23RawDphi",
             "tracklet23 raw dN/#Delta#phi;#Delta#phi;dN/#Delta#phi",Nx,x1,x2); 
  }

  if (AlphaTracklets12_) {
    TAxis *xa = AlphaTracklets12_->GetXaxis();
    const int Nx = xa->GetNbins();
    const double x1 = xa->GetBinLowEdge(1);
    const double x2 = xa->GetBinLowEdge(Nx+1);
    book1D(hdNdEtaRawTrkl12_,"hdNdEtaRawTracklets12",
           "raw dN/d#eta for tracklets12;#eta;dN/d#eta",Nx,x1,x2,0,0); 
    book1D(hdNdEtaTrklets12_,"hdNdEtaTracklets12",
           "dN/d#eta for tracklets12;#eta;dN/d#eta",Nx,x1,x2,0,0); 
  }

  if (AlphaTracklets13_) {
    TAxis *xa = AlphaTracklets13_->GetXaxis();
    const int Nx = xa->GetNbins();
    const double x1 = xa->GetBinLowEdge(1);
    const double x2 = xa->GetBinLowEdge(Nx+1);
    book1D(hdNdEtaRawTrkl13_,"hdNdEtaRawTracklets13",
           "raw dN/d#eta for tracklets13;#eta;dN/d#eta",Nx,x1,x2,0,0); 
    book1D(hdNdEtaTrklets13_,"hdNdEtaTracklets13",
           "dN/d#eta for tracklets13;#eta;dN/d#eta",Nx,x1,x2,0,0); 
  }

  if (AlphaTracklets23_) {
    TAxis *xa = AlphaTracklets23_->GetXaxis();
    const int Nx = xa->GetNbins();
    const double x1 = xa->GetBinLowEdge(1);
    const double x2 = xa->GetBinLowEdge(Nx+1);
    book1D(hdNdEtaRawTrkl23_,"hdNdEtaRawTracklets23",
           "raw dN/d#eta for tracklets23;#eta;dN/d#eta",Nx,x1,x2,0,0); 
    book1D(hdNdEtaTrklets23_,"hdNdEtaTracklets23",
           "dN/d#eta for tracklets23;#eta;dN/d#eta",Nx,x1,x2,0,0); 
  }

  if (1) {
    const int Nx = hltTrgNames_.size();
    const double x1 = -0.5;
    const double x2 = Nx-0.5;
    h2TrigCorr_ = theDbe_->book2D("h2TriCorr",";;",Nx,x1,x2,Nx,x1,x2);
    for(size_t i=1;i<=hltTrgNames_.size();++i) {
      h2TrigCorr_->setBinLabel(i,hltTrgNames_.at(i-1),1);
      h2TrigCorr_->setBinLabel(i,hltTrgNames_.at(i-1),2);
    }
    TH1 *h = h2TrigCorr_->getTH1();
    if (h)
      h->SetStats(0);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::endJob(void) 
{
  // At the moment, nothing needed to be done.
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::filldNdeta(const TH3F *AlphaTracklets,
                             const std::vector<TH3F*> &NrawTracklets,
                             const std::vector<TH3F*> &NsigTracklets,
                             const std::vector<TH3F*> &NbkgTracklets,
                             std::vector<MonitorElement*> &hdNdEtaRawTrkl,
                             std::vector<MonitorElement*> &hdNdEtaTrklets)
{
  // Fill raw and corrected dNdeta into histograms.

  if (!AlphaTracklets)
    return;

  for(size_t i=0;i<hdNdEtaRawTrkl.size();++i) {
    MonitorElement *mrawtrk = hdNdEtaRawTrkl.at(i);
    MonitorElement *mtrklet = hdNdEtaTrklets.at(i);
    mrawtrk->Reset();
    mtrklet->Reset();

    TH3F *hraw = NrawTracklets.at(i);
    TH3F *hsig = NsigTracklets.at(i);
    TH3F *hbkg = NbkgTracklets.at(i);

    const double norm = AlphaTracklets->GetXaxis()->GetBinWidth(1) * 
      h2TrigCorr_->getBinContent(i+1,i+1);

    for(int etabin=1;etabin<=AlphaTracklets->GetNbinsX();++etabin) {
      double dndeta        = 0;
      double dndetaraw     = 0;
      double dndetaerr2    = 0;
      double dndetarawerr2 = 0;
      double errcount = 0;
      for(int hitbin=1;hitbin<=AlphaTracklets->GetNbinsY();++hitbin) {
        for(int vzbin=1;vzbin<=AlphaTracklets->GetNbinsZ();++vzbin) {
          int gbin = AlphaTracklets->GetBin(etabin,hitbin,vzbin);
          double bs = hsig->GetBinContent(gbin);
          if (bs<=0) {
            continue;
          }
          double bb = hbkg->GetBinContent(gbin);
          double beta = bb / bs;
          double n = hraw->GetBinContent(gbin);
          double nraw = n * (1-beta);
          dndetaraw += nraw;
          double bs2= bs*bs;
          double n2 = n*n;
          double err2 = n2 * (n2/bs2*bb + bb*bb*n2/bs2/bs + beta*beta*n);
          dndetarawerr2 += err2;
          double alpha = AlphaTracklets->GetBinContent(gbin);
          dndeta += alpha*nraw;
          dndetaerr2 += alpha*alpha*err2;
          errcount += n2;
        }
      }
      if (norm) {
        dndetaraw /= norm;
        dndetarawerr2 /= norm;
        dndeta /= norm;
        dndetaerr2 /= norm;
      }
      if (errcount) {
        dndetarawerr2 /= errcount;
        dndetaerr2 /= errcount;
      }
      mrawtrk->setBinContent(etabin,dndetaraw);
      mrawtrk->setBinError(etabin,TMath::Sqrt(dndetarawerr2));
      mtrklet->setBinContent(etabin,dndeta);
      mtrklet->setBinError(etabin,TMath::Sqrt(dndetaerr2));
    }
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::endLuminosityBlock(const LuminosityBlock &l, 
                                     const EventSetup &iSetup)
{
  // Update various histograms.

  filldNdeta(AlphaTracklets12_, NrawTracklets12_,NsigTracklets12_,NbkgTracklets12_,
             hdNdEtaRawTrkl12_,hdNdEtaTrklets12_);
  filldNdeta(AlphaTracklets13_, NrawTracklets13_,NsigTracklets13_,NbkgTracklets13_,
             hdNdEtaRawTrkl13_,hdNdEtaTrklets13_);
  filldNdeta(AlphaTracklets23_, NrawTracklets23_,NsigTracklets23_,NbkgTracklets23_,
             hdNdEtaRawTrkl23_,hdNdEtaTrklets23_);
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::endRun(const edm::Run &, const edm::EventSetup &)
{
  // End run, cleanup.

  for(size_t i=0;i<NrawTracklets12_.size();++i) {
    NrawTracklets12_.at(i)->Reset();
    NsigTracklets12_.at(i)->Reset();
    NbkgTracklets12_.at(i)->Reset();
  }
  for(size_t i=0;i<NrawTracklets13_.size();++i) {
    NrawTracklets13_.at(i)->Reset();
    NsigTracklets13_.at(i)->Reset();
    NbkgTracklets13_.at(i)->Reset();
  }
  for(size_t i=0;i<NrawTracklets23_.size();++i) {
    NrawTracklets23_.at(i)->Reset();
    NsigTracklets23_.at(i)->Reset();
    NbkgTracklets23_.at(i)->Reset();
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
  getProduct(hltResName_, triggerResultsHLT, iEvent);

  for(size_t i=1;i<hltTrgBits_.size();++i) {
    int tbit = hltTrgBits_.at(i-1);
    if (i<0) //ignore unknown trigger 
      continue; 
    
    hltTrgDeci_[i] = triggerResultsHLT->accept(tbit);
    if (!triggerResultsHLT->accept(tbit)) 
      print(0, Form("Decision %i for %s",
                    (int)hltTrgDeci_.at(i), hltTrgNames_.at(i).c_str()));

    if (0) print(0, Form("Decision %i for %s",
                         (int)hltTrgDeci_.at(i), hltTrgNames_.at(i).c_str()));
  }

  // fill correlation histogram
  for(size_t i=0;i<hltTrgBits_.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    h2TrigCorr_->Fill(i,i);
    for(size_t j=i+1;j<hltTrgBits_.size();++j) {
    if (!hltTrgDeci_.at(j))
      continue;
      h2TrigCorr_->Fill(i,j);
      h2TrigCorr_->Fill(j,i);
    }
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fillPixels(const Event &iEvent) 
{
  // Fill pixel hit collections.

  bpix1_.clear();
  bpix2_.clear();
  bpix3_.clear();

  Handle<SiPixelRecHitCollection> hRecHits;
  if (!getProductSafe(pixelName_, hRecHits, iEvent)) {
    CP(2) print(2, Form("Can not obtain pixel hit collection with name %s", pixelName_.c_str()));
    return;
  }

  const SiPixelRecHitCollection *hits = hRecHits.product();
  for(SiPixelRecHitCollection::DataContainer::const_iterator hit = hits->data().begin(), 
        end = hits->data().end(); hit != end; ++hit) {

    if (!hit->isValid())
      continue;

    if (usePixelQ_) {
      if (hit->isOnEdge() || hit->hasBadPixels())
        continue;
    }

    DetId id(hit->geographicalId());
    if(id.subdetId() != int(PixelSubdetector::PixelBarrel))
      continue;

    LocalPoint lpos = LocalPoint(hit->localPosition().x(),
                                 hit->localPosition().y(),
                                 hit->localPosition().z());
    GlobalPoint gpos = tgeo_->idToDet(id)->toGlobal(lpos);
    Pixel pix(gpos);

    PXBDetId pid(id);
    int layer = pid.layer();

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
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fillTracklets(const Event &iEvent, int which) 
{
  // Fill pixel hit collections.

  if (which>=12) {
    fillTracklets(btracklets12_,bpix1_,bpix2_,trackletV12_); 
    fillTracklets(btracklets12_,bpix1_,trackletV12_,
                  AlphaTracklets12_, NrawTracklets12_,
                  NsigTracklets12_, NbkgTracklets12_,
                  hTrkRawDetaDphi12_,hTrkRawDeta12_,hTrkRawDphi12_);
  }
  if (which>=13) {
    fillTracklets(btracklets13_,bpix1_,bpix3_,trackletV12_);
    fillTracklets(btracklets13_,bpix1_,trackletV13_,
                  AlphaTracklets13_, NrawTracklets12_,
                  NsigTracklets13_, NbkgTracklets13_,
                  hTrkRawDetaDphi13_,hTrkRawDeta13_,hTrkRawDphi13_);
  }
  if (which>=23) {
    fillTracklets(btracklets23_,bpix2_,bpix3_,trackletV12_);
    fillTracklets(btracklets23_,bpix1_,trackletV12_,
                  AlphaTracklets23_, NrawTracklets23_,
                  NsigTracklets23_, NbkgTracklets23_,
                  hTrkRawDetaDphi23_,hTrkRawDeta23_,hTrkRawDphi23_);
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

  if (trackletV.z()>ZVCut_)
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
   }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fillTracklets(const std::vector<Tracklet> &tracklets, 
                                const std::vector<Pixel> &pixels,
                                const Vertex &trackletV,
                                const TH3F *AlphaTracklets,
                                std::vector<TH3F*> &NrawTracklets,
                                std::vector<TH3F*> &NsigTracklets,
                                std::vector<TH3F*> &NbkgTracklets,
                                std::vector<MonitorElement*> &detaphi,
                                std::vector<MonitorElement*> &deta,
                                std::vector<MonitorElement*> &dphi)
{
  // Fill tracklet related histograms.

  if (!AlphaTracklets)
    return;

  TAxis *xa = AlphaTracklets->GetXaxis();
  int ybin = AlphaTracklets->GetYaxis()->FindBin(pixels.size());
  int zbin = AlphaTracklets->GetZaxis()->FindBin(trackletV.z());
  int tbin = AlphaTracklets->GetBin(0,ybin,zbin);
 
  for(size_t k=0; k<tracklets.size(); ++k) {
    const Tracklet &tl(tracklets.at(k));
    fill2D(detaphi,tl.deta(),tl.dphi());
    fill1D(deta,tl.deta());
    fill1D(dphi,tl.dphi());
    int gbin = xa->FindBin(tl.eta()) + tbin;
    fill3D(NrawTracklets,gbin);
    if (TMath::Abs(tl.deta())>0.1)
      continue;
    double dphi = TMath::Abs(tl.dphi());
    if (dphi<1)
      fill3D(NsigTracklets,gbin);
    else if (dphi<2)
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
    edm::LogWarning("QcdLowPtDQM") << msg << std::endl;
  } else if (level==2) {
    edm::LogError("QcdLowPtDQM") << msg << std::endl;
  } else if (level==3) {
    edm::LogError("QcdLowPtDQM") << msg << std::endl;
    throw edm::Exception(edm::errors::Configuration, "QcdLowPtDQM\n") << msg << std::endl;
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::trackletVertexUnbinned(const edm::Event &iEvent, int which)
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
    const double r12 = p1.x()*p1.x()+p1.y()*p1.y()+p1.z()*p1.z();
    for(size_t j = 0; j<pix2.size(); ++j) {
      const Pixel &p2(pix2.at(j));
      if (TMath::Abs(Geom::deltaPhi(p1.phi(),p2.phi()))>dPhiVc_)
        continue;
      const double r22 = p2.x()*p2.x()+p2.y()*p2.y()+p2.z()*p2.z();
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
void QcdLowPtDQM::yieldAlphaHistogram()
{
  // Create alpha histogram. Code created by Yen-Jie and included by hand:
  // Alpha value for 1st + 2nd tracklet calculated from 1.9 M PYTHIA 900 GeV
  // sample produced by Yetkin with CMS official tune.

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

