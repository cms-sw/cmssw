// $Id:$

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

using namespace std;
using namespace edm;
//using namespace reco;

#define CP(level) \
  if (level>=verbose_)

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
  usePixelQ_(parameters.getUntrackedParameter<int>("usePixelQualityWord",0)),
  tgeo_(0),
  theDbe_(0),
  hNhitsL1_(0),
  hNhitsL2_(0),
  hNhitsL3_(0),
  hdNdEtaHitsL1_(0),
  hdNdEtaHitsL2_(0),
  hdNdEtaHitsL3_(0),
  hdNdPhiHitsL1_(0),
  hdNdPhiHitsL2_(0),
  hdNdPhiHitsL3_(0),
  hTrkVtxZ_(0),
  hTrkRawdEtadPhi(0),
  hTrkRawdEta(0),
  hTrkRawddPhi(0),
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
}

//--------------------------------------------------------------------------------------------------
QcdLowPtDQM::~QcdLowPtDQM()
{
  // Destructor. *TODO* delete
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::beginJob() 
{
  // Begin job and setup the DQM store.

  theDbe_ = Service<DQMStore>().operator->();
  theDbe_->setCurrentFolder("Physics/QcdLowPt");
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::beginRun(const edm::Run &, const edm::EventSetup &iSetup)
{
  // Begin run, get and create needed structures.

  // get tracker geometry
  edm::ESHandle<TrackerGeometry> trackerHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  tgeo_ = trackerHandle.product();

  // get trigger bits
  HLTConfigProvider hltConfig;
  if (!hltConfig.init(hltProcName_))
    throw edm::Exception(edm::errors::Configuration, "QcdLowPtDQM::beginJob()\n")
      << "Can not initialize HLT config using process name " << hltProcName_ << std::endl;

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
  
  bookHistos();
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::bookHistos()
{
  // Book histograms. *TODO* delete histograms

  if (1) {
    const int Nx = 25;
    const double x1 = 0;
    const double x2 = 150;
    hNhitsL1_ = theDbe_->book1D("hNhitsLayer1","#hits on layer 1;#;#", Nx, x1, x2);
    hNhitsL2_ = theDbe_->book1D("hNhitsLayer2","#hits on layer 2;#;#", Nx, x1, x2);
    hNhitsL3_ = theDbe_->book1D("hNhitsLayer3","#hits on layer 2;#;#", Nx, x1, x2);
  }
  if (1) {
    const int Nx = 60;
    const double x1 = -3;
    const double x2 = +3;
    hdNdEtaHitsL1_ = theDbe_->book1D("hdNdEtaHitsLayer1","dNd#eta on layer 1;#eta;#", Nx, x1, x2);
    hdNdEtaHitsL2_ = theDbe_->book1D("hdNdEtaHitsLayer2","dNd#eta on layer 2;#eta;#", Nx, x1, x2);
    hdNdEtaHitsL3_ = theDbe_->book1D("hdNdEtaHitsLayer3","dNd#eta on layer 3;#eta;#", Nx, x1, x2);
  }
  if (1) {
    const int Nx = 64;
    const double x1 = -3.2;
    const double x2 = +3.2;
    hdNdPhiHitsL1_ = theDbe_->book1D("hdNdPhiHitsLayer1","dNd#phi on layer 1;#phi;#", Nx, x1, x2);
    hdNdPhiHitsL2_ = theDbe_->book1D("hdNdPhiHitsLayer2","dNd#phi on layer 2;#phi;#", Nx, x1, x2);
    hdNdPhiHitsL3_ = theDbe_->book1D("hdNdPhiHitsLayer3","dNd#phi on layer 3;#phi;#", Nx, x1, x2);
  }
  if (1) {
    const int Nx = 100;
    const double x1 = -25;
    const double x2 = +25;
    hTrkVtxZ_ = theDbe_->book1D("hTrackletVtxZ","z vertex from tracklets;#z [cm];#", Nx, x1, x2);
  }

  if (1) {
    const int Nx = hltTrgNames_.size();
    const double x1 = -0.5;
    const double x2 = Nx-0.5;
    h2TrigCorr_ = theDbe_->book2D("h2TriCorr",";;", Nx, x1, x2, Nx, x1, x2);
    for(int i=1;i<=hltTrgNames_.size();++i) {
      h2TrigCorr_->setBinLabel(i, hltTrgNames_.at(i-1), 1);
      h2TrigCorr_->setBinLabel(i, hltTrgNames_.at(i-1), 2);
    }
    TH1 *h = h2TrigCorr_->getTH1();
    if (h)
      h->SetStats(0);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::analyze(const Event &iEvent, const EventSetup &iSetup) 
{
  // Analyze the given event.

  fillHltBits(iEvent);
  fillPixels(iEvent);
  trackletVertexUnbinned();
  fillTracklets(iEvent);
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::endJob(void) 
{
  // At the moment, nothing needed to be done.
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::endRun(const edm::Run &, const edm::EventSetup &)
{
  // At the moment, nothing needed to be done.
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
    if (0) print(0, Form("Decision %i for %s",
                         (int)hltTrgDeci_.at(i), hltTrgNames_.at(i).c_str()));
  }

  // fill correlation histogram
  for(size_t i=0;i<hltTrgBits_.size();++i) {
    h2TrigCorr_->Fill(i,i);
    for(size_t j=i+1;j<hltTrgBits_.size();++j) {
      h2TrigCorr_->Fill(i,j);
      h2TrigCorr_->Fill(j,1);
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
      hdNdEtaHitsL1_->Fill(pix.eta());
      hdNdPhiHitsL1_->Fill(pix.phi());
    } else if (layer==2) {
      bpix2_.push_back(pix);     
      hdNdEtaHitsL2_->Fill(pix.eta());
      hdNdPhiHitsL2_->Fill(pix.phi());
    } else {
      bpix3_.push_back(pix);     
      hdNdEtaHitsL3_->Fill(pix.eta());
      hdNdPhiHitsL3_->Fill(pix.phi());
    }
  }

  // fill overall histograms
  hNhitsL1_->Fill(bpix1_.size());
  hNhitsL2_->Fill(bpix2_.size());
  hNhitsL3_->Fill(bpix3_.size());
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fillTracklets(const Event &iEvent, int which) 
{
  // Fill pixel hit collections.

  if (which>=12)
    fillTracklets(btracklets12_,bpix1_,bpix2_);
  if (which>=13)
    fillTracklets(btracklets13_,bpix1_,bpix3_);
  if (which>=23)
    fillTracklets(btracklets23_,bpix2_,bpix3_);
}

//--------------------------------------------------------------------------------------------------
void QcdLowPtDQM::fillTracklets(std::vector<Tracklet> &tracklets, 
                                std::vector<Pixel> &pix1, std::vector<Pixel> &pix2)
{
  // Fill tracklet collection from given pixel hit collections.

  tracklets.clear();

  if (trackletV_.z()>ZVCut_)
    return;

  // build tracklets
  std::vector<Tracklet> tmptrkls;
  tmptrkls.reserve(pix1.size()*pix2.size());
  for(size_t i = 0; i<pix1.size(); ++i) {
    const GlobalPoint tmp1(pix1.at(i).x(),pix1.at(i).y(),pix1.at(i).z()-trackletV_.z());
    const Pixel p1(tmp1);
    for(size_t j = 0; j<pix2.size(); ++j) {
      const GlobalPoint tmp2(pix2.at(j).x(),pix2.at(j).y(),pix2.at(j).z()-trackletV_.z());
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
    size_t p2ind = tmptrkls.at(k).i2();
    if (secused.at(p2ind)) 
      continue;
    secused[p2ind] = true;
    tracklets.push_back(tmptrkls.at(k));
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
void QcdLowPtDQM::trackletVertexUnbinned()
{
  // Calculate tracklet based z vertex position. 
  // At first build zvertex candidates from tracklet prototypes,
  // then group zvertex candidates and calculate mean position
  // from most likely cluster.

  vector<double> zvCands;
  zvCands.reserve(bpix1_.size()*bpix2_.size());

  // build candidates
  for(size_t i = 0; i<bpix1_.size(); ++i) {
    const Pixel &p1(bpix1_.at(i));
    const double r12 = p1.x()*p1.x()+p1.y()*p1.y()+p1.z()*p1.z();
    for(size_t j = 0; j<bpix2_.size(); ++j) {
      const Pixel &p2(bpix2_.at(j));
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
  trackletV_.set(mcl, mzv, ms2);
  hTrkVtxZ_->Fill(mzv);
}
