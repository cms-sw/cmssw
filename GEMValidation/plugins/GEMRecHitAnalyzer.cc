#include <memory>
#include <fstream>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

// root include files
#include "TTree.h"
#include "TFile.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include <DataFormats/MuonDetId/interface/GEMDetId.h>

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
 
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include "CommonTools/UtilAlgos/interface/TFileService.h"

//
// class declaration
//
struct MyGEMRecHit
{  
  Int_t detId, particleType;
  Float_t x, y;
  Int_t region, ring, station, layer, chamber, roll;
  Float_t globalR, globalEta, globalPhi, globalX, globalY, globalZ;
  Int_t bx, clusterSize, firstClusterStrip;
};

class GEMRecHitAnalyzer : public edm::EDAnalyzer 
{
public:
  /// constructor
  explicit GEMRecHitAnalyzer(const edm::ParameterSet&);
  /// destructor
  ~GEMRecHitAnalyzer();

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);

  virtual void beginJob() ;

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void endJob() ;
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
 
  void bookGEMRecHitTree();
  void analyzeGEM();

  TTree* gem_tree_;

  edm::Handle<GEMRecHitCollection> gemRecHits_; 
  
  edm::ESHandle<GEMGeometry> gem_geom_;

  const GEMGeometry* gem_geometry_;

  MyGEMRecHit gem_recHit_;
};

//
// constructors and destructor
//
GEMRecHitAnalyzer::GEMRecHitAnalyzer(const edm::ParameterSet& iConfig)
{
  bookGEMRecHitTree();
}

GEMRecHitAnalyzer::~GEMRecHitAnalyzer()
{
}

void GEMRecHitAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  iSetup.get<MuonGeometryRecord>().get(gem_geom_);
  gem_geometry_ = &*gem_geom_;
}


void GEMRecHitAnalyzer::beginJob()
{
}

void GEMRecHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  iEvent.getByLabel("gemRecHits","",gemRecHits_);
  analyzeGEM();
}
void GEMRecHitAnalyzer::bookGEMRecHitTree()
{
  edm::Service<TFileService> fs;
  gem_tree_ = fs->make<TTree>("GEMRecHitTree", "GEMRecHitTree");
  gem_tree_->Branch("detId", &gem_recHit_.detId);
  gem_tree_->Branch("region", &gem_recHit_.region);
  gem_tree_->Branch("ring", &gem_recHit_.ring);
  gem_tree_->Branch("station", &gem_recHit_.station);
  gem_tree_->Branch("layer", &gem_recHit_.layer);
  gem_tree_->Branch("chamber", &gem_recHit_.chamber);
  gem_tree_->Branch("roll", &gem_recHit_.roll);
  gem_tree_->Branch("bx", &gem_recHit_.bx);
  gem_tree_->Branch("clusterSize", &gem_recHit_.clusterSize);
  gem_tree_->Branch("firstClusterStrip", &gem_recHit_.firstClusterStrip);
  gem_tree_->Branch("x", &gem_recHit_.x);
  gem_tree_->Branch("y", &gem_recHit_.y);
  gem_tree_->Branch("globalR", &gem_recHit_.globalR);
  gem_tree_->Branch("globalEta", &gem_recHit_.globalEta);
  gem_tree_->Branch("globalPhi", &gem_recHit_.globalPhi);
  gem_tree_->Branch("globalX", &gem_recHit_.globalX);
  gem_tree_->Branch("globalY", &gem_recHit_.globalY);
  gem_tree_->Branch("globalZ", &gem_recHit_.globalZ);
}

// ======= GEM RecHits =======
void GEMRecHitAnalyzer::analyzeGEM()
{
  for (GEMRecHitCollection::const_iterator recHit = gemRecHits_->begin(); recHit != gemRecHits_->end(); ++recHit) 
  {
    gem_recHit_.x = recHit->localPosition().x();
    gem_recHit_.y = recHit->localPosition().y();
    gem_recHit_.detId = (Short_t) (*recHit).gemId();
    gem_recHit_.bx = recHit->BunchX();
    gem_recHit_.clusterSize = recHit->clusterSize();
    gem_recHit_.firstClusterStrip = recHit->firstClusterStrip();
    
    GEMDetId id((*recHit).gemId());

    gem_recHit_.region = (Short_t) id.region();
    gem_recHit_.ring = (Short_t) id.ring();
    gem_recHit_.station = (Short_t) id.station();
    gem_recHit_.layer = (Short_t) id.layer();
    gem_recHit_.chamber = (Short_t) id.chamber();
    gem_recHit_.roll = (Short_t) id.roll();

    LocalPoint hitLP = recHit->localPosition();
    GlobalPoint hitGP = gem_geometry_->idToDet((*recHit).gemId())->surface().toGlobal(hitLP);
    
    gem_recHit_.globalR = hitGP.perp();
    gem_recHit_.globalEta = hitGP.eta();
    gem_recHit_.globalPhi = hitGP.phi();
    gem_recHit_.globalX = hitGP.x();
    gem_recHit_.globalY = hitGP.y();
    gem_recHit_.globalZ = hitGP.z();

    gem_tree_->Fill();
  }
}


void GEMRecHitAnalyzer::endJob() 
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GEMRecHitAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GEMRecHitAnalyzer);
