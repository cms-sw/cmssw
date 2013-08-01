//
// Package:    GEMSimHitAnalyzer
// Class:      GEMSimHitAnalyzer
// 
// \class GEMSimHitAnalyzer
//
// Description: Analyzer GEM SimHit information
// To be used for GEM algorithm development.
//
//

// system include files
#include <memory>
#include <algorithm>
#include <set>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "GEMCode/GEMValidation/src/SimTrackMatchManager.h"

#include "TTree.h"


struct MyGEMSimHit
{  
  Int_t eventNumber;
  Int_t detUnitId, particleType;
  Float_t x, y, energyLoss, pabs, timeOfFlight;
  Int_t region, ring, station, layer, chamber, roll;
  Float_t globalR, globalEta, globalPhi, globalX, globalY, globalZ;
  Int_t strip;
  Float_t Phi_0, DeltaPhi, R_0;
};

struct MySimTrack
{
  Float_t charge, pt, eta, phi;
  Char_t endcap;
  Char_t gem_sh_layer1, gem_sh_layer2; // bit1: in odd  bit2: even
  Float_t gem_sh_eta, gem_sh_phi;
  Float_t gem_trk_eta, gem_trk_phi, gem_trk_rho;
  Float_t gem_lx_even, gem_ly_even;
  Float_t gem_lx_odd, gem_ly_odd;
  Char_t  has_gem_sh_l1, has_gem_sh_l2;
};


class GEMSimHitAnalyzer : public edm::EDAnalyzer
{
public:
 /// Constructor
  explicit GEMSimHitAnalyzer(const edm::ParameterSet& iConfig);
  /// Destructor
  ~GEMSimHitAnalyzer();
  
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  
  void bookGEMSimHitsTree();
  void bookSimTracksTree();
  
  void analyzeGEM( const edm::Event& iEvent );
  bool isSimTrackGood(const SimTrack &t);
  void analyzeTracks(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void buildLUT();
  std::pair<int,int> getClosestChambers(int region, float phi);

  TTree* gem_sh_tree_;
  TTree* track_tree_;
  
  const GEMGeometry* gem_geometry_;
  
  MyGEMSimHit gem_sh;
  MySimTrack  track;

  edm::Handle<edm::SimVertexContainer> simVertices;
  edm::Handle<edm::SimTrackContainer> simTracks;
  edm::Handle<edm::PSimHitContainer> GEMHits;
  edm::ESHandle<GEMGeometry> gem_geom;
 
  edm::ParameterSet cfg_;
  std::string simInputLabel_;
  float minPt_;
  int verbose_;
  float radiusCenter_;
  float chamberHeight_;

  std::pair<std::vector<float>,std::vector<int> > positiveLUT_;
  std::pair<std::vector<float>,std::vector<int> > negativeLUT_;
};

// Constructor
GEMSimHitAnalyzer::GEMSimHitAnalyzer(const edm::ParameterSet& ps)
: cfg_(ps.getParameterSet("simTrackMatching"))
, simInputLabel_(ps.getUntrackedParameter<std::string>("simInputLabel", "g4SimHits"))
, minPt_(ps.getUntrackedParameter<double>("minPt", 4.5))
, verbose_(ps.getUntrackedParameter<int>("verbose", 0))
{
  bookGEMSimHitsTree();
  bookSimTracksTree();
}


GEMSimHitAnalyzer::~GEMSimHitAnalyzer()
{
}


void GEMSimHitAnalyzer::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup)
{
  iSetup.get<MuonGeometryRecord>().get(gem_geom);
  gem_geometry_ = &*gem_geom;

  const auto top_chamber = static_cast<const GEMEtaPartition*>(gem_geometry_->idToDetUnit(GEMDetId(1,1,1,1,1,1)));
   // TODO: it's really bad to hardcode max partition number!
  const auto bottom_chamber = static_cast<const GEMEtaPartition*>(gem_geometry_->idToDetUnit(GEMDetId(1,1,1,1,1,6)));
  const float top_half_striplength = top_chamber->specs()->specificTopology().stripLength()/2.;
  const float bottom_half_striplength = bottom_chamber->specs()->specificTopology().stripLength()/2.;
  const LocalPoint lp_top(0., top_half_striplength, 0.);
  const LocalPoint lp_bottom(0., -bottom_half_striplength, 0.);
  const GlobalPoint gp_top = top_chamber->toGlobal(lp_top);
  const GlobalPoint gp_bottom = bottom_chamber->toGlobal(lp_bottom);

  radiusCenter_ = (gp_bottom.perp() + gp_top.perp())/2.;
  chamberHeight_ = gp_top.perp() - gp_bottom.perp();

  using namespace std;
  cout<<"half top "<<top_half_striplength<<" bot "<<lp_bottom<<endl;
  cout<<"r  top "<<gp_top.perp()<<" bot "<<gp_bottom.perp()<<endl;
  LocalPoint p0(0.,0.,0.);
  cout<<"r0 top "<<top_chamber->toGlobal(p0).perp()<<" bot "<< bottom_chamber->toGlobal(p0).perp()<<endl;
  cout<<"rch "<<radiusCenter_<<" hch "<<chamberHeight_<<endl;

  buildLUT();
}


void GEMSimHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  iEvent.getByLabel(simInputLabel_, simVertices);
  iEvent.getByLabel(simInputLabel_, simTracks);
  analyzeTracks(iEvent,iSetup);

  iEvent.getByLabel(edm::InputTag(simInputLabel_,"MuonGEMHits"), GEMHits);
  if(GEMHits->size()) analyzeGEM( iEvent );
}


void GEMSimHitAnalyzer::bookGEMSimHitsTree()
{
  edm::Service< TFileService > fs;
  gem_sh_tree_ = fs->make< TTree >("GEMSimHits", "GEMSimHits");
  gem_sh_tree_->Branch("eventNumber", &gem_sh.eventNumber);
  gem_sh_tree_->Branch("detUnitId", &gem_sh.detUnitId);
  gem_sh_tree_->Branch("particleType", &gem_sh.particleType);
  gem_sh_tree_->Branch("x", &gem_sh.x);
  gem_sh_tree_->Branch("y", &gem_sh.y);
  gem_sh_tree_->Branch("energyLoss", &gem_sh.energyLoss);
  gem_sh_tree_->Branch("pabs", &gem_sh.pabs);
  gem_sh_tree_->Branch("timeOfFlight", &gem_sh.timeOfFlight);
  gem_sh_tree_->Branch("region", &gem_sh.region);
  gem_sh_tree_->Branch("ring", &gem_sh.ring);
  gem_sh_tree_->Branch("station", &gem_sh.station);
  gem_sh_tree_->Branch("chamber", &gem_sh.chamber);
  gem_sh_tree_->Branch("layer", &gem_sh.layer);
  gem_sh_tree_->Branch("roll", &gem_sh.roll);
  gem_sh_tree_->Branch("globalR", &gem_sh.globalR);
  gem_sh_tree_->Branch("globalEta", &gem_sh.globalEta);
  gem_sh_tree_->Branch("globalPhi", &gem_sh.globalPhi);
  gem_sh_tree_->Branch("globalX", &gem_sh.globalX);
  gem_sh_tree_->Branch("globalY", &gem_sh.globalY);
  gem_sh_tree_->Branch("globalZ", &gem_sh.globalZ);
  gem_sh_tree_->Branch("strip", &gem_sh.strip);
  gem_sh_tree_->Branch("Phi_0", &gem_sh.Phi_0);
  gem_sh_tree_->Branch("DeltaPhi", &gem_sh.DeltaPhi);
  gem_sh_tree_->Branch("R_0", &gem_sh.R_0);
}


void GEMSimHitAnalyzer::bookSimTracksTree()
{
  edm::Service< TFileService > fs;
  track_tree_ = fs->make< TTree >("Tracks", "Tracks");
  track_tree_->Branch("charge",&track.charge);
  track_tree_->Branch("pt",&track.pt);
  track_tree_->Branch("eta",&track.eta);
  track_tree_->Branch("phi",&track.phi);
  track_tree_->Branch("endcap",&track.endcap);
  track_tree_->Branch("gem_sh_layer1",&track.gem_sh_layer1);
  track_tree_->Branch("gem_sh_layer2",&track.gem_sh_layer2);
  track_tree_->Branch("gem_sh_eta",&track.gem_sh_eta);
  track_tree_->Branch("gem_sh_phi",&track.gem_sh_phi);
  track_tree_->Branch("gem_trk_eta",&track.gem_trk_eta);
  track_tree_->Branch("gem_trk_phi",&track.gem_trk_phi);
  track_tree_->Branch("gem_trk_rho",&track.gem_trk_rho);
  track_tree_->Branch("gem_lx_even",&track.gem_lx_even);
  track_tree_->Branch("gem_ly_even",&track.gem_ly_even);
  track_tree_->Branch("gem_lx_odd",&track.gem_lx_odd);
  track_tree_->Branch("gem_ly_odd",&track.gem_ly_odd);
  track_tree_->Branch("has_gem_sh_l1",&track.has_gem_sh_l1);
  track_tree_->Branch("has_gem_sh_l2",&track.has_gem_sh_l2);
}


void GEMSimHitAnalyzer::analyzeGEM( const edm::Event& iEvent )
{
  for (edm::PSimHitContainer::const_iterator itHit = GEMHits->begin(); itHit != GEMHits->end(); ++itHit)
  {
    gem_sh.eventNumber = iEvent.id().event();
    gem_sh.detUnitId = itHit->detUnitId();
    gem_sh.particleType = itHit->particleType();
    gem_sh.x = itHit->localPosition().x();
    gem_sh.y = itHit->localPosition().y();
    gem_sh.energyLoss = itHit->energyLoss();
    gem_sh.pabs = itHit->pabs();
    gem_sh.timeOfFlight = itHit->timeOfFlight();
    
    const GEMDetId id(itHit->detUnitId());
    
    gem_sh.region = id.region();
    gem_sh.ring = id.ring();
    gem_sh.station = id.station();
    gem_sh.layer = id.layer();
    gem_sh.chamber = id.chamber();
    gem_sh.roll = id.roll();

    const LocalPoint p0(0., 0., 0.);
    const GlobalPoint Gp0(gem_geometry_->idToDet(itHit->detUnitId())->surface().toGlobal(p0));

    gem_sh.Phi_0 = Gp0.phi();
    gem_sh.R_0 = Gp0.perp();
    gem_sh.DeltaPhi = atan(-1*id.region()*pow(-1,id.chamber())*itHit->localPosition().x()/(Gp0.perp() + itHit->localPosition().y()));
 
    const LocalPoint hitLP(itHit->localPosition());
    const GlobalPoint hitGP(gem_geometry_->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP));
    gem_sh.globalR = hitGP.perp();
    gem_sh.globalEta = hitGP.eta();
    gem_sh.globalPhi = hitGP.phi();
    gem_sh.globalX = hitGP.x();
    gem_sh.globalY = hitGP.y();
    gem_sh.globalZ = hitGP.z();

    //  Now filling strip info using entry point rather than local position to be
    //  consistent with digi strips. To change back, just switch the comments - WHF
    //  gem_sh.strip=gem_geometry_->etaPartition(itHit->detUnitId())->strip(hitLP);
    const LocalPoint hitEP(itHit->entryPoint());
    gem_sh.strip=gem_geometry_->etaPartition(itHit->detUnitId())->strip(hitEP);
    
    gem_sh_tree_->Fill();
  }
}

bool GEMSimHitAnalyzer::isSimTrackGood(const SimTrack &t)
{
  // SimTrack selection
  if (t.noVertex()) return false;
  if (t.noGenpart()) return false;
  if (std::abs(t.type()) != 13) return false; // only interested in direct muon simtracks
  if (t.momentum().pt() < minPt_) return false;
  const float eta(std::abs(t.momentum().eta()));
  if (eta > 2.18 || eta < 1.55) return false; // no GEMs could be in such eta
  return true;
}

void GEMSimHitAnalyzer::analyzeTracks(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  const edm::SimVertexContainer & sim_vert(*simVertices.product());
  
  for (auto& t: *simTracks.product())
  {
    if (!isSimTrackGood(t)) continue;
    
    // match hits and digis to this SimTrack
    const SimTrackMatchManager match(t, sim_vert[t.vertIndex()], cfg_, iEvent, iSetup);
    const SimHitMatcher& match_sh = match.simhits();
   
    track.pt = t.momentum().pt();
    track.phi = t.momentum().phi();
    track.eta = t.momentum().eta();
    track.charge = t.charge();
    track.endcap = (track.eta > 0.) ? 1 : -1;
    track.gem_sh_layer1 = 0;
    track.gem_sh_layer2 = 0;
    track.gem_sh_eta = -9.;
    track.gem_sh_phi = -9.;
    track.gem_trk_eta = -999.;
    track.gem_trk_phi = -999.;
    track.gem_trk_rho = -999.;
    track.gem_lx_even = -999.;
    track.gem_ly_even = -999.;
    track.gem_lx_odd = -999.;
    track.gem_ly_odd = -999.;
    track.has_gem_sh_l1 = 0;
    track.has_gem_sh_l2 = 0;


    // ** GEM SimHits ** //
    auto gem_sh_ids_sch = match_sh.superChamberIdsGEM();
    for(auto d: gem_sh_ids_sch)
    {
      const auto gem_simhits(match_sh.hitsInSuperChamber(d));
      const auto gem_simhits_gp(match_sh.simHitsMeanPosition(gem_simhits));

      track.gem_sh_eta = gem_simhits_gp.eta();
      track.gem_sh_phi = gem_simhits_gp.phi();
    }

    // Calculation of the localXY efficiency
    GlobalPoint gp_track(match_sh.propagatedPositionGEM());
    track.gem_trk_eta = gp_track.eta();
    track.gem_trk_phi = gp_track.phi();
    track.gem_trk_rho = gp_track.perp();
    std::cout << "track eta phi rho = " << track.gem_trk_eta << " " << track.gem_trk_phi << " " << track.gem_trk_rho << std::endl;
    
    float track_angle = gp_track.phi().degrees();
    if (track_angle < 0.) track_angle += 360.;
    std::cout << "track angle = " << track_angle << std::endl;
    const int track_region = (gp_track.z() > 0 ? 1 : -1);
    
    // closest chambers in phi
    const auto mypair = getClosestChambers(track_region, track_angle);
    
    // chambers
    GEMDetId detId_first(mypair.first);
    GEMDetId detId_second(mypair.second);

    // assignment of local even and odd chambers (there is always an even and an odd chamber)
    bool firstIsOdd = detId_first.chamber() & 1;
    
    GEMDetId detId_even_L1(firstIsOdd ? detId_second : detId_first);
    GEMDetId detId_odd_L1(firstIsOdd ? detId_first  : detId_second);

    auto even_partition = gem_geometry_->idToDetUnit(detId_even_L1)->surface();
    auto odd_partition  = gem_geometry_->idToDetUnit(detId_odd_L1)->surface();

    // global positions of partitions' centers
    LocalPoint p0(0.,0.,0.);
    GlobalPoint gp_even_partition = even_partition.toGlobal(p0);
    GlobalPoint gp_odd_partition = odd_partition.toGlobal(p0);
    
    LocalPoint lp_track_even_partition = even_partition.toLocal(gp_track);
    LocalPoint lp_track_odd_partition = odd_partition.toLocal(gp_track);

    // track chamber local x is the same as track partition local x
    track.gem_lx_even = lp_track_even_partition.x();
    track.gem_lx_odd = lp_track_odd_partition.x();

    // track chamber local y is the same as track partition local y
    // corrected for partition's local y WRT chamber
    track.gem_ly_even = lp_track_even_partition.y() + (gp_even_partition.perp() - radiusCenter_);
    track.gem_ly_odd = lp_track_odd_partition.y() + (gp_odd_partition.perp() - radiusCenter_);

    std::cout << track.gem_lx_even << " " << track.gem_ly_even << std::endl;
    std::cout << track.gem_lx_odd << " " << track.gem_ly_odd << std::endl;

    
    // check for hit chambers
    const auto gem_sh_ids_ch = match_sh.chamberIdsGEM();
    for(auto d: gem_sh_ids_ch)
    {
      const GEMDetId id(d);
      const bool odd(id.chamber() & 1);
      
      if (id.layer() == 1)
      {
        if (odd) track.gem_sh_layer1 |= 1;
        else     track.gem_sh_layer1 |= 2;
      }
      else if (id.layer() == 2)
      {
        if (odd) track.gem_sh_layer2 |= 1;
        else     track.gem_sh_layer2 |= 2;
      }
    }
  
    // Construct Chamber DetIds from the "projected" ids:
    GEMDetId id_ch_even_L1(detId_even_L1.region(), detId_even_L1.ring(), detId_even_L1.station(), 1, detId_even_L1.chamber(), 0);
    GEMDetId id_ch_odd_L1(detId_odd_L1.region(), detId_odd_L1.ring(), detId_odd_L1.station(), 1, detId_odd_L1.chamber(), 0);
    GEMDetId id_ch_even_L2(detId_even_L1.region(), detId_even_L1.ring(), detId_even_L1.station(), 2, detId_even_L1.chamber(), 0);
    GEMDetId id_ch_odd_L2(detId_odd_L1.region(), detId_odd_L1.ring(), detId_odd_L1.station(), 2, detId_odd_L1.chamber(), 0);

    // check if track has sh 
    if(gem_sh_ids_ch.count(id_ch_even_L1)!=0) track.has_gem_sh_l1 |= 2;
    if(gem_sh_ids_ch.count(id_ch_odd_L1)!=0)  track.has_gem_sh_l1 |= 1;
    if(gem_sh_ids_ch.count(id_ch_even_L2)!=0) track.has_gem_sh_l2 |= 2;
    if(gem_sh_ids_ch.count(id_ch_odd_L2)!=0)  track.has_gem_sh_l2 |= 1;

    track_tree_->Fill();
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GEMSimHitAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void GEMSimHitAnalyzer::buildLUT()
{
  std::vector<int> pos_ids;
  pos_ids.push_back(GEMDetId(1,1,1,1,36,1).rawId());

  std::vector<int> neg_ids;
  neg_ids.push_back(GEMDetId(-1,1,1,1,36,1).rawId());

  // VK: I would really suggest getting phis from GEMGeometry
  
  std::vector<float> phis;
  phis.push_back(0.);
  for(int i=1; i<37; ++i)
  {
    pos_ids.push_back(GEMDetId(1,1,1,1,i,1).rawId());
    neg_ids.push_back(GEMDetId(-1,1,1,1,i,1).rawId());
    phis.push_back(i*10.);
  }
  positiveLUT_ = std::make_pair(phis,pos_ids);
  negativeLUT_ = std::make_pair(phis,neg_ids);
}

std::pair<int,int>
GEMSimHitAnalyzer::getClosestChambers(int region, float phi)
{
  auto& phis(positiveLUT_.first);
  auto upper = std::upper_bound(phis.begin(), phis.end(), phi);
  std::cout << "lower = " << upper - phis.begin()  << std::endl;
  std::cout << "upper = " << upper - phis.begin() + 1 << std::endl;
  auto& LUT = (region == 1 ? positiveLUT_.second : negativeLUT_.second);
  return std::make_pair(LUT.at(upper - phis.begin()), (LUT.at((upper - phis.begin() + 1)%36)));
}

//define this as a plug-in
DEFINE_FWK_MODULE(GEMSimHitAnalyzer);

