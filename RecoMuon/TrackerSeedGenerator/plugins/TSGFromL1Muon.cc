#include "TSGFromL1Muon.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"


#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"


#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedFromProtoTrack.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

using namespace reco;
using namespace ctfseeding;

template <class T> T sqr( T t) {return t*t;}


TSGFromL1Muon::TSGFromL1Muon(const edm::ParameterSet& cfg)
  : theConfig(cfg), theHitGenerator(0)
{
  produces<TrajectorySeedCollection>();
}

TSGFromL1Muon::~TSGFromL1Muon()
{
  delete theHitGenerator;
}

void TSGFromL1Muon::beginJob(const edm::EventSetup& es)
{
  edm::ParameterSet hitsfactoryPSet =
      theConfig.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string hitsfactoryName = hitsfactoryPSet.getParameter<std::string>("ComponentName");
  theHitGenerator = OrderedHitsGeneratorFactory::get()->create( hitsfactoryName, hitsfactoryPSet);

  edm::ParameterSet fitterPSet = theConfig.getParameter<edm::ParameterSet>("FitterPSet");
  std::string fitterName = fitterPSet.getParameter<std::string>("ComponentName");
  theFitter = PixelFitterFactory::get()->create( fitterName, fitterPSet);

}


void TSGFromL1Muon::produce(edm::Event& ev, const edm::EventSetup& es)
{
  std::auto_ptr<TrajectorySeedCollection> result(new TrajectorySeedCollection());

  // get L1 muon
  edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle;
  ev.getByLabel("l1GmtEmulDigis",gmtrc_handle);
  L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();
  std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  std::vector<L1MuGMTReadoutRecord>::const_iterator igmtrr;

  std::cout << " HERE - producing SEEDS" << std::endl;
  for(igmtrr=gmt_records.begin(); igmtrr!=gmt_records.end(); igmtrr++) {
    std::vector<L1MuGMTExtendedCand>::const_iterator gmt_iter;
    std::vector<L1MuGMTExtendedCand> exc = igmtrr->getGMTCands();
    if (exc.size() <= 0) continue;
    std::cout <<" HAS L1 MUON" << std::endl;
    L1MuGMTExtendedCand & muon = exc.front();

    float phi_rec = muon.phiValue()+0.021817;

    float dx = cos(phi_rec);
    float dy = sin(phi_rec);
    float dz = sinh(muon.etaValue());
    GlobalVector dir(dx,dy,dz);        // muon direction
    GlobalPoint vtx(0.,0.,0.);         // FIXME - develop


    // FIXME 0 preliminary optimisation for ptcut=10
    RectangularEtaPhiTrackingRegion region( dir, vtx, 10.,  0.1, 16., 0.15, 0.35);
    const OrderedSeedingHits & candidates = theHitGenerator->run(region,ev,es);
    std::cout << "*** TSGFromL1Muon, size: " << candidates.size() << std::endl;

    unsigned int nSets = candidates.size();
    for (unsigned int ic= 0; ic <nSets; ic++) {
      typedef std::vector<ctfseeding::SeedingHit> RecHits;
      const RecHits & hits = candidates[ic].hits();
      float r0 = hits[0].r();
      float r1 = hits[1].r();
      GlobalPoint p0(r0*cos(hits[0].phi()), r0*sin(hits[0].phi()), 0.);
      GlobalPoint p1(r1*cos(hits[1].phi()), r1*sin(hits[1].phi()), 0.);

      float cotTheta = (hits[1].z()-hits[0].z())/(hits[1].r()-hits[0].r());
      float eta_rec = asinh(cotTheta);

      float phi_vtx = (p1-p0).phi();

      //FIXME - rely on L1 charge?
      float pt_rec = std::max(getPt(phi_vtx, phi_rec, muon.etaValue(), muon.charge()),
                         getPt(phi_vtx, phi_rec, muon.etaValue(), -muon.charge()));


      // FIXME move it to the filter
      if (pt_rec < 8) continue;

      std::vector<const TrackingRecHit *> trh;
      for (unsigned int i= 0, nHits = hits.size(); i< nHits; ++i) trh.push_back( hits[i] );
      reco::Track* track = theFitter->run(es, trh, region);

      if (!track) continue;

      SeedFromProtoTrack seed( *track, hits, es);
      if (seed.isValid()) (*result).push_back( seed.trajectorySeed() );

//      GlobalError vtxerr( sqr(region.originRBound()), 0, sqr(region.originRBound()),
//                                               0, 0, sqr(region.originZBound()));
//      SeedFromConsecutiveHits seed( candidates[ic],region.origin(), vtxerr, es); 
//      if (seed.isValid()) (*result).push_back( seed.TrajSeed() );
      delete track;

    }
  }





  ev.put(result);
}




float TSGFromL1Muon::deltaPhi(float phi1, float phi2) const
{
  while ( phi1 >= 2*M_PI) phi1 -= 2*M_PI;
  while ( phi2 >= 2*M_PI) phi2 -= 2*M_PI;
  while ( phi1 < 0) phi1 += 2*M_PI;
  while ( phi2 < 0) phi2 += 2*M_PI;
  float dPhi = phi2-phi1;

  if ( dPhi > M_PI ) dPhi =- 2*M_PI;
  if ( dPhi < -M_PI ) dPhi =+ 2*M_PI;

  return dPhi;
}




float TSGFromL1Muon::getPt(float phi0, float phiL1, float eta, float charge) const {

  float dphi_min = fabs(deltaPhi(phi0,phiL1));
  float pt_best = 1.;
  float pt_cur = 1;
  while ( pt_cur < 100.) {
    float phi_exp = phi0+getBending(eta, pt_cur, charge);
    float dphi = fabs(deltaPhi(phi_exp,phiL1));
    if ( dphi < dphi_min) {
      pt_best = pt_cur;
      dphi_min = dphi;
    }
    pt_cur += 0.01;
  };
  return pt_best;
}




float TSGFromL1Muon::getBending(float eta, float pt, float charge) const
{
  float p1, p2;
  param(eta,p1,p2);
  return charge*p1/pt + charge*p2/(pt*pt); // - 0.0218;
}

void TSGFromL1Muon::param(float eta, float &p1, float& p2) const
{

  int ieta = int (10*fabs(eta));
  switch (ieta) {
  case 0:  { p1 = -2.658; p2 = -1.551; break; }
  case 1:
  case 2:  { p1 = -2.733; p2 = -0.6316; break; }
  case 3:  { p1 = -2.607; p2 = -2.558; break; }
  case 4:  { p1 = -2.715; p2 = -0.9311; break; }
  case 5:  { p1 = -2.674; p2 = -1.145; break; }
  case 6:  { p1 = -2.731; p2 = -0.4343; break; }
  case 7:
  case 8:  { p1 = -2.684; p2 = -0.7035; break; }
  case 9:
  case 10: { p1 = -2.659; p2 = -0.0325; break; }
  case 11: { p1 = -2.580; p2 = -0.77; break; }
  case 12: { p1 = -2412; p2 = 0.5242; break; }
  case 13: { p1 = -2.192; p2 = 1.691; break; }
  case 14:
  case 15: { p1 = -1.891; p2 = 0.8936; break; }
  case 16: { p1 = -1.873; p2 = 2.287; break; }
  case 17: { p1 = -1.636; p2 = 1.881; break; }
  case 18: { p1 = -1.338; p2 = -0.006; break; }
  case 19: { p1 = -1.334; p2 = 1.036; break; }
  case 20: { p1 = -1.247; p2 = 0.461; break; }
  default: {p1 = -1.141; p2 = 2.06; }             //above eta 2.1
  }

}

