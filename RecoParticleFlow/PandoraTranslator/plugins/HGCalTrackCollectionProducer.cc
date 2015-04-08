// S. Zenz, 12 February 2015
//
// Splits a track collection into two, based on whether they propagate to the HGCal or not
// Tracks with bad pt resolution (suspected fakes) are dropped and not in either collection

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

//#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"


// for track propagation through HGC  
// N.B. we are only propogating to first layer, so check these later
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/FCalGeometry/interface/HGCalGeometry.h"

//geometry records
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include<unordered_map>

class HGCalTrackCollectionProducer : public edm::EDProducer {

public:
  HGCalTrackCollectionProducer( const edm::ParameterSet & );
private:
  bool goodPtResolution( const reco::TrackRef& trackref ) const;
  void produce( edm::Event &, const edm::EventSetup & ) override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, 
			    const edm::EventSetup&) override;

  edm::EDGetTokenT<edm::View<reco::PFRecTrack> > _src;
  
  // variables needed for copied goodPtResolution function
  // need to go back and figure out sensible values
  bool _debug;
  const std::vector<double> _DPtovPtCut;
  const std::vector<unsigned> _NHitCut;
  const bool _useIterTracking;
  //  const bool _useFirstLayerOnly; // always true now

  // variables needed for copied extrapolation
  edm::ESHandle<MagneticField> _bField;
  edm::ESHandle<TrackerGeometry> _tkGeom;
  std::array<std::string,1> _hgc_names; // 3 --> 1; extrapolate to hgcee only
  std::array<edm::ESHandle<HGCalGeometry>,1> _hgcGeometries; // 3 --> 1; extrapolate to hgcee only
  std::array<std::vector<ReferenceCountingPointer<BoundDisk> >,1> _plusSurface,_minusSurface; // 3 --> 1; extrapolate to hgcee only
  std::unique_ptr<PropagatorWithMaterial> _mat_prop;

  float _diskOuterRadius;
  float _diskInnerRadius;
};

HGCalTrackCollectionProducer::HGCalTrackCollectionProducer(const edm::ParameterSet & iConfig) :
  _src(consumes<edm::View<reco::PFRecTrack> >(iConfig.getParameter<edm::InputTag> ("src"))),
  _debug(iConfig.getParameter<bool>("debug")),
  _DPtovPtCut(iConfig.getParameter<std::vector<double> >("DPtOverPtCuts_byTrackAlgo")),
  _NHitCut(iConfig.getParameter<std::vector<unsigned> >("NHitCuts_byTrackAlgo")),
  _useIterTracking(iConfig.getParameter<bool>("useIterativeTracking"))
  //  _useFirstLayerOnly(iConfig.getParameter<bool>("UseFirstLayerOnly")) // always true now
{

  if (_debug) std::cout << " HGCalTrackCollectionProducer::HGCalTrackCollectionProducer " << std::endl;

  const edm::ParameterSet& geoconf = iConfig.getParameterSet("hgcalGeometryNames");
  _hgc_names[0] = geoconf.getParameter<std::string>("HGC_ECAL");
  // 3 --> 1; extrapolate to hgcee only
  //  _hgc_names[1] = geoconf.getParameter<std::string>("HGC_HCALF"); 
  //  _hgc_names[2] = geoconf.getParameter<std::string>("HGC_HCALB");

  produces<reco::PFRecTrackCollection>("TracksInHGCal");
  produces<reco::PFRecTrackCollection>("TracksNotInHGCal");

}

// From https://github.com/cms-sw/cmssw/blob/CMSSW_6_2_X_SLHC/RecoParticleFlow/PFClusterProducer/src/HGCClusterizer.cc#L441-L447 and beyond
// TODO: we only need the front of the calorimeter, so modify this
void HGCalTrackCollectionProducer::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& es) {
  constexpr float m_pion = 0.1396;
  // get dependencies for setting up propagator  
  es.get<IdealMagneticFieldRecord>().get(_bField);
  es.get<TrackerDigiGeometryRecord>().get(_tkGeom);
  // get HGC geometries (assume that layers are ordered in Z!)
  for( unsigned i = 0; i < _hgcGeometries.size(); ++i ) {
    es.get<IdealGeometryRecord>().get(_hgc_names[i],_hgcGeometries[i]);
  }
  
  // make propagator
  _mat_prop.reset( new PropagatorWithMaterial(alongMomentum, m_pion, _bField.product()) );
  // setup HGC layers for track propagation
  Surface::RotationType rot; //unit rotation matrix
  for( unsigned i = 0; i < _hgcGeometries.size(); ++i ) {
    _minusSurface[i].clear();
    _plusSurface[i].clear();
    const HGCalDDDConstants &dddCons=_hgcGeometries[i]->topology().dddConstants();
    std::map<float,float> zrhoCoord;
    std::map<float,float> innerRadiusCoord;
    auto firstLayerIt = dddCons.getFirstTrForm();
    auto firstmod = dddCons.getFirstModule(true);
    float Z(fabs(firstLayerIt->h3v.z()));
    _diskInnerRadius = firstLayerIt->h3v.perp() - firstmod->h;
    _diskOuterRadius = firstmod->h+firstLayerIt->h3v.perp();
    if (_debug) std::cout << "O HAI I'm making a bound disk with Outer R=" << _diskOuterRadius << " Inner R=" << _diskInnerRadius << " and Z=" << Z << std::endl;
    _minusSurface[i].push_back(ReferenceCountingPointer<BoundDisk> ( new BoundDisk( Surface::PositionType(0,0,-Z), rot, 
										    new SimpleDiskBounds( _diskInnerRadius, _diskOuterRadius, -0.001, 0.001))));
    _plusSurface[i].push_back(ReferenceCountingPointer<BoundDisk> ( new BoundDisk( Surface::PositionType(0,0,+Z), rot, 
										   new SimpleDiskBounds( _diskInnerRadius, _diskOuterRadius, -0.001, 0.001))));
  }
}

    /*
    //    auto lastLayerIt = dddCons.getLastTrForm();
    for(auto layerIt=firstLayerIt; layerIt !=lastLayerIt; layerIt++) {
      float Z(fabs(layerIt->h3v.z()));
      auto lastmod = std::reverse_iterator<std::vector<HGCalDDDConstants::hgtrap>::const_iterator>(dddCons.getLastModule(true));
      auto firstmod = dddCons.getFirstModule(true);
      float Radius(firstmod->h+layerIt->h3v.perp());
      float innerRadius(layerIt->h3v.perp() - firstmod->h);
      if (_debug) std::cout << " SCZ MEGADEBUG Radius=" << Radius << " layerIt->h3v.perp()=" << layerIt->h3v.perp() << " firstmod->h=" << firstmod->h << std::endl;
      if (_debug) std::cout << " SCZ MEGADEBUG Radius=" << Radius << " layerIt->h3v.perp()=" << layerIt->h3v.perp() << " lastmod->h=" << lastmod->h << std::endl;
      //      if (_debug) std::cout << " SCZ MEBADEBUG std::distance(firstmod,lastmod)=" << std::distance(firstmod,lastmod) << std::endl;
      zrhoCoord[Z]=Radius;
      innerRadiusCoord[Z] = innerRadius;
      // Note from Seth:
      // I don't understand how the geometry works, and it's very hard to search the code base
      // So this is hacky as hell
      // The outer radius calculation I didn't touch, and it currently gives R = 126; with Z=320, this gives eta ~1.65
      // (plausible but I expected 1.47)
      // If I hypothesize that lastmod->tl is the inner radius, I get 28 cm for that, which gives eta ~3.13
      // (also plausible but expected 3.0)
      // sooooo.. close enough?
    }
    for(auto it=zrhoCoord.begin(); it != zrhoCoord.end(); it++) {
      float Z(it->first);
      float Radius(it->second);
      float InnerRadius(innerRadiusCoord[it->first]);
      if (_debug) std::cout << "O HAI I'm making a bound disk with Outer R=" << Radius << " Inner R=" << InnerRadius << " and Z=" << Z << std::endl;
      _minusSurface[i].push_back(ReferenceCountingPointer<BoundDisk> ( new BoundDisk( Surface::PositionType(0,0,-Z), rot, new SimpleDiskBounds( InnerRadius, Radius, -0.001, 0.001))));
      _plusSurface[i].push_back(ReferenceCountingPointer<BoundDisk> ( new BoundDisk( Surface::PositionType(0,0,+Z), rot, new SimpleDiskBounds( InnerRadius, Radius, -0.001, 0.001))));
      _diskOuterRadius = Radius;
      _diskInnerRadius = InnerRadius;
      if (_useFirstLayerOnly) break; // quick hack to take only innermost layer 
    }    
  }
}
    */

void HGCalTrackCollectionProducer::produce(edm::Event & evt, const edm::EventSetup & iSetup) {

  edm::Handle<edm::View<reco::PFRecTrack> > trackHandle;
  evt.getByToken(_src,trackHandle);
  //  const reco::PFRecTrackRefVector& tracks = trackHandle->refVector();
  //  const edm::RefToBaseVector<reco::PFRecTrack>& tracks = trackHandle->refVector();
  const edm::PtrVector<reco::PFRecTrack>& tracks = trackHandle->ptrVector();

  std::auto_ptr<reco::PFRecTrackCollection> outputInHGCal(new reco::PFRecTrackCollection);
  std::auto_ptr<reco::PFRecTrackCollection> outputNotInHGCal(new reco::PFRecTrackCollection);

  for ( unsigned int i = 0 ; i < tracks.size() ; i++) {
    bool isGood = goodPtResolution(tracks[i]->trackRef());
    if (_debug) std::cout << "HGCalTrackCollectionProducer Track number " << i << " has a goodPtResolution result of " << isGood << std::endl;
    if (!isGood) continue;
    bool found = false;
    const TrajectoryStateOnSurface myTSOS = trajectoryStateTransform::outerStateOnSurface(*(tracks[i]->trackRef()), *(_tkGeom.product()),_bField.product());
    auto detbegin = myTSOS.globalPosition().z() > 0 ? _plusSurface.begin() : _minusSurface.begin();
    auto detend = myTSOS.globalPosition().z() > 0 ? _plusSurface.end() : _minusSurface.end();
    for( auto det = detbegin; det != detend; ++det ) {  
      if (_debug) std::cout << "at HGC detector: " << std::distance(detbegin,det) << std::endl;
      unsigned layer_count = 1;
      for( const auto& layer : *det ) {
	if (_debug) std::cout << "  at DET layer: " << layer_count++ << std::endl;
	TrajectoryStateOnSurface piStateAtSurface = _mat_prop->propagate(myTSOS, *layer);
	if( piStateAtSurface.isValid() ) {
	  if (_debug) std::cout << "Extrapolation is valid!" << std::endl;
	  GlobalPoint pt = piStateAtSurface.globalPosition();
	  if (pt.perp() < _diskOuterRadius) {
	    if (pt.perp() > _diskInnerRadius) {
	      if (_debug) std::cout << "(x,y,z,r)=(" << pt.x() << ", " << pt.y() << ", " << pt.z() << ", " << sqrt(pt.x()*pt.x() + pt.y()*pt.y()) << ")" << std::endl;
	      if (_debug && fabs (tracks[i]->trackRef()->eta()) < 1.47) std::cout << " ETA IN BARREL REGION: " << tracks[i]->trackRef()->eta() 
										  << " (PT: " << tracks[i]->trackRef()->pt() << ")" << std::endl;
	      found = true;
	    } else {
	      if (_debug) std::cout << " but r=" << pt.perp() << " < diskInnerRadius=" << _diskInnerRadius << " so skipping " << std::endl;
	    }
	  } else {
	    if (_debug) std::cout << " but r=" << pt.perp() << " > diskOuterRadius=" << _diskOuterRadius << " so skipping " << std::endl;
	  }
	} else {
	  if (_debug) std::cout << "Extrapolation is NOT valid!" << std::endl;
	  //	  outputNotInHGCal->push_back(*tracks[i]);
	}
      }
    }
    if (found) {
      if (_debug) std::cout << " Track going to outputInHGCal pt eta " << tracks[i]->trackRef()->pt() << " " << tracks[i]->trackRef()->eta() << std::endl;
      outputInHGCal->push_back(*tracks[i]);
    } else {
      outputNotInHGCal->push_back(*tracks[i]);
    }
  } // track loop

  evt.put(outputInHGCal,"TracksInHGCal");
  evt.put(outputNotInHGCal,"TracksNotInHGCal");
}

// Copied from https://github.com/cms-sw/cmssw/blob/CMSSW_6_2_X_SLHC/RecoParticleFlow/PFProducer/plugins/importers/GeneralTracksImporter.cc#L140
bool HGCalTrackCollectionProducer::
goodPtResolution( const reco::TrackRef& trackref) const {

  const double P = trackref->p();
  const double Pt = trackref->pt();
  const double DPt = trackref->ptError();
  const unsigned int NHit = 
    trackref->hitPattern().trackerLayersWithMeasurement();
  const unsigned int NLostHit = 
    trackref->hitPattern().trackerLayersWithoutMeasurement();
  const unsigned int LostHits = trackref->numberOfLostHits();
  const double sigmaHad = sqrt(1.20*1.20/P+0.06*0.06) / (1.+LostHits);

  // iteration 1,2,3,4,5 correspond to algo = 1/4,5,6,7,8,9
  unsigned int Algo = 0; 
  switch (trackref->algo()) {
  case reco::TrackBase::ctf:
  case reco::TrackBase::iter0:
  case reco::TrackBase::iter1:
  case reco::TrackBase::iter2:
    Algo = 0;
    break;
  case reco::TrackBase::iter3:
    Algo = 1;
    break;
  case reco::TrackBase::iter4:
    Algo = 2;
    break;
  case reco::TrackBase::iter5:
    Algo = 3;
    break;
  case reco::TrackBase::iter6:
    Algo = 4;
    break;
  default:
    Algo = _useIterTracking ? 5 : 0;
    break;
  }

  // Protection against 0 momentum tracks
  if ( P < 0.05 ) return false;

  // Temporary : Reject all tracking iteration beyond 5th step. 
  if ( Algo > 4 ) return false;
 
  if (_debug) std::cout << " PFBlockAlgo: PFrecTrack->Track Pt= "
			<< Pt << " DPt = " << DPt << std::endl;
  if ( ( _DPtovPtCut[Algo] > 0. && 
	 DPt/Pt > _DPtovPtCut[Algo]*sigmaHad ) || 
       NHit < _NHitCut[Algo] ) { 
    // (Algo >= 3 && LostHits != 0) ) {
    if (_debug) std::cout << " PFBlockAlgo: skip badly measured track"
			  << ", P = " << P 
			  << ", Pt = " << Pt 
			  << " DPt = " << DPt 
			  << ", N(hits) = " << NHit << " (Lost : " << LostHits << "/" << NLostHit << ")"
			  << ", Algo = " << Algo
			  << std::endl;
    if (_debug) std::cout << " cut is DPt/Pt < " << _DPtovPtCut[Algo] * sigmaHad << std::endl;
    if (_debug) std::cout << " cut is NHit >= " << _NHitCut[Algo] << std::endl;
    /*
      std::cout << "Track REJECTED : ";
      std::cout << ", P = " << P 
      << ", Pt = " << Pt 
      << " DPt = " << DPt 
      << ", N(hits) = " << NHit << " (Lost : " << LostHits << "/" << NLostHit << ")"
      << ", Algo = " << Algo
      << std::std::endl;
    */
    return false;
  }
  /*
    std::cout << "Track Accepted : ";
    std::cout << ", P = " << P 
    << ", Pt = " << Pt 
    << " DPt = " << DPt 
    << ", N(hits) = " << NHit << " (Lost : " << LostHits << "/" << NLostHit << ")"
    << ", Algo = " << Algo
    << std::std::endl;
  */
  return true;
}


DEFINE_FWK_MODULE(HGCalTrackCollectionProducer);
