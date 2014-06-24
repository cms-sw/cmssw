// -*- C++ -*-
//
// Package:    METProducers
// Class:      MuonTCMETValueMapProducer
// 
// Original Author:  Frank Golf
//         Created:  Sun Mar 15 11:33:20 CDT 2009
//
//

//____________________________________________________________________________||
#include "RecoMET/METProducers/interface/MuonTCMETValueMapProducer.h"

#include "RecoMET/METAlgorithms/interface/TCMETAlgo.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h" 
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "TVector3.h"
#include "TMath.h"

//____________________________________________________________________________||
typedef math::XYZPoint Point;

//____________________________________________________________________________||
namespace cms
{

MuonTCMETValueMapProducer::MuonTCMETValueMapProducer(const edm::ParameterSet& iConfig)
{
  produces<edm::ValueMap<reco::MuonMETCorrectionData> > ("muCorrData");

  rfType_     = iConfig.getParameter<int>("rf_type");

  nLayers_                = iConfig.getParameter<int>      ("nLayers");
  nLayersTight_           = iConfig.getParameter<int>      ("nLayersTight");
  vertexNdof_             = iConfig.getParameter<int>      ("vertexNdof");
  vertexZ_                = iConfig.getParameter<double>   ("vertexZ");
  vertexRho_              = iConfig.getParameter<double>   ("vertexRho");
  vertexMaxDZ_            = iConfig.getParameter<double>   ("vertexMaxDZ");
  maxpt_eta20_            = iConfig.getParameter<double>   ("maxpt_eta20");
  maxpt_eta25_            = iConfig.getParameter<double>   ("maxpt_eta25");

  // get configuration parameters
  maxTrackAlgo_    = iConfig.getParameter<int>("trackAlgo_max");
  maxd0cut_        = iConfig.getParameter<double>("d0_max"       );
  minpt_           = iConfig.getParameter<double>("pt_min"       );
  maxpt_           = iConfig.getParameter<double>("pt_max"       );
  maxeta_          = iConfig.getParameter<double>("eta_max"      );
  maxchi2_         = iConfig.getParameter<double>("chi2_max"     );
  minhits_         = iConfig.getParameter<double>("nhits_min"    );
  maxPtErr_        = iConfig.getParameter<double>("ptErr_max"    );

  trkQuality_      = iConfig.getParameter<std::vector<int> >("track_quality");
  trkAlgos_        = iConfig.getParameter<std::vector<int> >("track_algos"  );
  maxchi2_tight_   = iConfig.getParameter<double>("chi2_max_tight");
  minhits_tight_   = iConfig.getParameter<double>("nhits_min_tight");
  maxPtErr_tight_  = iConfig.getParameter<double>("ptErr_max_tight");
  usePvtxd0_       = iConfig.getParameter<bool>("usePvtxd0");
  d0cuta_          = iConfig.getParameter<double>("d0cuta");
  d0cutb_          = iConfig.getParameter<double>("d0cutb");

  muon_dptrel_  = iConfig.getParameter<double>("muon_dptrel");
  muond0_     = iConfig.getParameter<double>("d0_muon"    );
  muonpt_     = iConfig.getParameter<double>("pt_muon"    );
  muoneta_    = iConfig.getParameter<double>("eta_muon"   );
  muonchi2_   = iConfig.getParameter<double>("chi2_muon"  );
  muonhits_   = iConfig.getParameter<double>("nhits_muon" );
  muonGlobal_   = iConfig.getParameter<bool>("global_muon");
  muonTracker_  = iConfig.getParameter<bool>("tracker_muon");
  muonDeltaR_ = iConfig.getParameter<double>("deltaR_muon");
  useCaloMuons_ = iConfig.getParameter<bool>("useCaloMuons");
  muonMinValidStaHits_ = iConfig.getParameter<int>("muonMinValidStaHits");

  response_function = 0;
  tcmetAlgo_=new TCMETAlgo();

  if( rfType_ == 1 )
    response_function = tcmetAlgo_->getResponseFunction_fit();
  else if( rfType_ == 2 )
    response_function = tcmetAlgo_->getResponseFunction_mode();

  muonToken_ = consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muonInputTag"));
  beamSpotToken_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotInputTag"));
  vertexToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexInputTag"));

}

//____________________________________________________________________________||
MuonTCMETValueMapProducer::~MuonTCMETValueMapProducer()
{
    delete tcmetAlgo_;
}

//____________________________________________________________________________||
void MuonTCMETValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  iEvent.getByToken(muonToken_ , muons_);
  iEvent.getByToken(beamSpotToken_, beamSpot_);

  hasValidVertex = false;
  if( usePvtxd0_ ){
    iEvent.getByToken(vertexToken_  ,vertexHandle_);

    if( vertexHandle_.isValid() ) {
      vertices_ = vertexHandle_.product();
      hasValidVertex = isValidVertex();
    }
  }

  edm::ESHandle<MagneticField> theMagField;
  iSetup.get<IdealMagneticFieldRecord>().get(theMagField);
  bField = theMagField.product();

  std::auto_ptr<edm::ValueMap<reco::MuonMETCorrectionData> > vm_muCorrData(new edm::ValueMap<reco::MuonMETCorrectionData>());

  std::vector<reco::MuonMETCorrectionData> v_muCorrData;

  unsigned int nMuons = muons_->size();

  for (unsigned int iMu = 0; iMu < nMuons; iMu++) {

    const reco::Muon* mu = &(*muons_)[iMu];
    double deltax = 0.0;
    double deltay = 0.0;

    reco::MuonMETCorrectionData muMETCorrData(reco::MuonMETCorrectionData::NotUsed, deltax, deltay);
    
    reco::TrackRef mu_track;
    if( mu->isGlobalMuon() || mu->isTrackerMuon() || mu->isCaloMuon() )
      mu_track = mu->innerTrack();
    else {
      v_muCorrData.push_back( muMETCorrData );
      continue;
    }

    // figure out depositions muons would make if they were treated as pions
    if( isGoodTrack( mu ) ) {

      if( mu_track->pt() < minpt_ )
	muMETCorrData = reco::MuonMETCorrectionData(reco::MuonMETCorrectionData::TreatedAsPion, deltax, deltay);

      else {
	int bin_index   = response_function->FindBin( mu_track->eta(), mu_track->pt() );
	double response = response_function->GetBinContent( bin_index );

	TVector3 outerTrkPosition = propagateTrack( mu );

	deltax = response * mu_track->p() * sin( outerTrkPosition.Theta() ) * cos( outerTrkPosition.Phi() );
	deltay = response * mu_track->p() * sin( outerTrkPosition.Theta() ) * sin( outerTrkPosition.Phi() );

	muMETCorrData = reco::MuonMETCorrectionData(reco::MuonMETCorrectionData::TreatedAsPion, deltax, deltay);
      }
    }

    // figure out muon flag
    if( isGoodMuon( mu ) )
      v_muCorrData.push_back( reco::MuonMETCorrectionData(reco::MuonMETCorrectionData::MuonCandidateValuesUsed, deltax, deltay) );

    else if( useCaloMuons_ && isGoodCaloMuon( mu, iMu ) )
      v_muCorrData.push_back( reco::MuonMETCorrectionData(reco::MuonMETCorrectionData::MuonCandidateValuesUsed, deltax, deltay) );

    else v_muCorrData.push_back( muMETCorrData );
  }
    
  edm::ValueMap<reco::MuonMETCorrectionData>::Filler dataFiller(*vm_muCorrData);

  dataFiller.insert( muons_, v_muCorrData.begin(), v_muCorrData.end());
  dataFiller.fill();
    
  iEvent.put(vm_muCorrData, "muCorrData");
}
  
//____________________________________________________________________________||
bool MuonTCMETValueMapProducer::isGoodMuon( const reco::Muon* muon )
{
  double d0    = -999;
  double nhits = 0;
  double chi2  = 999;

  // get d0 corrected for beam spot
  bool haveBeamSpot = true;
  if( !beamSpot_.isValid() ) haveBeamSpot = false;

  if( muonGlobal_  && !muon->isGlobalMuon()  ) return false;
  if( muonTracker_ && !muon->isTrackerMuon() ) return false;

  const reco::TrackRef siTrack     = muon->innerTrack();
  const reco::TrackRef globalTrack = muon->globalTrack();

  Point bspot = haveBeamSpot ? beamSpot_->position() : Point(0,0,0);
  if( siTrack.isNonnull() ) nhits = siTrack->numberOfValidHits();
  if( globalTrack.isNonnull() ) 
    {
      d0   = -1 * globalTrack->dxy( bspot );
      chi2 = globalTrack->normalizedChi2();
    }

  if( fabs( d0 ) > muond0_ )                          return false;
  if( muon->pt() < muonpt_ )                          return false;
  if( fabs( muon->eta() ) > muoneta_ )                return false;
  if( nhits < muonhits_ )                             return false;
  if( chi2 > muonchi2_ )                              return false;
  if( globalTrack->hitPattern().numberOfValidMuonHits() < muonMinValidStaHits_ ) return false;

  //reject muons with tracker dpt/pt > X
  if( !siTrack.isNonnull() )                                return false;
  if( siTrack->ptError() / siTrack->pt() > muon_dptrel_ )   return false;

  else return true;
}

//____________________________________________________________________________||
bool MuonTCMETValueMapProducer::isGoodCaloMuon( const reco::Muon* muon, const unsigned int index )
{

  if( muon->pt() < 10 ) return false;

  if( !isGoodTrack( muon ) ) return false;

  const reco::TrackRef inputSiliconTrack = muon->innerTrack();
  if( !inputSiliconTrack.isNonnull() ) return false;

  //check if it is in the vicinity of a global or tracker muon
  unsigned int nMuons = muons_->size();
  for (unsigned int iMu = 0; iMu < nMuons; iMu++)
    {

      if( iMu == index ) continue;

      const reco::Muon* mu = &(*muons_)[iMu];

      const reco::TrackRef testSiliconTrack = mu->innerTrack();
      if( !testSiliconTrack.isNonnull() ) continue;

      double deltaEta = inputSiliconTrack.get()->eta() - testSiliconTrack.get()->eta();
      double deltaPhi = acos( cos( inputSiliconTrack.get()->phi() - testSiliconTrack.get()->phi() ) );
      double deltaR   = TMath::Sqrt( deltaEta * deltaEta + deltaPhi * deltaPhi );

      if( deltaR < muonDeltaR_ ) return false;
    }

  return true;
}

//____________________________________________________________________________||
bool MuonTCMETValueMapProducer::isGoodTrack( const reco::Muon* muon )
{
  double d0    = -999;

  const reco::TrackRef siTrack = muon->innerTrack();
  if (!siTrack.isNonnull())
    return false;

  if( hasValidVertex )
    {
      //get d0 corrected for primary vertex
            
      const Point pvtx = Point(vertices_->begin()->x(),
			       vertices_->begin()->y(), 
			       vertices_->begin()->z());
            
      d0 = -1 * siTrack->dxy( pvtx );
            
      double dz = siTrack->dz( pvtx );
            
      if( fabs( dz ) < vertexMaxDZ_ ){
              
	//get d0 corrected for pvtx
	d0 = -1 * siTrack->dxy( pvtx );
              
      }else{
              
	// get d0 corrected for beam spot
	bool haveBeamSpot = true;
	if( !beamSpot_.isValid() ) haveBeamSpot = false;
              
	Point bspot = haveBeamSpot ? beamSpot_->position() : Point(0,0,0);
	d0 = -1 * siTrack->dxy( bspot );
              
      }
    }
  else
    {
       
      // get d0 corrected for beam spot
      bool haveBeamSpot = true;
      if( !beamSpot_.isValid() ) haveBeamSpot = false;
       
      Point bspot = haveBeamSpot ? beamSpot_->position() : Point(0,0,0);
      d0 = -1 * siTrack->dxy( bspot );
    }

  if( siTrack->algo() < maxTrackAlgo_ )
    {
      //1st 4 tracking iterations (pT-dependent d0 cut)
       
      float d0cut = sqrt(std::pow(d0cuta_,2) + std::pow(d0cutb_/siTrack->pt(),2)); 
      if(d0cut > maxd0cut_) d0cut = maxd0cut_;
       
      if( fabs( d0 ) > d0cut )            return false;
      if( nLayers( siTrack ) < nLayers_ ) return false;
    }
  else
    {
      //last 2 tracking iterations (tighten chi2, nhits, pt error cuts)
     
      if( siTrack->normalizedChi2() > maxchi2_tight_ )               return false;
      if( siTrack->numberOfValidHits() < minhits_tight_ )            return false;
      if( (siTrack->ptError() / siTrack->pt()) > maxPtErr_tight_ )   return false;
      if( nLayers( siTrack ) < nLayersTight_ )                       return false;
    }

  if( siTrack->numberOfValidHits() < minhits_ )                         return false;
  if( siTrack->normalizedChi2() > maxchi2_ )                            return false;
  if( fabs( siTrack->eta() ) > maxeta_ )                                return false;
  if( siTrack->pt() > maxpt_ )                                          return false;
  if( (siTrack->ptError() / siTrack->pt()) > maxPtErr_ )                return false;
  if( fabs( siTrack->eta() ) > 2.5 && siTrack->pt() > maxpt_eta25_ )    return false;
  if( fabs( siTrack->eta() ) > 2.0 && siTrack->pt() > maxpt_eta20_ )    return false;

  int cut = 0;
  for( unsigned int i = 0; i < trkQuality_.size(); i++ )
    {
      cut |= (1 << trkQuality_.at(i));
    }

  if( !( (siTrack->qualityMask() & cut) == cut ) ) return false;
	  
  bool isGoodAlgo = false;
  if( trkAlgos_.size() == 0 ) isGoodAlgo = true;
  for( unsigned int i = 0; i < trkAlgos_.size(); i++ )
    {
      if( siTrack->algo() == trkAlgos_.at(i) ) isGoodAlgo = true;
    }

  if( !isGoodAlgo ) return false;
	  
  return true;
}

//____________________________________________________________________________||
TVector3 MuonTCMETValueMapProducer::propagateTrack( const reco::Muon* muon)
{

  TVector3 outerTrkPosition;

  outerTrkPosition.SetPtEtaPhi( 999., -10., 2 * TMath::Pi() );

  const reco::TrackRef track = muon->innerTrack();

  if( !track.isNonnull() )
    {
      return outerTrkPosition;
    }

  GlobalPoint  tpVertex ( track->vx(), track->vy(), track->vz() );
  GlobalVector tpMomentum ( track.get()->px(), track.get()->py(), track.get()->pz() );
  int tpCharge ( track->charge() );

  FreeTrajectoryState fts ( tpVertex, tpMomentum, tpCharge, bField);

  const double zdist = 314.;

  const double radius = 130.;

  const double corner = 1.479;

  Plane::PlanePointer lendcap = Plane::build( Plane::PositionType (0, 0, -zdist), Plane::RotationType () );
  Plane::PlanePointer rendcap = Plane::build( Plane::PositionType (0, 0, zdist),  Plane::RotationType () );

  Cylinder::CylinderPointer barrel = Cylinder::build( Cylinder::PositionType (0, 0, 0), Cylinder::RotationType (), radius);

  AnalyticalPropagator myAP (bField, alongMomentum, 2*M_PI);

  TrajectoryStateOnSurface tsos;

  if( track.get()->eta() < -corner )
    {
      tsos = myAP.propagate( fts, *lendcap);
    }
  else if( fabs(track.get()->eta()) < corner )
    {
      tsos = myAP.propagate( fts, *barrel);
    }
  else if( track.get()->eta() > corner )
    {
      tsos = myAP.propagate( fts, *rendcap);
    }

  if( tsos.isValid() )
    outerTrkPosition.SetXYZ( tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z() );

  else
    outerTrkPosition.SetPtEtaPhi( 999., -10., 2 * TMath::Pi() );

  return outerTrkPosition;
}


//____________________________________________________________________________||
int MuonTCMETValueMapProducer::nLayers(const reco::TrackRef track)
{
  const reco::HitPattern& p = track->hitPattern();
  return p.trackerLayersWithMeasurement();
}

//____________________________________________________________________________||
bool MuonTCMETValueMapProducer::isValidVertex()
{
  if( vertices_->begin()->isFake()                ) return false;
  if( vertices_->begin()->ndof() < vertexNdof_    ) return false;
  if( fabs( vertices_->begin()->z() ) > vertexZ_  ) return false;
  if( sqrt( std::pow( vertices_->begin()->x() , 2 ) + std::pow( vertices_->begin()->y() , 2 ) ) > vertexRho_ ) return false;
    
  return true;
    
}

}

//____________________________________________________________________________||
