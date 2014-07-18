#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"

#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

EcalClusterLazyTools::EcalClusterLazyTools( const edm::Event &ev, const edm::EventSetup &es, edm::EDGetTokenT<EcalRecHitCollection> token1, edm::EDGetTokenT<EcalRecHitCollection> token2) {

  ebRHToken_ = token1;
  eeRHToken_ = token2;

  getGeometry( es );
  getTopology( es );
  getEBRecHits( ev );
  getEERecHits( ev );
  getIntercalibConstants( es );
  getADCToGeV ( es );
  getLaserDbService ( es );
}

EcalClusterLazyTools::~EcalClusterLazyTools()
{}

void EcalClusterLazyTools::getGeometry( const edm::EventSetup &es ) {
        edm::ESHandle<CaloGeometry> pGeometry;
        es.get<CaloGeometryRecord>().get(pGeometry);
        geometry_ = pGeometry.product();
}

void EcalClusterLazyTools::getTopology( const edm::EventSetup &es ) {
        edm::ESHandle<CaloTopology> pTopology;
        es.get<CaloTopologyRecord>().get(pTopology);
        topology_ = pTopology.product();
}

void EcalClusterLazyTools::getEBRecHits( const edm::Event &ev ) {
  edm::Handle< EcalRecHitCollection > pEBRecHits;
  ev.getByToken( ebRHToken_, pEBRecHits );
  ebRecHits_ = pEBRecHits.product();
}

void EcalClusterLazyTools::getEERecHits( const edm::Event &ev ) {
  edm::Handle< EcalRecHitCollection > pEERecHits;
  ev.getByToken( eeRHToken_, pEERecHits );
  eeRecHits_ = pEERecHits.product();
}

void EcalClusterLazyTools::getESRecHits( const edm::Event &ev ) {
  edm::Handle< EcalRecHitCollection > pESRecHits;
  ev.getByToken( esRHToken_, pESRecHits );
  esRecHits_ = pESRecHits.product();
  // make the map of rechits
  rechits_map_.clear();
  if (pESRecHits.isValid()) {
    EcalRecHitCollection::const_iterator it;
    for (it = pESRecHits->begin(); it != pESRecHits->end(); ++it) {
      // remove bad ES rechits
  	std::vector<int> badf = {
  	  EcalRecHit::ESFlags::kESDead, // 1
  	  EcalRecHit::ESFlags::kESTwoGoodRatios,
  	  EcalRecHit::ESFlags::kESBadRatioFor12, // 5
  	  EcalRecHit::ESFlags::kESBadRatioFor23Upper,
  	  EcalRecHit::ESFlags::kESBadRatioFor23Lower,
  	  EcalRecHit::ESFlags::kESTS1Largest,
  	  EcalRecHit::ESFlags::kESTS3Largest,
  	  EcalRecHit::ESFlags::kESTS3Negative, // 10
  	  EcalRecHit::ESFlags::kESTS13Sigmas, // 14
  	};
  	
  	if (it->checkFlags(badf)) continue;

      //Make the map of DetID, EcalRecHit pairs
      rechits_map_.insert(std::make_pair(it->id(), *it));
    }
  }
}



void EcalClusterLazyTools::getIntercalibConstants( const edm::EventSetup &es )
{
  // get IC's
  es.get<EcalIntercalibConstantsRcd>().get(ical);
  icalMap = ical->getMap();
}



void EcalClusterLazyTools::getADCToGeV( const edm::EventSetup &es )
{
  // get ADCtoGeV
  es.get<EcalADCToGeVConstantRcd>().get(agc);
}



void EcalClusterLazyTools::getLaserDbService     ( const edm::EventSetup &es ){
  // transp corrections
  es.get<EcalLaserDbRecord>().get(laser);
}


const EcalRecHitCollection * EcalClusterLazyTools::getEcalRecHitCollection( const reco::BasicCluster &cluster )
{
        if ( cluster.size() == 0 ) {
                throw cms::Exception("InvalidCluster") << "The cluster has no crystals!";
        }
        DetId id = (cluster.hitsAndFractions()[0]).first; // size is by definition > 0 -- FIXME??
        const EcalRecHitCollection *recHits = 0;
        if ( id.subdetId() == EcalBarrel ) {
                recHits = ebRecHits_;
        } else if ( id.subdetId() == EcalEndcap ) {
                recHits = eeRecHits_;
        } else {
                throw cms::Exception("InvalidSubdetector") << "The subdetId() " << id.subdetId() << " does not correspond to EcalBarrel neither EcalEndcap";
        }
        return recHits;
}



float EcalClusterLazyTools::e1x3( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e1x3( cluster, getEcalRecHitCollection(cluster), topology_ );
}


float EcalClusterLazyTools::e3x1( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e3x1( cluster, getEcalRecHitCollection(cluster), topology_ );
}


float EcalClusterLazyTools::e1x5( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e1x5( cluster, getEcalRecHitCollection(cluster), topology_ );
}


float EcalClusterLazyTools::e5x1( const reco::BasicCluster &cluster )
{
  return EcalClusterTools::e5x1( cluster, getEcalRecHitCollection(cluster), topology_ );
	}


float EcalClusterLazyTools::e2x2( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e2x2( cluster, getEcalRecHitCollection(cluster), topology_ );
}


float EcalClusterLazyTools::e3x2( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e3x2( cluster, getEcalRecHitCollection(cluster), topology_ );
}


float EcalClusterLazyTools::e3x3( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e3x3( cluster, getEcalRecHitCollection(cluster), topology_ );
}


float EcalClusterLazyTools::e4x4( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e4x4( cluster, getEcalRecHitCollection(cluster), topology_ );
}



float EcalClusterLazyTools::e5x5( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e5x5( cluster, getEcalRecHitCollection(cluster), topology_ );
}


float EcalClusterLazyTools::e2x5Right( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e2x5Right( cluster, getEcalRecHitCollection(cluster), topology_ );
}



float EcalClusterLazyTools::e2x5Left( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e2x5Left( cluster, getEcalRecHitCollection(cluster), topology_ );
}



float EcalClusterLazyTools::e2x5Top( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e2x5Top( cluster, getEcalRecHitCollection(cluster), topology_ );
}



float EcalClusterLazyTools::e2x5Bottom( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e2x5Bottom( cluster, getEcalRecHitCollection(cluster), topology_ );
}


// Energy in 2x5 strip containing the max crystal.
float EcalClusterLazyTools::e2x5Max( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e2x5Max( cluster, getEcalRecHitCollection(cluster), topology_ );
}


float EcalClusterLazyTools::eLeft( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::eLeft( cluster, getEcalRecHitCollection(cluster), topology_ );
}


float EcalClusterLazyTools::eRight( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::eRight( cluster, getEcalRecHitCollection(cluster), topology_ );
}


float EcalClusterLazyTools::eTop( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::eTop( cluster, getEcalRecHitCollection(cluster), topology_ );
}



float EcalClusterLazyTools::eBottom( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::eBottom( cluster, getEcalRecHitCollection(cluster), topology_ );
}


float EcalClusterLazyTools::eMax( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::eMax( cluster, getEcalRecHitCollection(cluster) );
}


float EcalClusterLazyTools::e2nd( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::e2nd( cluster, getEcalRecHitCollection(cluster) );
}


std::pair<DetId, float> EcalClusterLazyTools::getMaximum( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::getMaximum( cluster, getEcalRecHitCollection(cluster) );
}


std::vector<float> EcalClusterLazyTools::energyBasketFractionEta( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::energyBasketFractionEta( cluster, getEcalRecHitCollection(cluster) );
}



std::vector<float> EcalClusterLazyTools::energyBasketFractionPhi( const reco::BasicCluster &cluster )
{
        return EcalClusterTools::energyBasketFractionPhi( cluster, getEcalRecHitCollection(cluster) );
}


std::vector<float> EcalClusterLazyTools::lat( const reco::BasicCluster &cluster, bool logW, float w0 )
{
        return EcalClusterTools::lat( cluster, getEcalRecHitCollection(cluster), geometry_, logW, w0 );
}

//std::vector<float> EcalClusterLazyTools::lat( const reco::BasicCluster &cluster, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv, bool logW, float w0 )
//{
//        return EcalClusterTools::lat( cluster, getEcalRecHitCollection(cluster), geometry_,flagsexcl, severitiesexcl, sevLv, logW, w0 );
//} 


std::vector<float> EcalClusterLazyTools::covariances(const reco::BasicCluster &cluster, float w0 )
{
        return EcalClusterTools::covariances( cluster, getEcalRecHitCollection(cluster), topology_, geometry_, w0 );
}


std::vector<float> EcalClusterLazyTools::localCovariances(const reco::BasicCluster &cluster, float w0 )
{
        return EcalClusterTools::localCovariances( cluster, getEcalRecHitCollection(cluster), topology_, w0 );
}


std::vector<float> EcalClusterLazyTools::scLocalCovariances(const reco::SuperCluster &cluster, float w0 )
{
        return EcalClusterTools::scLocalCovariances( cluster, getEcalRecHitCollection(cluster), topology_, w0 );
}


double EcalClusterLazyTools::zernike20( const reco::BasicCluster &cluster, double R0, bool logW, float w0 )
{
        return EcalClusterTools::zernike20( cluster, getEcalRecHitCollection(cluster), geometry_, R0, logW, w0 );
}


double EcalClusterLazyTools::zernike42( const reco::BasicCluster &cluster, double R0, bool logW, float w0 )
{
        return EcalClusterTools::zernike42( cluster, getEcalRecHitCollection(cluster), geometry_, R0, logW, w0 );
}

std::vector<DetId> EcalClusterLazyTools::matrixDetId( DetId id, int ixMin, int ixMax, int iyMin, int iyMax )
{
        return EcalClusterTools::matrixDetId( topology_, id, ixMin, ixMax, iyMin, iyMax );
}

float EcalClusterLazyTools::matrixEnergy( const reco::BasicCluster &cluster, DetId id, int ixMin, int ixMax, int iyMin, int iyMax )
{
  return EcalClusterTools::matrixEnergy( cluster, getEcalRecHitCollection(cluster), topology_, id, ixMin, ixMax, iyMin, iyMax );
}


// get time of basic cluster seed crystal 
float EcalClusterLazyTools::BasicClusterSeedTime(const reco::BasicCluster &cluster)
{
  
  const EcalRecHitCollection *recHits = getEcalRecHitCollection( cluster );
  
  DetId id = cluster.seed();
  EcalRecHitCollection::const_iterator theSeedHit = recHits->find (id);
  //  std::cout << "the seed of the BC has time: " 
  //<< (*theSeedHit).time() 
  //<< "and energy: " << (*theSeedHit).energy() << " collection size: " << recHits->size() 
  //<< "\n" <<std::endl; // GF debug
  
  return (*theSeedHit).time();
}


// error-weighted average of time from constituents of basic cluster 
float EcalClusterLazyTools::BasicClusterTime(const reco::BasicCluster &cluster, const edm::Event &ev)
{
  
  std::vector<std::pair<DetId, float> > clusterComponents = (cluster).hitsAndFractions() ;
  //std::cout << "BC has this many components: " << clusterComponents.size() << std::endl; // GF debug
  
  const EcalRecHitCollection *recHits = getEcalRecHitCollection( cluster );
  //std::cout << "BasicClusterClusterTime - rechits are this many: " << recHits->size() << std::endl; // GF debug
  
  
  float weightedTsum   = 0;
  float sumOfWeights   = 0;
  
  for (std::vector<std::pair<DetId, float> >::const_iterator detitr = clusterComponents.begin(); detitr != clusterComponents.end(); detitr++ )
    {
      //      EcalRecHitCollection::const_iterator theSeedHit = recHits->find (id); // trash this
      EcalRecHitCollection::const_iterator oneHit = recHits->find( (detitr -> first) ) ;
      
      // in order to get back the ADC counts from the recHit energy, three ingredients are necessary:
      // 1) get laser correction coefficient
      float lasercalib = 1.;
      lasercalib = laser->getLaserCorrection( detitr->first, ev.time());
      // 2) get intercalibration
      EcalIntercalibConstantMap::const_iterator icalit = icalMap.find(detitr->first);
      EcalIntercalibConstant icalconst = 1.;
      if( icalit!=icalMap.end() ) {
	icalconst = (*icalit);
	// std::cout << "icalconst set to: " << icalconst << std::endl;
      } else {
	edm::LogError("EcalClusterLazyTools") << "No intercalib const found for xtal "  << (detitr->first).rawId() << "bailing out";
	assert(0);
      }
      // 3) get adc2GeV
      float adcToGeV = 1.;
      if       ( (detitr -> first).subdetId() == EcalBarrel )  adcToGeV = float(agc->getEBValue());
      else if  ( (detitr -> first).subdetId() == EcalEndcap )  adcToGeV = float(agc->getEEValue());
            float adc = 2.;
      if (icalconst>0 && lasercalib>0 && adcToGeV>0)  adc= (*oneHit).energy()/(icalconst*lasercalib*adcToGeV);

      // don't consider recHits with too little amplitude; take sigma_noise_total into account
      if( (detitr -> first).subdetId() == EcalBarrel  &&  adc< (1.1*20) ) continue;
      if( (detitr -> first).subdetId() == EcalEndcap  &&  adc< (2.2*20) ) continue;

      // count only on rechits whose error is trusted by the method (ratio)
      if(! (*oneHit).isTimeErrorValid()) continue;

      float timeError    = (*oneHit).timeError();
      // the constant used to build timeError is largely over-estimated ; remove in quadrature 0.6 and add 0.15 back.
      // could be prettier if value of constant term was changed at recHit production level
      if (timeError>0.6) timeError = sqrt( timeError*timeError - 0.6*0.6 + 0.15*0.15);
      else               timeError = sqrt( timeError*timeError           + 0.15*0.15);

      // do the error weighting
      weightedTsum += (*oneHit).time() / (timeError*timeError);
      sumOfWeights += 1. / (timeError*timeError);

    }
  
  // what if no crytal is available for weighted average?
  if     ( sumOfWeights ==0 )  return -999;
  else   return ( weightedTsum / sumOfWeights);

}


// get BasicClusterSeedTime of the seed basic cluser of the supercluster
float EcalClusterLazyTools::SuperClusterSeedTime(const reco::SuperCluster &cluster){

  return BasicClusterSeedTime ( (*cluster.seed()) );

}


// get BasicClusterTime of the seed basic cluser of the supercluster
float EcalClusterLazyTools::SuperClusterTime(const reco::SuperCluster &cluster, const edm::Event &ev){
  
  return BasicClusterTime ( (*cluster.seed()) , ev);

}


// get Preshower effective sigmaIRIR
float EcalClusterLazyTools::eseffsirir(const reco::SuperCluster &cluster)
{
  if (!(fabs(cluster.eta()) > 1.6 && fabs(cluster.eta()) < 3.)) return 0.;

  const CaloSubdetectorGeometry *geometryES = geometry_->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  CaloSubdetectorTopology *topology_p = 0;
  if (geometryES) topology_p = new EcalPreshowerTopology(geometry_);

  std::vector<float> phoESHitsIXIX = getESHits(cluster.x(), cluster.y(), cluster.z(), rechits_map_, geometry_, topology_p, 0, 1);
  std::vector<float> phoESHitsIYIY = getESHits(cluster.x(), cluster.y(), cluster.z(), rechits_map_, geometry_, topology_p, 0, 2);
  float phoESShapeIXIX = getESShape(phoESHitsIXIX);
  float phoESShapeIYIY = getESShape(phoESHitsIYIY);

  return sqrt(phoESShapeIXIX*phoESShapeIXIX + phoESShapeIYIY*phoESShapeIYIY);
}

// get Preshower effective sigmaIXIX
float EcalClusterLazyTools::eseffsixix(const reco::SuperCluster &cluster)
{
  if (!(fabs(cluster.eta()) > 1.6 && fabs(cluster.eta()) < 3.)) return 0.;

  const CaloSubdetectorGeometry *geometryES = geometry_->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  CaloSubdetectorTopology *topology_p = 0;
  if (geometryES) topology_p = new EcalPreshowerTopology(geometry_);

  std::vector<float> phoESHitsIXIX = getESHits(cluster.x(), cluster.y(), cluster.z(), rechits_map_, geometry_, topology_p, 0, 1);
  float phoESShapeIXIX = getESShape(phoESHitsIXIX);

  return phoESShapeIXIX;
}

// get Preshower effective sigmaIYIY
float EcalClusterLazyTools::eseffsiyiy(const reco::SuperCluster &cluster)
{
  if (!(fabs(cluster.eta()) > 1.6 && fabs(cluster.eta()) < 3.)) return 0.;

  const CaloSubdetectorGeometry *geometryES = geometry_->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  CaloSubdetectorTopology *topology_p = 0;
  if (geometryES) topology_p = new EcalPreshowerTopology(geometry_);

  std::vector<float> phoESHitsIYIY = getESHits(cluster.x(), cluster.y(), cluster.z(), rechits_map_, geometry_, topology_p, 0, 2);
  float phoESShapeIYIY = getESShape(phoESHitsIYIY);

  return phoESShapeIYIY;
}

// get Preshower Rechits
std::vector<float> EcalClusterLazyTools::getESHits(double X, double Y, double Z, const std::map<DetId, EcalRecHit>& _rechits_map, const CaloGeometry* geometry, CaloSubdetectorTopology *topology_p, int row, int plane) 
{
  std::map<DetId, EcalRecHit> rechits_map = _rechits_map;
  std::vector<float> esHits;

  const GlobalPoint point(X,Y,Z);

  const CaloSubdetectorGeometry *geometry_p ;
  geometry_p = geometry->getSubdetectorGeometry (DetId::Ecal,EcalPreshower) ;

  DetId esId = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(point, plane);
  ESDetId esDetId = (esId == DetId(0)) ? ESDetId(0) : ESDetId(esId);

  std::map<DetId, EcalRecHit>::iterator it;
  ESDetId next;
  ESDetId strip;
  strip = esDetId;

  EcalPreshowerNavigator theESNav(strip, topology_p);
  theESNav.setHome(strip);

  if (row == 1) {
    if (plane==1 && strip != ESDetId(0)) strip = theESNav.north();
    if (plane==2 && strip != ESDetId(0)) strip = theESNav.east();
  } else if (row == -1) {
    if (plane==1 && strip != ESDetId(0)) strip = theESNav.south();
    if (plane==2 && strip != ESDetId(0)) strip = theESNav.west();
  }


  if (strip == ESDetId(0)) {
    for (int i=0; i<31; ++i) esHits.push_back(0);
  } else {
    it = rechits_map.find(strip);
    if (it->second.energy() > 1.0e-10 && it != rechits_map.end()) esHits.push_back(it->second.energy());
    else esHits.push_back(0);
    //cout<<"center : "<<strip<<" "<<it->second.energy()<<endl;

    // Front Plane
    if (plane==1) {
      // east road
      for (int i=0; i<15; ++i) {
        next = theESNav.east();
        if (next != ESDetId(0)) {
          it = rechits_map.find(next);
          if (it->second.energy() > 1.0e-10 && it != rechits_map.end()) esHits.push_back(it->second.energy());
          else esHits.push_back(0);
          //cout<<"east "<<i<<" : "<<next<<" "<<it->second.energy()<<endl;
        } else {
          for (int j=i; j<15; j++) esHits.push_back(0);
          break;
          //cout<<"east "<<i<<" : "<<next<<" "<<0<<endl;
        }
      }

      // west road
      theESNav.setHome(strip);
      theESNav.home();
      for (int i=0; i<15; ++i) {
        next = theESNav.west();
        if (next != ESDetId(0)) {
          it = rechits_map.find(next);
          if (it->second.energy() > 1.0e-10 && it != rechits_map.end()) esHits.push_back(it->second.energy());
          else esHits.push_back(0);
          //cout<<"west "<<i<<" : "<<next<<" "<<it->second.energy()<<endl;
        } else {
          for (int j=i; j<15; j++) esHits.push_back(0);
          break;
          //cout<<"west "<<i<<" : "<<next<<" "<<0<<endl;
        }
      }
    } // End of Front Plane

    // Rear Plane
    if (plane==2) {
      // north road
      for (int i=0; i<15; ++i) {
        next = theESNav.north();
        if (next != ESDetId(0)) {
          it = rechits_map.find(next);
          if (it->second.energy() > 1.0e-10 && it != rechits_map.end()) esHits.push_back(it->second.energy());
          else esHits.push_back(0);
          //cout<<"north "<<i<<" : "<<next<<" "<<it->second.energy()<<endl;
        } else {
          for (int j=i; j<15; j++) esHits.push_back(0);
          break;
          //cout<<"north "<<i<<" : "<<next<<" "<<0<<endl;
        }
      }

      // south road
      theESNav.setHome(strip);
      theESNav.home();
      for (int i=0; i<15; ++i) {
        next = theESNav.south();
        if (next != ESDetId(0)) {
          it = rechits_map.find(next);
          if (it->second.energy() > 1.0e-10 && it != rechits_map.end()) esHits.push_back(it->second.energy());
          else esHits.push_back(0);
          //cout<<"south "<<i<<" : "<<next<<" "<<it->second.energy()<<endl;
        } else {
          for (int j=i; j<15; j++) esHits.push_back(0);
          break;
          //cout<<"south "<<i<<" : "<<next<<" "<<0<<endl;
        }
      }
    } // End of Rear Plane
  } // Fill ES RecHits

  return esHits;
}


// get Preshower hit shape
float EcalClusterLazyTools::getESShape(const std::vector<float>& ESHits0)
{
  const int nBIN = 21;
  float esRH[nBIN];
  for (int idx=0; idx<nBIN; idx++) {
    esRH[idx] = 0.;
  }

  for(int ibin=0; ibin<((nBIN+1)/2); ibin++) {
    if (ibin==0) {
      esRH[(nBIN-1)/2] = ESHits0[ibin];
    } else {
      esRH[(nBIN-1)/2+ibin] = ESHits0[ibin];
      esRH[(nBIN-1)/2-ibin] = ESHits0[ibin+15];
    }
  }

  // ---- Effective Energy Deposit Width ---- //
  double EffWidthSigmaISIS = 0.;
  double totalEnergyISIS   = 0.;
  double EffStatsISIS      = 0.;
  for (int id_X=0; id_X<21; id_X++) {
    totalEnergyISIS  += esRH[id_X];
    EffStatsISIS     += esRH[id_X]*(id_X-10)*(id_X-10);
  }
  EffWidthSigmaISIS  = (totalEnergyISIS>0.)  ? sqrt(fabs(EffStatsISIS  / totalEnergyISIS))   : 0.;

  return EffWidthSigmaISIS;
}
