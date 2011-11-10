#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

EcalClusterLazyTools::EcalClusterLazyTools( const edm::Event &ev, const edm::EventSetup &es, edm::InputTag redEBRecHits, edm::InputTag redEERecHits )
{
        getGeometry( es );
        getTopology( es );
        getEBRecHits( ev, redEBRecHits );
        getEERecHits( ev, redEERecHits );
	getIntercalibConstants( es );
	getADCToGeV ( es );
	getLaserDbService ( es );
}



EcalClusterLazyTools::~EcalClusterLazyTools()
{
}



void EcalClusterLazyTools::getGeometry( const edm::EventSetup &es )
{
        edm::ESHandle<CaloGeometry> pGeometry;
        es.get<CaloGeometryRecord>().get(pGeometry);
        geometry_ = pGeometry.product();
}



void EcalClusterLazyTools::getTopology( const edm::EventSetup &es )
{
        edm::ESHandle<CaloTopology> pTopology;
        es.get<CaloTopologyRecord>().get(pTopology);
        topology_ = pTopology.product();
}



void EcalClusterLazyTools::getEBRecHits( const edm::Event &ev, edm::InputTag redEBRecHits )
{
        edm::Handle< EcalRecHitCollection > pEBRecHits;
        ev.getByLabel( redEBRecHits, pEBRecHits );
        ebRecHits_ = pEBRecHits.product();
}



void EcalClusterLazyTools::getEERecHits( const edm::Event &ev, edm::InputTag redEERecHits )
{
        edm::Handle< EcalRecHitCollection > pEERecHits;
        ev.getByLabel( redEERecHits, pEERecHits );
        eeRecHits_ = pEERecHits.product();
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


//float EcalClusterLazyTools::e5x1( const reco::BasicCluster &cluster )
//{
  //return EcalClusterTools::e5x1( cluster, getEcalRecHitCollection(cluster), topology_ );
	//}


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
  std::cout << "BC has this many components: " << clusterComponents.size() << std::endl; // GF debug
  
  const EcalRecHitCollection *recHits = getEcalRecHitCollection( cluster );
  std::cout << "BasicClusterClusterTime - rechits are this many: " << recHits->size() << std::endl; // GF debug
  
  
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
