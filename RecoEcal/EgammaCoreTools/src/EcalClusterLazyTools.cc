#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

EcalClusterLazyTools::EcalClusterLazyTools( const edm::Event &ev, const edm::EventSetup &es, edm::InputTag redEBRecHits, edm::InputTag redEERecHits )
{
        getGeometry( es );
        getTopology( es );
        getEBRecHits( ev, redEBRecHits );
        getEERecHits( ev, redEERecHits );
}



EcalClusterLazyTools::~EcalClusterLazyTools()
{
}



void EcalClusterLazyTools::getGeometry( const edm::EventSetup &es )
{
        edm::ESHandle<CaloGeometry> pGeometry;
        es.get<IdealGeometryRecord>().get(pGeometry);
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



const EcalRecHitCollection * EcalClusterLazyTools::getEcalRecHitCollection( const reco::BasicCluster &cluster )
{
        if ( cluster.size() == 0 ) {
                throw cms::Exception("InvalidCluster") << "The cluster has no crystals!";
        }
        DetId id = cluster.getHitsByDetId()[0]; // size is by definition > 0 -- FIXME??
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
