#ifndef RecoEcal_EgammaCoreTools_EcalClusterLazyTools_h
#define RecoEcal_EgammaCoreTools_EcalClusterLazyTools_h

/** \class EcalClusterLazyTools
 *  
 * various cluster tools (e.g. cluster shapes)
 *
 * \author Federico Ferri
 * 
 * \version $Id: 
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"


class CaloTopology;
class CaloGeometry;
class CaloSubdetectorTopology;

class EcalClusterLazyToolsBase {
 public:
  EcalClusterLazyToolsBase( const edm::Event &ev, const edm::EventSetup &es, edm::EDGetTokenT<EcalRecHitCollection> token1, edm::EDGetTokenT<EcalRecHitCollection> token2);
  EcalClusterLazyToolsBase( const edm::Event &ev, const edm::EventSetup &es, edm::EDGetTokenT<EcalRecHitCollection> token1, edm::EDGetTokenT<EcalRecHitCollection> token2, edm::EDGetTokenT<EcalRecHitCollection> token3);
  ~EcalClusterLazyToolsBase();
  

  // get time of basic cluster seed crystal 
  float BasicClusterSeedTime(const reco::BasicCluster &cluster);
  // error-weighted average of time from constituents of basic cluster 
  float BasicClusterTime(const reco::BasicCluster &cluster, const edm::Event &ev);
  // get BasicClusterSeedTime of the seed basic cluser of the supercluster
  float SuperClusterSeedTime(const reco::SuperCluster &cluster);
  // get BasicClusterTime of the seed basic cluser of the supercluster
  float SuperClusterTime(const reco::SuperCluster &cluster, const edm::Event &ev);
  
  // mapping for preshower rechits
  std::map<DetId, EcalRecHit> rechits_map_;
  // get Preshower hit array
  std::vector<float> getESHits(double X, double Y, double Z, const std::map<DetId, EcalRecHit>& rechits_map, const CaloGeometry* geometry, CaloSubdetectorTopology *topology_p, int row=0, int plane=1);
  // get Preshower hit shape
  float getESShape(const std::vector<float>& ESHits0);
  // get Preshower effective sigmaRR
  float eseffsirir( const reco::SuperCluster &cluster );
  float eseffsixix( const reco::SuperCluster &cluster );
  float eseffsiyiy( const reco::SuperCluster &cluster );
  
  //  std::vector<int> flagsexcl_;
  //std::vector<int> severitiesexcl_;
  // const EcalSeverityLevelAlgo *sevLv;
  
 protected:
  void getGeometry( const edm::EventSetup &es );
  void getTopology( const edm::EventSetup &es );
  void getEBRecHits( const edm::Event &ev );
  void getEERecHits( const edm::Event &ev );
  void getESRecHits( const edm::Event &ev );
  const EcalRecHitCollection * getEcalRecHitCollection( const reco::BasicCluster &cluster );
  
  const CaloGeometry *geometry_;
  const CaloTopology *topology_;
  const EcalRecHitCollection *ebRecHits_;
  const EcalRecHitCollection *eeRecHits_;
  const EcalRecHitCollection *esRecHits_;
  
  edm::EDGetTokenT<EcalRecHitCollection> ebRHToken_, eeRHToken_, esRHToken_;

  //const EcalIntercalibConstantMap& icalMap;
  edm::ESHandle<EcalIntercalibConstants> ical;
  const EcalIntercalibConstantMap*        icalMap;
  edm::ESHandle<EcalADCToGeVConstant>    agc;
  edm::ESHandle<EcalLaserDbService>      laser;
  void getIntercalibConstants( const edm::EventSetup &es );
  void getADCToGeV           ( const edm::EventSetup &es );
  void getLaserDbService     ( const edm::EventSetup &es );
  
  //  std::vector<int> flagsexcl_;
  //  std::vector<int> severitiesexcl_;

 public:
  inline const EcalRecHitCollection *getEcalEBRecHitCollection(void){return ebRecHits_;};
  inline const EcalRecHitCollection *getEcalEERecHitCollection(void){return eeRecHits_;};
  inline const EcalRecHitCollection *getEcalESRecHitCollection(void){return esRecHits_;};
  inline const EcalIntercalibConstants& getEcalIntercalibConstants(void){return *icalMap;};
  inline const edm::ESHandle<EcalLaserDbService>& getLaserHandle(void){return laser;};
  
}; // class EcalClusterLazyToolsBase

template<class EcalClusterToolsImpl> 
class EcalClusterLazyToolsT : public EcalClusterLazyToolsBase {
    public:

 EcalClusterLazyToolsT( const edm::Event &ev, const edm::EventSetup &es, edm::EDGetTokenT<EcalRecHitCollection> token1, edm::EDGetTokenT<EcalRecHitCollection> token2):
  EcalClusterLazyToolsBase(ev,es,token1,token2) {}

 EcalClusterLazyToolsT( const edm::Event &ev, const edm::EventSetup &es, edm::EDGetTokenT<EcalRecHitCollection> token1, edm::EDGetTokenT<EcalRecHitCollection> token2, edm::EDGetTokenT<EcalRecHitCollection> token3):
  EcalClusterLazyToolsBase(ev,es,token1,token2,token3) {}
        ~EcalClusterLazyToolsT() {}

        // various energies in the matrix nxn surrounding the maximum energy crystal of the input cluster  
        //NOTE (29/10/08): we now use an eta/phi coordinate system rather than phi/eta
        //to minmise possible screwups, for now e5x1 isnt defined all the majority of people who call it actually want e1x5 and 
        //it is thought it is better that their code doesnt compile rather than pick up the wrong function
        //therefore in this version and later e1x5 = e5x1 in the old version 
        //so 1x5 is 1 crystal in eta and 5 crystals in phi
        //note e3x2 does not have a definate eta/phi geometry, it takes the maximum 3x2 block containing the 
        //seed regardless of whether that 3 in eta or phi
        float e1x3( const reco::BasicCluster &cluster );

        float e3x1( const reco::BasicCluster &cluster );

        float e1x5( const reco::BasicCluster &cluster );

        float e5x1( const reco::BasicCluster &cluster );

        float e2x2( const reco::BasicCluster &cluster );

        float e3x2( const reco::BasicCluster &cluster );

        float e3x3( const reco::BasicCluster &cluster );

        float e4x4( const reco::BasicCluster &cluster );

        float e5x5( const reco::BasicCluster &cluster );
        int   n5x5( const reco::BasicCluster &cluster );
        // energy in the 2x5 strip right of the max crystal (does not contain max crystal)
        // 2 crystals wide in eta, 5 wide in phi.
        float e2x5Right( const reco::BasicCluster &cluster );
        // energy in the 2x5 strip left of the max crystal (does not contain max crystal)
        float e2x5Left( const reco::BasicCluster &cluster );
        // energy in the 5x2 strip above the max crystal (does not contain max crystal)
        // 5 crystals wide in eta, 2 wide in phi.
        float e2x5Top( const reco::BasicCluster &cluster );

        // energy in the 5x2 strip below the max crystal (does not contain max crystal)
        float e2x5Bottom( const reco::BasicCluster &cluster );
        // energy in a 2x5 strip containing the seed (max) crystal.
        // 2 crystals wide in eta, 5 wide in phi.
        // it is the maximum of either (1x5left + 1x5center) or (1x5right + 1x5center)
        float e2x5Max( const reco::BasicCluster &cluster );
        // energies in the crystal left, right, top, bottom w.r.t. to the most energetic crystal
        float eLeft( const reco::BasicCluster &cluster );

        float eRight( const reco::BasicCluster &cluster );

        float eTop( const reco::BasicCluster &cluster );

        float eBottom( const reco::BasicCluster &cluster );
        // the energy of the most energetic crystal in the cluster
        float eMax( const reco::BasicCluster &cluster );
        // the energy of the second most energetic crystal in the cluster
        float e2nd( const reco::BasicCluster &cluster );

        // get the DetId and the energy of the maximum energy crystal of the input cluster
        std::pair<DetId, float> getMaximum( const reco::BasicCluster &cluster );

        std::vector<float> energyBasketFractionEta( const reco::BasicCluster &cluster );

        std::vector<float> energyBasketFractionPhi( const reco::BasicCluster &cluster );

        // return a vector v with v[0] = etaLat, v[1] = phiLat, v[2] = lat
        std::vector<float> lat( const reco::BasicCluster &cluster, bool logW = true, float w0 = 4.7 );

        // return a vector v with v[0] = covEtaEta, v[1] = covEtaPhi, v[2] = covPhiPhi
        std::vector<float> covariances(const reco::BasicCluster &cluster, float w0 = 4.7 );

        // return a vector v with v[0] = covIEtaIEta, v[1] = covIEtaIPhi, v[2] = covIPhiIPhi
        //this function calculates differences in eta/phi in units of crystals not global eta/phi
        //this is gives better performance in the crack regions of the calorimeter but gives otherwise identical results to covariances function
        //this is only defined for the barrel, it returns covariances when the cluster is in the endcap
        //Warning: covIEtaIEta has been studied by egamma, but so far covIPhiIPhi hasnt been studied extensively so there could be a bug in 
        //         the covIPhiIEta or covIPhiIPhi calculations. I dont think there is but as it hasnt been heavily used, there might be one
        std::vector<float> localCovariances(const reco::BasicCluster &cluster, float w0 = 4.7);

        std::vector<float> scLocalCovariances(const reco::SuperCluster &cluster, float w0 = 4.7);

        double zernike20( const reco::BasicCluster &cluster, double R0 = 6.6, bool logW = true, float w0 = 4.7 );
        double zernike42( const reco::BasicCluster &cluster, double R0 = 6.6, bool logW = true, float w0 = 4.7 );

        // get the detId's of a matrix centered in the maximum energy crystal = (0,0)
        // the size is specified by ixMin, ixMax, iyMin, iyMax in unit of crystals
        std::vector<DetId> matrixDetId( DetId id, int ixMin, int ixMax, int iyMin, int iyMax );
        // get the energy deposited in a matrix centered in the maximum energy crystal = (0,0)
        // the size is specified by ixMin, ixMax, iyMin, iyMax in unit of crystals
        float matrixEnergy( const reco::BasicCluster &cluster, DetId id, int ixMin, int ixMax, int iyMin, int iyMax );
  
}; // class EcalClusterLazyToolsT

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e1x3( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e1x3( cluster, getEcalRecHitCollection(cluster), topology_ );
}


template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e3x1( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e3x1( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e1x5( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e1x5( cluster, getEcalRecHitCollection(cluster), topology_ );
}


template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e5x1( const reco::BasicCluster &cluster )
{
  return EcalClusterToolsImpl::e5x1( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e2x2( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e2x2( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e3x2( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e3x2( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e3x3( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e3x3( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e4x4( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e4x4( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e5x5( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e5x5( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
int EcalClusterLazyToolsT<EcalClusterToolsImpl>::n5x5( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::n5x5( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e2x5Right( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e2x5Right( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e2x5Left( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e2x5Left( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e2x5Top( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e2x5Top( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e2x5Bottom( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e2x5Bottom( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e2x5Max( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e2x5Max( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::eLeft( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::eLeft( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::eRight( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::eRight( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::eTop( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::eTop( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::eBottom( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::eBottom( cluster, getEcalRecHitCollection(cluster), topology_ );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::eMax( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::eMax( cluster, getEcalRecHitCollection(cluster) );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::e2nd( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::e2nd( cluster, getEcalRecHitCollection(cluster) );
}

template<class EcalClusterToolsImpl>
std::pair<DetId, float> EcalClusterLazyToolsT<EcalClusterToolsImpl>::getMaximum( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::getMaximum( cluster, getEcalRecHitCollection(cluster) );
}

template<class EcalClusterToolsImpl>
std::vector<float> EcalClusterLazyToolsT<EcalClusterToolsImpl>::energyBasketFractionEta( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::energyBasketFractionEta( cluster, getEcalRecHitCollection(cluster) );
}

template<class EcalClusterToolsImpl>
std::vector<float> EcalClusterLazyToolsT<EcalClusterToolsImpl>::energyBasketFractionPhi( const reco::BasicCluster &cluster )
{
        return EcalClusterToolsImpl::energyBasketFractionPhi( cluster, getEcalRecHitCollection(cluster) );
}

template<class EcalClusterToolsImpl>
std::vector<float> EcalClusterLazyToolsT<EcalClusterToolsImpl>::lat( const reco::BasicCluster &cluster, bool logW, float w0 )
{
        return EcalClusterToolsImpl::lat( cluster, getEcalRecHitCollection(cluster), geometry_, logW, w0 );
}

template<class EcalClusterToolsImpl>
std::vector<float> EcalClusterLazyToolsT<EcalClusterToolsImpl>::covariances(const reco::BasicCluster &cluster, float w0 )
{
        return EcalClusterToolsImpl::covariances( cluster, getEcalRecHitCollection(cluster), topology_, geometry_, w0 );
}

template<class EcalClusterToolsImpl>
std::vector<float> EcalClusterLazyToolsT<EcalClusterToolsImpl>::localCovariances(const reco::BasicCluster &cluster, float w0 )
{
        return EcalClusterToolsImpl::localCovariances( cluster, getEcalRecHitCollection(cluster), topology_, w0 );
}

template<class EcalClusterToolsImpl>
std::vector<float> EcalClusterLazyToolsT<EcalClusterToolsImpl>::scLocalCovariances(const reco::SuperCluster &cluster, float w0 )
{
        return EcalClusterToolsImpl::scLocalCovariances( cluster, getEcalRecHitCollection(cluster), topology_, w0 );
}

template<class EcalClusterToolsImpl>
double EcalClusterLazyToolsT<EcalClusterToolsImpl>::zernike20( const reco::BasicCluster &cluster, double R0, bool logW, float w0 )
{
        return EcalClusterToolsImpl::zernike20( cluster, getEcalRecHitCollection(cluster), geometry_, R0, logW, w0 );
}


template<class EcalClusterToolsImpl>
double EcalClusterLazyToolsT<EcalClusterToolsImpl>::zernike42( const reco::BasicCluster &cluster, double R0, bool logW, float w0 )
{
        return EcalClusterToolsImpl::zernike42( cluster, getEcalRecHitCollection(cluster), geometry_, R0, logW, w0 );
}

template<class EcalClusterToolsImpl>
std::vector<DetId> EcalClusterLazyToolsT<EcalClusterToolsImpl>::matrixDetId( DetId id, int ixMin, int ixMax, int iyMin, int iyMax )
{
        return EcalClusterToolsImpl::matrixDetId( topology_, id, ixMin, ixMax, iyMin, iyMax );
}

template<class EcalClusterToolsImpl>
float EcalClusterLazyToolsT<EcalClusterToolsImpl>::matrixEnergy( const reco::BasicCluster &cluster, DetId id, int ixMin, int ixMax, int iyMin, int iyMax )
{
  return EcalClusterToolsImpl::matrixEnergy( cluster, getEcalRecHitCollection(cluster), topology_, id, ixMin, ixMax, iyMin, iyMax );
}

namespace noZS {
    typedef EcalClusterLazyToolsT<noZS::EcalClusterTools> EcalClusterLazyTools;
}
typedef EcalClusterLazyToolsT<::EcalClusterTools> EcalClusterLazyTools;



#endif
