//*****************************************************************************
// File:      EgammaRecHitExtractor.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, hacked by Sam Harper (ie the ugly stuff is mine)
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************
//C++ includes
#include <vector>
#include <functional>

//ROOT includes
#include <Math/VectorUtil.h>

//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaRecHitExtractor.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace egammaisolation;
using namespace reco::isodeposit;

EgammaRecHitExtractor::EgammaRecHitExtractor(const edm::ParameterSet& par) : 
   etMin_(par.getParameter<double>("etMin")),
   extRadius_(par.getParameter<double>("extRadius")),
   intRadius_(par.getParameter<double>("intRadius")),
   barrelRecHitsTag_(par.getParameter<edm::InputTag>("barrelRecHits")), 
   endcapRecHitsTag_(par.getParameter<edm::InputTag>("endcapRecHits")),
   fakeNegativeDeposit_(par.getParameter<bool>("subtractSuperClusterEnergy")),
   tryBoth_(par.getParameter<bool>("tryBoth")),
   sameTag_(false)
{ 
   if ((intRadius_ != 0.0) && (fakeNegativeDeposit_)) {
      throw cms::Exception("Configuration Error") << "EgammaRecHitExtractor: " << 
         "If you use 'subtractSuperClusterEnergy', you *must* set 'intRadius' to ZERO; it does not make sense, otherwise.";
   }
   std::string isoVariable = par.getParameter<std::string>("isolationVariable");
   if (isoVariable == "et") {
      useEt_ = true;
   } else if (isoVariable == "energy") {
      useEt_ = false;
   } else {
      throw cms::Exception("Configuration Error") << "EgammaRecHitExtractor: isolationVariable '" << isoVariable << "' not known. " 
         << " Supported values are 'et', 'energy'. ";
   }
   std::string detector = par.getParameter<std::string>("detector");
   if (detector == "Ecal") {
      detector_ = DetId::Ecal;
   } else if (detector == "Hcal") {
      detector_ = DetId::Hcal;
   } else {
      throw cms::Exception("Configuration Error") << "EgammaRecHitExtractor: detector '" << detector << "' not known. " 
         << " Supported values are 'Ecal', 'Hcal'. ";
   }

   if (endcapRecHitsTag_.encode() ==  barrelRecHitsTag_.encode()) {
         sameTag_ = true;
	 if (tryBoth_) {
	   edm::LogWarning("EgammaRecHitExtractor") << "If you have configured 'barrelRecHits' == 'endcapRecHits', so I'm switching 'tryBoth' to FALSE.";
	   tryBoth_ = false;
	 }
   }
}

EgammaRecHitExtractor::~EgammaRecHitExtractor() { }

reco::IsoDeposit EgammaRecHitExtractor::deposit(const edm::Event & iEvent, 
      const edm::EventSetup & iSetup, const reco::Candidate &emObject ) const {
   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);
   const CaloGeometry* caloGeom = pG.product(); 

   std::auto_ptr<const CaloRecHitMetaCollectionV> barrelRecHits(0), endcapRecHits(0);

   if (detector_ == DetId::Ecal) {
      edm::Handle<EcalRecHitCollection> barrelEcalRecHitsH;
      iEvent.getByLabel(barrelRecHitsTag_, barrelEcalRecHitsH);
      barrelRecHits = std::auto_ptr<const CaloRecHitMetaCollectionV>(new EcalRecHitMetaCollection(*barrelEcalRecHitsH));
      if (!sameTag_) {
          edm::Handle<EcalRecHitCollection> endcapEcalRecHitsH;
          iEvent.getByLabel(endcapRecHitsTag_, endcapEcalRecHitsH);
          endcapRecHits = std::auto_ptr<const CaloRecHitMetaCollectionV>(new EcalRecHitMetaCollection(*endcapEcalRecHitsH));
      }
   } else if (detector_ == DetId::Hcal) {
      edm::Handle<HBHERecHitCollection> barrelHcalRecHitsH, endcapHcalRecHitsH;
      iEvent.getByLabel(barrelRecHitsTag_, barrelHcalRecHitsH);
      barrelRecHits = std::auto_ptr<const CaloRecHitMetaCollectionV>(new HBHERecHitMetaCollection(*barrelHcalRecHitsH));
      if (!sameTag_) {
         iEvent.getByLabel(endcapRecHitsTag_, endcapHcalRecHitsH);
         endcapRecHits = std::auto_ptr<const CaloRecHitMetaCollectionV>(new HBHERecHitMetaCollection(*endcapHcalRecHitsH));
      }
   }

   reco::SuperClusterRef sc = emObject.get<reco::SuperClusterRef>();
   math::XYZPoint caloPosition = sc->position();
   GlobalPoint point(caloPosition.x(), caloPosition.y() , caloPosition.z());

   Direction candDir(caloPosition.eta(), caloPosition.phi());
   reco::IsoDeposit deposit( candDir );
   deposit.setVeto( reco::IsoDeposit::Veto(candDir, intRadius_) ); 
   double sinTheta = sin(2*atan(exp(-sc->eta())));
   deposit.addCandEnergy(sc->energy() * (useEt_ ? sinTheta : 1.0)) ;

   CaloDualConeSelector doubleConeSel(intRadius_ ,extRadius_, caloGeom, detector_);

   double fakeEnergy = -sc->rawEnergy();
   if (fakeNegativeDeposit_) {
      deposit.addDeposit(candDir, fakeEnergy * (useEt_ ?  sinTheta : 1.0)); // not exactly clean...
   }

   // 3 possible options
   bool inBarrel = sameTag_ || ( abs(sc->eta()) < 1.5 );
   if (inBarrel || tryBoth_) {
      collect(deposit, point, doubleConeSel, caloGeom, *barrelRecHits);
   } 
   if ((!inBarrel) || tryBoth_) {
      collect(deposit, point, doubleConeSel, caloGeom, *endcapRecHits);
   }
   return deposit;
}

void EgammaRecHitExtractor::collect(reco::IsoDeposit &deposit, 
            const GlobalPoint &caloPosition, CaloDualConeSelector &cone, 
            const CaloGeometry* caloGeom,
            const CaloRecHitMetaCollectionV &hits) const 
{
   std::auto_ptr<CaloRecHitMetaCollectionV> chosen = cone.select(caloPosition, hits);
   for (CaloRecHitMetaCollectionV::const_iterator i = chosen->begin(), end = chosen->end() ; i != end;  ++i)  {
      const  GlobalPoint & position = caloGeom->getPosition(i->detid());
      double eta = position.eta();
      double energy = i->energy();
      double et = energy*sin(2*atan(exp(-eta)));
      if ( et > etMin_){
         deposit.addDeposit( Direction(eta, position.phi()), (useEt_ ? et : energy) );
      }
   }
} 


