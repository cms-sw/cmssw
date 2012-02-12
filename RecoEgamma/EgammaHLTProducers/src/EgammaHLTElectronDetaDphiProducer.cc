/** \class EgammaHLTElectronDetaDphiProducer
 *
 *  \author Roberto Covarelli (CERN)
 * 
 * $Id: EgammaHLTElectronDetaDphiProducer.cc,v 1.3 2011/12/08 14:29:17 sani Exp $
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTElectronDetaDphiProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

//#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"//needed?
//#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"//needed?

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "RecoEgamma/EgammaTools/interface/ECALPositionCalculator.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

EgammaHLTElectronDetaDphiProducer::EgammaHLTElectronDetaDphiProducer(const edm::ParameterSet& config) : conf_(config)
{

  electronProducer_             = conf_.getParameter<edm::InputTag>("electronProducer");
  BSProducer_                   = conf_.getParameter<edm::InputTag>("BSProducer");
  useTrackProjectionToEcal_     = conf_.getParameter<bool>("useTrackProjectionToEcal");

  //register your products
  produces < reco::ElectronIsolationMap >( "Deta" ).setBranchAlias( "deta" );
  produces < reco::ElectronIsolationMap >( "Dphi" ).setBranchAlias( "dphi" ); 
}

EgammaHLTElectronDetaDphiProducer::~EgammaHLTElectronDetaDphiProducer(){}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EgammaHLTElectronDetaDphiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Get the HLT filtered objects
  edm::Handle<reco::ElectronCollection> electronHandle;
  iEvent.getByLabel(electronProducer_,electronHandle);

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(BSProducer_,recoBeamSpotHandle);
  // gets its position
  const reco::BeamSpot::Point& BSPosition = recoBeamSpotHandle->position(); 

  edm::ESHandle<MagneticField> theMagField;
  iSetup.get<IdealMagneticFieldRecord>().get(theMagField);

  reco::ElectronIsolationMap detaMap;
  reco::ElectronIsolationMap dphiMap;

  for(reco::ElectronCollection::const_iterator iElectron = electronHandle->begin(); iElectron != electronHandle->end(); iElectron++){
    
    reco::ElectronRef eleref(reco::ElectronRef(electronHandle,iElectron - electronHandle->begin()));
    const reco::SuperClusterRef theClus = eleref->superCluster();
    const math::XYZVector trackMom =  eleref->track()->momentum();
    
    math::XYZPoint SCcorrPosition(theClus->x()-BSPosition.x(), theClus->y()-BSPosition.y() , theClus->z()-eleref->track()->vz() );
    float deltaeta = fabs(SCcorrPosition.eta()-eleref->track()->eta());
    float deltaphi;

    if (useTrackProjectionToEcal_) { 

      ECALPositionCalculator posCalc;
      const math::XYZPoint vertex(BSPosition.x(),BSPosition.y(),eleref->track()->vz());

      float phi1= posCalc.ecalPhi(theMagField.product(),trackMom,vertex,1);
      float phi2= posCalc.ecalPhi(theMagField.product(),trackMom,vertex,-1);

      float deltaphi1=fabs( phi1 - theClus->position().phi() );
      if(deltaphi1>6.283185308) deltaphi1 -= 6.283185308;
      if(deltaphi1>3.141592654) deltaphi1 = 6.283185308-deltaphi1;

      float deltaphi2=fabs( phi2 - theClus->position().phi() );
      if(deltaphi2>6.283185308) deltaphi2 -= 6.283185308;
      if(deltaphi2>3.141592654) deltaphi2 = 6.283185308-deltaphi2;

      deltaphi = deltaphi1;
      if(deltaphi2<deltaphi1){ deltaphi = deltaphi2;}
    } else {

      deltaphi=fabs(eleref->track()->outerPosition().phi()-theClus->phi());
      if(deltaphi>6.283185308) deltaphi -= 6.283185308;
      if(deltaphi>3.141592654) deltaphi = 6.283185308-deltaphi;
    }
      
    detaMap.insert(eleref, deltaeta);
    dphiMap.insert(eleref, deltaphi);
  }

  std::auto_ptr<reco::ElectronIsolationMap> detMap(new reco::ElectronIsolationMap(detaMap));
  std::auto_ptr<reco::ElectronIsolationMap> dphMap(new reco::ElectronIsolationMap(dphiMap));
  iEvent.put(detMap, "Deta" );
  iEvent.put(dphMap, "Dphi" );

}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTTrackIsolationProducers);
