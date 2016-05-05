//-------------------------------------------------
//
/**  \class L2MuonSeedGeneratorFromL1T
 * 
 *   L2 muon seed generator:
 *   Transform the L1 informations in seeds for the
 *   L2 muon reconstruction
 *
 *
 *
 *   \author  A.Everett, R.Bellan, J. Alcaraz
 *
 *    ORCA's author: N. Neumeister 
 */
//
//--------------------------------------------------

// Class Header
#include "RecoMuon/L2MuonSeedGenerator/src/L2MuonSeedGeneratorFromL1T.h"


// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

using namespace std;
using namespace edm;
using namespace l1t;

// constructors
L2MuonSeedGeneratorFromL1T::L2MuonSeedGeneratorFromL1T(const edm::ParameterSet& iConfig) : 
  theSource(iConfig.getParameter<InputTag>("InputObjects")),
  theL1GMTReadoutCollection(iConfig.getParameter<InputTag>("GMTReadoutCollection")), // to be removed
  thePropagatorName(iConfig.getParameter<string>("Propagator")),
  theL1MinPt(iConfig.getParameter<double>("L1MinPt")),
  theL1MaxEta(iConfig.getParameter<double>("L1MaxEta")),
  theL1MinQuality(iConfig.getParameter<unsigned int>("L1MinQuality")),
  useOfflineSeed(iConfig.getUntrackedParameter<bool>("UseOfflineSeed", false)),
  useUnassociatedL1(iConfig.existsAs<bool>("UseUnassociatedL1") ? 
            iConfig.getParameter<bool>("UseUnassociatedL1") : true),
  matchingDR(iConfig.getParameter<std::vector<double> >("MatchDR")),          
  etaBins(iConfig.getParameter<std::vector<double> >("EtaMatchingBins")),          
  centralBxOnly_( iConfig.getParameter<bool>("CentralBxOnly") )
  {

  muCollToken_ = consumes<MuonBxCollection>(theSource);

  if(useOfflineSeed) {
    theOfflineSeedLabel = iConfig.getUntrackedParameter<InputTag>("OfflineSeedLabel");
    offlineSeedToken_ = consumes<edm::View<TrajectorySeed> >(theOfflineSeedLabel);
  
  // check that number of eta bins -1 matches number of dR cones
  if( matchingDR.size()!=etaBins.size() - 1  ) {
    throw cms::Exception("Configuration") << "Size of MatchDR "
					  << "does not match number of eta bins." << endl;
  }

  
  }
  
  // service parameters
  ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);

  // the estimator
  theEstimator = new Chi2MeasurementEstimator(10000.);


  produces<L2MuonTrajectorySeedCollection>(); 
}

// destructor
L2MuonSeedGeneratorFromL1T::~L2MuonSeedGeneratorFromL1T(){
  if (theService)   delete theService;
  if (theEstimator) delete theEstimator;
}

void
L2MuonSeedGeneratorFromL1T::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("GMTReadoutCollection", edm::InputTag("")); // to be removed
  desc.add<edm::InputTag>("InputObjects",edm::InputTag("hltGmtStage2Digis"));
  desc.add<string>("Propagator", "");
  desc.add<double>("L1MinPt",-1.);
  desc.add<double>("L1MaxEta",5.0);
  desc.add<unsigned int>("L1MinQuality",0);   
  desc.addUntracked<bool>("UseOfflineSeed",false);
  desc.add<bool>("UseUnassociatedL1", true);
  desc.add<std::vector<double>>("MatchDR", {0.3});
  desc.add<std::vector<double>>("EtaMatchingBins", {0., 2.5});
  desc.add<bool>("CentralBxOnly", true);
  desc.addUntracked<edm::InputTag>("OfflineSeedLabel", edm::InputTag(""));

  edm::ParameterSetDescription psd0;
  psd0.addUntracked<std::vector<std::string>>("Propagators", {
      "SteppingHelixPropagatorAny"
  });
  psd0.add<bool>("RPCLayers", true);
  psd0.addUntracked<bool>("UseMuonNavigation", true);
  desc.add<edm::ParameterSetDescription>("ServiceParameters", psd0);
  descriptions.add("L2MuonSeedGeneratorFromL1T",desc);
}

void L2MuonSeedGeneratorFromL1T::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  const std::string metname = "Muon|RecoMuon|L2MuonSeedGeneratorFromL1T";
  MuonPatternRecoDumper debug;

  auto_ptr<L2MuonTrajectorySeedCollection> output(new L2MuonTrajectorySeedCollection());
  
  edm::Handle<MuonBxCollection> muColl;
  iEvent.getByToken(muCollToken_, muColl);
  LogTrace(metname) << "Number of muons " << muColl->size() << endl;

  edm::Handle<edm::View<TrajectorySeed> > offlineSeedHandle;
  vector<int> offlineSeedMap;
  if(useOfflineSeed) {
    iEvent.getByToken(offlineSeedToken_, offlineSeedHandle);
    LogTrace(metname) << "Number of offline seeds " << offlineSeedHandle->size() << endl;
    offlineSeedMap = vector<int>(offlineSeedHandle->size(), 0);
  }

  for (int ibx = muColl->getFirstBX(); ibx <= muColl->getLastBX(); ++ibx) {
    if (centralBxOnly_ && (ibx != 0)) continue;
    for (auto it = muColl->begin(ibx); it != muColl->end(ibx); it++){
  
      unsigned int quality = it->hwQual();
      int valid_charge = it->hwChargeValid();
  
      float pt    =  it->pt();
      float eta   =  it->eta();
      float theta =  2*atan(exp(-eta));
      float phi   =  it->phi();      
      int charge  =  it->charge();
      // Set charge=0 for the time being if the valid charge bit is zero
      if (!valid_charge) charge = 0;

      int link = 36 + (int)(it -> tfMuonIndex() / 3.);
      bool barrel = true;
      if ( (link >= 36 && link <= 41) || (link >= 66 && link <= 71)) barrel = false;

      if ( pt < theL1MinPt || fabs(eta) > theL1MaxEta ) continue;
  
      LogTrace(metname) << "New L2 Muon Seed" ;
      LogTrace(metname) << "Pt = "         << pt     << " GeV/c";
      LogTrace(metname) << "eta = "        << eta; 
      LogTrace(metname) << "theta = "      << theta  << " rad";
      LogTrace(metname) << "phi = "        << phi    << " rad";
      LogTrace(metname) << "charge = "     << charge;
      LogTrace(metname) << "In Barrel? = " << barrel;
  
      if ( quality <= theL1MinQuality ) continue;
      LogTrace(metname) << "quality = "<< quality; 
  
      // Update the services
      theService->update(iSetup);

      const DetLayer *detLayer = 0;
      float radius = 0.;

      CLHEP::Hep3Vector vec(0.,1.,0.);
      vec.setTheta(theta);
      vec.setPhi(phi);
  
      DetId theid; 
      // Get the det layer on which the state should be put
      if ( barrel ){
        LogTrace(metname) << "The seed is in the barrel";
    
        // MB2
        DetId id = DTChamberId(0,2,0);
        detLayer = theService->detLayerGeometry()->idToLayer(id);
        LogTrace(metname) << "L2 Layer: " << debug.dumpLayer(detLayer);
    
        const BoundSurface* sur = &(detLayer->surface());
        const BoundCylinder* bc = dynamic_cast<const BoundCylinder*>(sur);

        radius = fabs(bc->radius()/sin(theta));
        theid  = id;    

        LogTrace(metname) << "radius "<<radius;

        if ( pt < 3.5 ) pt = 3.5;
      }
      else { 
        LogTrace(metname) << "The seed is in the endcap";
    
        DetId id;
        // ME2
        if ( theta < Geom::pi()/2. )
          id = CSCDetId(1,2,0,0,0); 
        else
          id = CSCDetId(2,2,0,0,0); 
    
        detLayer = theService->detLayerGeometry()->idToLayer(id);
        LogTrace(metname) << "L2 Layer: " << debug.dumpLayer(detLayer);

        radius = fabs(detLayer->position().z()/cos(theta));      
        theid = id;    
    
        if( pt < 1.0) pt = 1.0;
      }
      
      vec.setMag(radius);
  
      GlobalPoint pos(vec.x(),vec.y(),vec.z());
    
      GlobalVector mom(pt*cos(phi), pt*sin(phi), pt*cos(theta)/sin(theta));

      GlobalTrajectoryParameters param(pos,mom,charge,&*theService->magneticField());
      AlgebraicSymMatrix55 mat;
  
      mat[0][0] = (0.25/pt)*(0.25/pt);  // sigma^2(charge/abs_momentum)
      if ( !barrel ) mat[0][0] = (0.4/pt)*(0.4/pt);

      //Assign q/pt = 0 +- 1/pt if charge has been declared invalid
      if (!valid_charge) mat[0][0] = (1./pt)*(1./pt);
  
      mat[1][1] = 0.05*0.05;        // sigma^2(lambda)
      mat[2][2] = 0.2*0.2;          // sigma^2(phi)
      mat[3][3] = 20.*20.;          // sigma^2(x_transverse))
      mat[4][4] = 20.*20.;          // sigma^2(y_transverse))
  
      CurvilinearTrajectoryError error(mat);

      const FreeTrajectoryState state(param,error);
 
      LogTrace(metname) << "Free trajectory State from the parameters";
      LogTrace(metname) << debug.dumpFTS(state);

      // Propagate the state on the MB2/ME2 surface
      TrajectoryStateOnSurface tsos = theService->propagator(thePropagatorName)->propagate(state, detLayer->surface());
 
      LogTrace(metname) << "State after the propagation on the layer";
      LogTrace(metname) << debug.dumpLayer(detLayer);
      LogTrace(metname) << debug.dumpTSOS(tsos);

 	  double dRcone = matchingDR[0];
	  if ( fabs(eta) < etaBins.back() ){
		std::vector<double>::iterator lowEdge = std::upper_bound (etaBins.begin(), etaBins.end(), fabs(eta));     
		dRcone    =  matchingDR.at( lowEdge - etaBins.begin() - 1);   
	  }

      if (tsos.isValid()) {

        edm::OwnVector<TrackingRecHit> container;
   
        if(useOfflineSeed && ( !valid_charge || charge == 0) ) {

          const TrajectorySeed *assoOffseed = 
   	        associateOfflineSeedToL1(offlineSeedHandle, offlineSeedMap, tsos, dRcone );
    
          if(assoOffseed!=0) {
            PTrajectoryStateOnDet const & seedTSOS = assoOffseed->startingState();
            TrajectorySeed::const_iterator 
            tsci  = assoOffseed->recHits().first,
            tscie = assoOffseed->recHits().second;
            for(; tsci!=tscie; ++tsci) {
              container.push_back(*tsci);
            }
            output->push_back(L2MuonTrajectorySeed(seedTSOS,container,alongMomentum,
                              MuonRef(muColl,  distance(muColl->begin(muColl->getFirstBX()),it)  )));
          }
          else {
            if(useUnassociatedL1) {
              // convert the TSOS into a PTSOD
              PTrajectoryStateOnDet const & seedTSOS = trajectoryStateTransform::persistentState( tsos, theid.rawId());
              output->push_back(L2MuonTrajectorySeed(seedTSOS,container,alongMomentum,
                                MuonRef(muColl,  distance(muColl->begin(muColl->getFirstBX()),it)  )));
            }
          }
        }
        else if (useOfflineSeed && valid_charge){
          // Get the compatible dets on the layer
          std::vector< pair<const GeomDet*,TrajectoryStateOnSurface> > 
            detsWithStates = detLayer->compatibleDets(tsos, 
                              *theService->propagator(thePropagatorName), 
                              *theEstimator);   
          if (detsWithStates.size()){
  
            TrajectoryStateOnSurface newTSOS = detsWithStates.front().second;
            const GeomDet *newTSOSDet = detsWithStates.front().first;
    
            LogTrace(metname) << "Most compatible det";
            LogTrace(metname) << debug.dumpMuonId(newTSOSDet->geographicalId());
  
            LogDebug(metname) << "L1 info: Det and State:";
            LogDebug(metname) << debug.dumpMuonId(newTSOSDet->geographicalId());
  
            if (newTSOS.isValid()){
  
              //LogDebug(metname) << "(x, y, z) = (" << newTSOS.globalPosition().x() << ", " 
              //                  << newTSOS.globalPosition().y() << ", " << newTSOS.globalPosition().z() << ")";
              LogDebug(metname) << "pos: (r=" << newTSOS.globalPosition().mag() << ", phi=" 
                                << newTSOS.globalPosition().phi() << ", eta=" << newTSOS.globalPosition().eta() << ")";
              LogDebug(metname) << "mom: (q*pt=" << newTSOS.charge()*newTSOS.globalMomentum().perp() << ", phi=" 
                                << newTSOS.globalMomentum().phi() << ", eta=" << newTSOS.globalMomentum().eta() << ")";
  
              //LogDebug(metname) << "State on it";
              //LogDebug(metname) << debug.dumpTSOS(newTSOS);
  
              const TrajectorySeed *assoOffseed = 
                associateOfflineSeedToL1(offlineSeedHandle, offlineSeedMap, newTSOS, dRcone);
  
              if(assoOffseed!=0) {
                PTrajectoryStateOnDet const & seedTSOS = assoOffseed->startingState();
                TrajectorySeed::const_iterator 
                  tsci  = assoOffseed->recHits().first,
                  tscie = assoOffseed->recHits().second;
                for(; tsci!=tscie; ++tsci) {
                  container.push_back(*tsci);
                }
                output->push_back(L2MuonTrajectorySeed(seedTSOS,container,alongMomentum,
                                  MuonRef(muColl,  distance(muColl->begin(muColl->getFirstBX()),it)  )));
              }
              else {
                if(useUnassociatedL1) {
                  // convert the TSOS into a PTSOD
                  PTrajectoryStateOnDet const & seedTSOS = trajectoryStateTransform::persistentState( newTSOS,newTSOSDet->geographicalId().rawId());
                  output->push_back(L2MuonTrajectorySeed(seedTSOS,container,alongMomentum,
                                    MuonRef(muColl,  distance(muColl->begin(muColl->getFirstBX()),it)  )));
                }
              } 
            }
          } 
        }
        else {
          // convert the TSOS into a PTSOD
          PTrajectoryStateOnDet const & seedTSOS = trajectoryStateTransform::persistentState( tsos, theid.rawId());
          output->push_back(L2MuonTrajectorySeed(seedTSOS,container,alongMomentum,
                            MuonRef(muColl,  distance(muColl->begin(muColl->getFirstBX()),it)  )));
        }      
      }  
    }
  }
  
  iEvent.put(output);
}


// FIXME: does not resolve ambiguities yet! 
const TrajectorySeed* L2MuonSeedGeneratorFromL1T::associateOfflineSeedToL1( edm::Handle<edm::View<TrajectorySeed> > & offseeds, 
                                     std::vector<int> & offseedMap, 
                                     TrajectoryStateOnSurface & newTsos,
                                     double dRcone ) {

  const std::string metlabel = "Muon|RecoMuon|L2MuonSeedGeneratorFromL1T";
  MuonPatternRecoDumper debugtmp;

  edm::View<TrajectorySeed>::const_iterator offseed, endOffseed = offseeds->end();
  const TrajectorySeed *selOffseed = 0;
  double bestDr = 99999.;
  unsigned int nOffseed(0);
  int lastOffseed(-1);

  for(offseed=offseeds->begin(); offseed!=endOffseed; ++offseed, ++nOffseed) {
    if(offseedMap[nOffseed]!=0) continue;
    GlobalPoint  glbPos = theService->trackingGeometry()->idToDet(offseed->startingState().detId())->surface().toGlobal(offseed->startingState().parameters().position());
    GlobalVector glbMom = theService->trackingGeometry()->idToDet(offseed->startingState().detId())->surface().toGlobal(offseed->startingState().parameters().momentum());

    // Preliminary check
    double preDr = deltaR( newTsos.globalPosition().eta(), newTsos.globalPosition().phi(), glbPos.eta(), glbPos.phi() );
    if(preDr > 1.0) continue;

    const FreeTrajectoryState offseedFTS(glbPos, glbMom, offseed->startingState().parameters().charge(), &*theService->magneticField()); 
    TrajectoryStateOnSurface offseedTsos = theService->propagator(thePropagatorName)->propagate(offseedFTS, newTsos.surface());
    LogDebug(metlabel) << "Offline seed info: Det and State" << std::endl;
    LogDebug(metlabel) << debugtmp.dumpMuonId(offseed->startingState().detId()) << std::endl;
    //LogDebug(metlabel) << "(x, y, z) = (" << newTSOS.globalPosition().x() << ", " 
    //                     << newTSOS.globalPosition().y() << ", " << newTSOS.globalPosition().z() << ")" << std::endl;
    LogDebug(metlabel) << "pos: (r=" << offseedFTS.position().mag() << ", phi=" 
               << offseedFTS.position().phi() << ", eta=" << offseedFTS.position().eta() << ")" << std::endl;
    LogDebug(metlabel) << "mom: (q*pt=" << offseedFTS.charge()*offseedFTS.momentum().perp() << ", phi=" 
               << offseedFTS.momentum().phi() << ", eta=" << offseedFTS.momentum().eta() << ")" << std::endl << std::endl;
    //LogDebug(metlabel) << debugtmp.dumpFTS(offseedFTS) << std::endl;

    if(offseedTsos.isValid()) {
      LogDebug(metlabel) << "Offline seed info after propagation to L1 layer:" << std::endl;
      //LogDebug(metlabel) << "(x, y, z) = (" << offseedTsos.globalPosition().x() << ", " 
      //                   << offseedTsos.globalPosition().y() << ", " << offseedTsos.globalPosition().z() << ")" << std::endl;
      LogDebug(metlabel) << "pos: (r=" << offseedTsos.globalPosition().mag() << ", phi=" 
             << offseedTsos.globalPosition().phi() << ", eta=" << offseedTsos.globalPosition().eta() << ")" << std::endl;
      LogDebug(metlabel) << "mom: (q*pt=" << offseedTsos.charge()*offseedTsos.globalMomentum().perp() << ", phi=" 
             << offseedTsos.globalMomentum().phi() << ", eta=" << offseedTsos.globalMomentum().eta() << ")" << std::endl << std::endl;
      //LogDebug(metlabel) << debugtmp.dumpTSOS(offseedTsos) << std::endl;
      double newDr = deltaR( newTsos.globalPosition().eta(),     newTsos.globalPosition().phi(), 
                 offseedTsos.globalPosition().eta(), offseedTsos.globalPosition().phi() );
      LogDebug(metlabel) << "   -- DR = " << newDr << std::endl;
      if( newDr < dRcone && newDr<bestDr ) {  
        LogDebug(metlabel) << "          --> OK! " << newDr << std::endl << std::endl;
        selOffseed = &*offseed;
        bestDr = newDr;
        offseedMap[nOffseed] = 1;
        if(lastOffseed>-1) offseedMap[lastOffseed] = 0;
        lastOffseed = nOffseed;
      }
      else {
        LogDebug(metlabel) << "          --> Rejected. " << newDr << std::endl << std::endl;
      }
    }
    else {
      LogDebug(metlabel) << "Invalid offline seed TSOS after propagation!" << std::endl << std::endl;
    }
  }

  return selOffseed;
}
