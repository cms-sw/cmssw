// -*- C++ -*-
//
// Package:    ConversionSeedFilterCharge
// Class:      ConversionSeedFilterCharge
// 
/**\class ConversionSeedFilterCharge ConversionSeedFilterCharge.cc RecoTracker/TkSeedGenerator/src/ConversionSeedFilterCharge.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giuseppe Cerati
//         Created:  Thu Mar 11 10:48:48 CET 2010
// $Id: ConversionSeedFilterCharge.cc,v 1.3 2013/02/27 14:58:16 muzaffar Exp $
//
//

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/Math/interface/deltaPhi.h"

class ConversionSeedFilterCharge : public edm::EDProducer {
public:
  explicit ConversionSeedFilterCharge(const edm::ParameterSet&);
  ~ConversionSeedFilterCharge();
  
private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;
  edm::InputTag inputCollPos;
  edm::InputTag inputCollNeg;
  double deltaPhiCut, deltaCotThetaCut, deltaRCut, deltaZCut;
  
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  uint32_t maxInputSeeds;
};

ConversionSeedFilterCharge::ConversionSeedFilterCharge(const edm::ParameterSet& cfg): 
  inputCollPos(cfg.getParameter<edm::InputTag>("seedCollectionPos")),
  inputCollNeg(cfg.getParameter<edm::InputTag>("seedCollectionNeg")),
  deltaPhiCut(cfg.getParameter<double>("deltaPhiCut")),
  deltaCotThetaCut(cfg.getParameter<double>("deltaCotThetaCut")),
  deltaRCut(cfg.getParameter<double>("deltaRCut")),
  deltaZCut(cfg.getParameter<double>("deltaZCut")),
  maxInputSeeds(cfg.getParameter<uint32_t>("maxInputSeeds"))
{
  produces<TrajectorySeedCollection>();
}


ConversionSeedFilterCharge::~ConversionSeedFilterCharge() {}


void ConversionSeedFilterCharge::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
   using namespace edm;
   using namespace std;
   Handle<TrajectorySeedCollection> pInPos;
   iEvent.getByLabel(inputCollPos,pInPos);
   Handle<TrajectorySeedCollection> pInNeg;
   iEvent.getByLabel(inputCollNeg,pInNeg);

   iSetup.get<TrackerDigiGeometryRecord>().get(theG);
   iSetup.get<IdealMagneticFieldRecord>().get(theMF);  

   std::auto_ptr<TrajectorySeedCollection> result(new TrajectorySeedCollection());
   result->reserve(pInPos->size());

   if (pInPos->size()<maxInputSeeds && pInNeg->size()<maxInputSeeds) {

     edm::LogInfo("ConversionSeedFilterCharge") << "New Event \t Pos " << pInPos->size() << " \t Neg " << pInNeg->size() << std::endl;

     std::vector<int> inResult;
     for (TrajectorySeedCollection::const_iterator iS1=pInPos->begin(); iS1!=pInPos->end(); ++iS1){
       PTrajectoryStateOnDet state1 = iS1->startingState();
       DetId detId1(state1.detId());
       TrajectoryStateOnSurface tsos1 = trajectoryStateTransform::transientState( state1, &(theG->idToDet(detId1)->surface()), theMF.product());
       double phi1 = tsos1.globalMomentum().phi();
       double cotTheta1 = 1/tan(tsos1.globalMomentum().theta());
       double r1 = tsos1.globalPosition().perp();
       double z1 = tsos1.globalPosition().z();
       //cout << "detId1=" << detId1 << " phi1=" << phi1 << " cotTheta1=" << cotTheta1 << " r1=" << r1 << " z1=" << z1 << endl;
       
       bool pushed = false;
       for (TrajectorySeedCollection::const_iterator iS2=pInNeg->begin(); iS2!=pInNeg->end(); ++iS2){
	 PTrajectoryStateOnDet state2 = iS2->startingState();
	 DetId detId2(state2.detId());
	 TrajectoryStateOnSurface tsos2 = trajectoryStateTransform::transientState( state2, &(theG->idToDet(detId2)->surface()), theMF.product());
	 
	 double deltaPhi = fabs(reco::deltaPhi(phi1,tsos2.globalMomentum().phi()));
	 double deltaCotTheta = fabs(cotTheta1-1/tan(tsos2.globalMomentum().theta()));
	 double deltaR = fabs(r1-tsos2.globalPosition().perp());
	 double deltaZ = fabs(z1-tsos2.globalPosition().z());
	 
	 //	 double phi2 = tsos2.globalMomentum().phi();
	 //	 double cotTheta2 = 2/tan(tsos2.globalMomentum().theta());
	 //	 double r2 = tsos2.globalPosition().perp();
	 //	 double z2 = tsos2.globalPosition().z();
	 //cout << "j=" << j << " detId2=" << detId2 << " phi2=" << phi2 << " cotTheta2=" << cotTheta2 << " r2=" << r2 << " z2=" << z2 << endl;

	 if (deltaPhi<deltaPhiCut && deltaCotTheta<deltaCotThetaCut && deltaR<deltaRCut && deltaZ<deltaZCut) {
	   edm::LogInfo("ConversionSeedFilterCharge") << "[SearchAmongSeeds] match in pos " << iS1-pInPos->begin() << " " << iS2-pInNeg->begin() << std::endl;
	   //cout << "match" << endl;
	   if (!pushed) {
	     result->push_back(*iS1);
	     pushed = true;
	   }
	   if (std::find(inResult.begin(),inResult.end(),iS2-pInNeg->begin())==inResult.end()){
	     result->push_back(*iS2);
	     inResult.push_back(iS2-pInNeg->begin());
	   }
	 }
       }
     }
     
   }

   edm::LogInfo("ConversionSeedFilterCharge") << "\nNew Event : result size " << result->size() << std::endl;

   //cout << "sizes: pInPos=" << pInPos->size() << " pInNeg=" << pInNeg->size() << " result=" << result->size() << endl; 
   iEvent.put(result);
   
}

void ConversionSeedFilterCharge::beginJob(){}

void ConversionSeedFilterCharge::endJob() {}

DEFINE_FWK_MODULE(ConversionSeedFilterCharge);
