
#include "L1Trigger/VertexFinder/interface/Stub.h"


#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "L1Trigger/VertexFinder/interface/Settings.h"



namespace l1tVertexFinder {


//=== Store useful info about this stub.

Stub::Stub(const TTStubRef& ttStubRef, unsigned int index_in_vStubs, const Settings* settings, 
           const TrackerGeometry*  trackerGeometry, const TrackerTopology*  trackerTopology, const std::map<DetId, DetId>& geoDetIdMap) :
  TTStubRef(ttStubRef),
  settings_(settings)
{
  // Get coordinates of stub.
  // const TTStub<Ref_Phase2TrackerDigi_> *ttStubP = ttStubRef.get(); 

  DetId geoDetId = geoDetIdMap.find(ttStubRef->getDetId())->second;

  const GeomDetUnit* det0 = trackerGeometry->idToDetUnit( geoDetId );
  // To get other module, can do this
  // const GeomDetUnit* det1 = trackerGeometry->idToDetUnit( trackerTopology->partnerDetId( geoDetId ) );

  const PixelGeomDetUnit* theGeomDet = dynamic_cast< const PixelGeomDetUnit* >( det0 );
  const PixelTopology* topol = dynamic_cast< const PixelTopology* >( &(theGeomDet->specificTopology()) );
  MeasurementPoint measurementPoint = ttStubRef->getClusterRef(0)->findAverageLocalCoordinatesCentered();
  LocalPoint clustlp   = topol->localPosition(measurementPoint);
  GlobalPoint pos  =  theGeomDet->surface().toGlobal(clustlp);

  phi_ = pos.phi();
  r_   = pos.perp();
  z_   = pos.z();

  if (r_ < settings_->trackerInnerRadius() || r_ > settings_->trackerOuterRadius() || fabs(z_) > settings_->trackerHalfLength()) {
    throw cms::Exception("Stub: Stub found outside assumed tracker volume. Please update tracker dimensions specified in Settingsm.h!")<<" r="<<r_<<" z="<<z_<<" "<<ttStubRef->getDetId().subdetId()<<std::endl;
  }

  // Set info about the module this stub is in
  this->setModuleInfo(trackerGeometry, trackerTopology, geoDetId);
}


//=== Note which tracking particle(s), if any, produced this stub.
//=== The 1st argument is a map relating TrackingParticles to TP.

void Stub::fillTruth(const std::map<edm::Ptr< TrackingParticle >, const TP* >& translateTP, edm::Handle<TTStubAssMap> mcTruthTTStubHandle, edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle){

  const TTStubRef& ttStubRef(*this); // Cast to base class

  //--- Fill assocTP_ info. If both clusters in this stub were produced by the same single tracking particle, find out which one it was.

  bool genuine =  mcTruthTTStubHandle->isGenuine(ttStubRef); // Same TP contributed to both clusters?
  assocTP_ = nullptr;

  // Require same TP contributed to both clusters.
  if ( genuine ) {
    edm::Ptr< TrackingParticle > tpPtr = mcTruthTTStubHandle->findTrackingParticlePtr(ttStubRef);
    if (translateTP.find(tpPtr) != translateTP.end()) {
      assocTP_ = translateTP.at(tpPtr);
      // N.B. Since not all tracking particles are stored in InputData::vTPs_, sometimes no match will be found.
    }
  }

  // Fill assocTPs_ info.

  if (settings_->stubMatchStrict()) {

    // We consider only stubs in which this TP contributed to both clusters.
    if (assocTP_ != nullptr) assocTPs_.insert(assocTP_);

  } else {

    // We consider stubs in which this TP contributed to either cluster.

    for (unsigned int iClus = 0; iClus <= 1; iClus++) { // Loop over both clusters that make up stub.
       const TTClusterRef& ttClusterRef = ttStubRef->getClusterRef(iClus);

      // Now identify all TP's contributing to either cluster in stub.
      std::vector< edm::Ptr< TrackingParticle > > vecTpPtr = mcTruthTTClusterHandle->findTrackingParticlePtrs(ttClusterRef);

      for (const edm::Ptr< TrackingParticle>& tpPtr : vecTpPtr) {
        if (translateTP.find(tpPtr) != translateTP.end()) {
          assocTPs_.insert( translateTP.at(tpPtr) );
          // N.B. Since not all tracking particles are stored in InputData::vTPs_, sometimes no match will be found.
        }
      }
    }
  }

  //--- Also note which tracking particles produced the two clusters that make up the stub

  for (unsigned int iClus = 0; iClus <= 1; iClus++) { // Loop over both clusters that make up stub.
    const TTClusterRef& ttClusterRef = ttStubRef->getClusterRef(iClus);

    bool genuineCluster =  mcTruthTTClusterHandle->isGenuine(ttClusterRef); // Only 1 TP made cluster?
    assocTPofCluster_[iClus] = nullptr;

    // Only consider clusters produced by just one TP.
    if ( genuineCluster ) {
      edm::Ptr< TrackingParticle > tpPtr = mcTruthTTClusterHandle->findTrackingParticlePtr(ttClusterRef);

      if (translateTP.find(tpPtr) != translateTP.end()) {
        assocTPofCluster_[iClus] = translateTP.at(tpPtr);
        // N.B. Since not all tracking particles are stored in InputData::vTPs_, sometimes no match will be found.
      }
    }
  }

  // Sanity check - is truth info of stub consistent with that of its clusters?
  // Commented this out, as it throws errors for unknown reason with iErr=1. Apparently, "genuine" stubs can be composed of two clusters that are
  // not "genuine", providing that one of the TP that contributed to each cluster was the same.
  /*
  unsigned int iErr = 0;
  if (this->genuine()) { // Stub matches truth particle
    if ( ! ( this->genuineCluster()[0] && (this->assocTPofCluster()[0] == this->assocTPofCluster()[1]) ) ) iErr = 1;
  } else {
    if ( ! ( ! this->genuineCluster()[0] || (this->assocTPofCluster()[0] != this->assocTPofCluster()[1]) )  ) iErr = 2;
  }
  if (iErr > 0) {
    cout<<" DEBUGA "<<(this->assocTP() == nullptr)<<endl;
    cout<<" DEBUGB "<<(this->assocTPofCluster()[0] == nullptr)<<" "<<(this->assocTPofCluster()[1] == nullptr)<<endl;
    cout<<" DEBUGC "<<this->genuineCluster()[0]<<" "<<this->genuineCluster()[1]<<endl;
    if (this->assocTPofCluster()[0] != nullptr) cout<<" DEBUGD "<<this->assocTPofCluster()[0]->index()<<endl;
    if (this->assocTPofCluster()[1] != nullptr) cout<<" DEBUGE "<<this->assocTPofCluster()[1]->index()<<endl;
    //    throw cms::Exception("Stub: Truth info of stub & its clusters inconsistent!")<<iErr<<endl;
  }
  */
}


//=== Get reduced layer ID (in range 1-7), which can be packed into 3 bits so simplifying the firmware).

unsigned int Stub::layerIdReduced() const {
  // Don't bother distinguishing two endcaps, as no track can have stubs in both.
  unsigned int lay = (layerId_ < 20) ? layerId_ : layerId_ - 10; 

  // No genuine track can have stubs in both barrel layer 6 and endcap disk 11 etc., so merge their layer IDs.
  // WARNING: This is tracker geometry dependent, so may need changing in future ...
  if (lay == 6) lay = 11; 
  if (lay == 5) lay = 12; 
  if (lay == 4) lay = 13; 
  if (lay == 3) lay = 15; 
  // At this point, the reduced layer ID can have values of 1, 2, 11, 12, 13, 14, 15. So correct to put in range 1-7.
  if (lay > 10) lay -= 8;

  if (lay < 1 || lay > 7) throw cms::Exception("Stub: Reduced layer ID out of expected range");

  return lay;
}


void Stub::setModuleInfo(const TrackerGeometry* trackerGeometry, const TrackerTopology* trackerTopology, const DetId& detId) {

  idDet_ = detId();

  // Get min & max (r,phi,z) coordinates of the centre of the two sensors containing this stub.
  const GeomDetUnit* det0 = trackerGeometry->idToDetUnit( detId );
  const GeomDetUnit* det1 = trackerGeometry->idToDetUnit( trackerTopology->partnerDetId( detId ) );

  float R0 = det0->position().perp();
  float R1 = det1->position().perp();
  float PHI0 = det0->position().phi();
  float PHI1 = det1->position().phi();
  float Z0 = det0->position().z();
  float Z1 = det1->position().z();
  moduleMinR_   = std::min(R0,R1);
  moduleMaxR_   = std::max(R0,R1);
  moduleMinPhi_ = std::min(PHI0,PHI1);
  moduleMaxPhi_ = std::max(PHI0,PHI1);
  moduleMinZ_   = std::min(Z0,Z1);
  moduleMaxZ_   = std::max(Z0,Z1);

  // Note if module is PS or 2S, and whether in barrel or endcap.
  psModule_ = trackerGeometry->getDetectorType( detId ) == TrackerGeometry::ModuleType::Ph2PSP; // From https://github.com/cms-sw/cmssw/blob/CMSSW_8_1_X/Geometry/TrackerGeometryBuilder/README.md
  barrel_ = detId.subdetId()==StripSubdetector::TOB || detId.subdetId()==StripSubdetector::TIB;

  //  cout<<"DEBUG STUB "<<barrel_<<" "<<psModule_<<"  sep(r,z)=( "<<moduleMaxR_ - moduleMinR_<<" , "<<moduleMaxZ_ - moduleMinZ_<<" )    stub(r,z)=( "<<0.5*(moduleMaxR_ + moduleMinR_) - r_<<" , "<<0.5*(moduleMaxZ_ + moduleMinZ_) - z_<<" )"<<endl;

  // Encode layer ID.
  if (barrel_) {
    layerId_ = trackerTopology->layer( detId ); // barrel layer 1-6 encoded as 1-6
  } else {
    // layerId_ = 10*detId.iSide() + detId.iDisk(); // endcap layer 1-5 encoded as 11-15 (endcap A) or 21-25 (endcapB)
    // EJC This seems to give the same encoding
    layerId_ = 10*trackerTopology->side( detId ) + trackerTopology->tidWheel( detId );
  }

  // Note module ring in endcap
  // endcapRing_ = barrel_  ?  0  :  detId.iRing();
  endcapRing_ = barrel_  ?  0  :  trackerTopology->tidRing( detId );

  // Get sensor strip or pixel pitch using innermost sensor of pair.

  const PixelGeomDetUnit* unit = reinterpret_cast<const PixelGeomDetUnit*>( det0 );
  const PixelTopology& topo = unit->specificTopology();
  const Bounds& bounds = det0->surface().bounds();

  std::pair<float, float> pitch = topo.pitch();
  stripPitch_ = pitch.first; // Strip pitch (or pixel pitch along shortest axis)
  stripLength_ = pitch.second;  //  Strip length (or pixel pitch along longest axis)
  nStrips_ = topo.nrows(); // No. of strips in sensor
  sensorWidth_ = bounds.width(); // Width of sensitive region of sensor (= stripPitch * nStrips).

  outerModuleAtSmallerR_ = false;
  if ( barrel_ && det0->position().perp() > det1->position().perp() ) {
    outerModuleAtSmallerR_ = true;
  }


  sigmaPerp_ = stripPitch_/sqrt(12.); // resolution perpendicular to strip (or to longest pixel axis)
  sigmaPar_  = stripLength_/sqrt(12.); // resolution parallel to strip (or to longest pixel axis)
}

} // end namespace l1tVertexFinder
