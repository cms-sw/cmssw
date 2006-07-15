
/** \file
 *
 *  implementation of RPCMonitorEfficiency class
 *
 *  $Date: 2006/06/26 13:25:29 $
 *  $Revision: 1.1 $
 *
 * \author  Camilo Carrillo
 */

#include <map>
#include <string>
#include <DQM/RPCMonitorDigi/interface/RPCMonitorEfficiency.h>
///Log messages
#include <FWCore/ServiceRegistry/interface/Service.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

/* This Class Header */
#include "RecoLocalMuon/DTSegment/test/DTRecSegment4DReader.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "TFile.h"
#include "TH1F.h"

/// for DQM ILA
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/eventSetupGetImplementation.h>
#include <MagneticField/Records/interface/IdealMagneticFieldRecord.h>
#include <TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h>
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h>

RPCMonitorEfficiency::RPCMonitorEfficiency( const edm::ParameterSet& pset ):counter(0){
  nameInLog = pset.getUntrackedParameter<std::string>("moduleLogName", "RPCEfficiency");
  saveRootFile  = pset.getUntrackedParameter<bool>("EfficDQMSaveRootFile", false); 
  saveRootFileEventsInterval  = pset.getUntrackedParameter<int>("EfficEventsInterval", 100000); 
  RootFileName  = pset.getUntrackedParameter<std::string>("RootFileNameEfficiency", "RPCMonitorEfficiency.root"); 
  /// get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();
}

RPCMonitorEfficiency::~RPCMonitorEfficiency(){
}

void RPCMonitorEfficiency::endJob(void){
  dbe->save(RootFileName);  
}


void RPCMonitorEfficiency::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup ){
  edm::LogInfo (nameInLog) <<"Beginning event efficiency evaluation " << counter;
  
  if((!(counter%saveRootFileEventsInterval))&&(saveRootFile) ) {
    dbe->save(RootFileName);
  }
  
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  iEvent.getByLabel(theRecHits4DLabel, all4DSegments);
  DTRecSegment4DCollection::const_iterator segment;
  //std::cout<<"Reconstructed segments: "<<std::endl;
  
  for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){
    //std::cout<<"Chamber ID= "<<(*segment).chamberId()<<std::endl;
    hPositionX->Fill( (*segment).localPosition().x());
    LocalTrajectoryParameters localTrajectParameters=this->makeLocalTrajectory(*segment);
    LocalTrajectoryError localError;
    const BoundPlane  theSurface= this->makeSurface(iSetup,*segment);
    const MagneticField * magField= this->makeMagneticField(iSetup);
    edm::ESHandle<MagneticField> magfield;
    iSetup.get<IdealMagneticFieldRecord>().get(magfield);
    const MagneticField *field = magfield.product();
    TrajectoryStateOnSurface*tsos=new TrajectoryStateOnSurface(localTrajectParameters,localError,theSurface,field,1);
  }
  //std::cout<<"---"<<std::endl;
  counter++;
}

LocalTrajectoryParameters RPCMonitorEfficiency::makeLocalTrajectory(DTRecSegment4D theSegment){
  LocalPoint  thePoint= theSegment.localPosition();  
  LocalVector theDirection=theSegment.localDirection();
  LocalTrajectoryParameters theTrajectoryParameters(thePoint,theDirection,1);
  return theTrajectoryParameters;
}

const BoundPlane RPCMonitorEfficiency::makeSurface(const edm::EventSetup&  eventSetup,const DTRecSegment4D &  theSegment){
  edm::ESHandle<DTGeometry> pDD;
  eventSetup.get<MuonGeometryRecord>().get(pDD);
  const GeomDet* gdet=pDD->idToDet(theSegment.geographicalId());
  const BoundPlane & surface = gdet->surface();
  return  surface;
}

const MagneticField * RPCMonitorEfficiency::makeMagneticField(const edm::EventSetup& eventSetup){
  edm::ESHandle<MagneticField> magfield;
  eventSetup.get<IdealMagneticFieldRecord>().get(magfield);
  const MagneticField *field = magfield.product();
  return field;
}


