// -*- C++ -*-
//
// Package:    FakeAlignmentSource
// Class:      FakeAlignmentSource
// 
/**\class FakeAlignmentSource FakeAlignmentSource.cc Alignment/FakeAlignmentProducer/plugins/FakeAlignmentSource.cc

Description: Producer of fake alignment data for all geometries (currently: Tracker, DT and CSC)
             (including IOV, in contrast to FakeAlignmentProducer)

Implementation: 
The alignment objects are filled with dummy/empty data, 
reconstruction Geometry should notice that and not pass to GeometryAligner.
*/
//
// Original Author:  Gero Flucke (based on FakeAlignmentProducer written by Frederic Ronga)
//         Created:  June 24, 2007
// $Id: FakeAlignmentSource.cc,v 1.3 2010/11/23 13:24:32 mussgill Exp $
//
//


// System
#include <memory>
#include <string>

// Framework
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"

class FakeAlignmentSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
public:
  FakeAlignmentSource(const edm::ParameterSet&);
  ~FakeAlignmentSource() {}

  /// Tracker and its APE
  std::auto_ptr<Alignments> produceTkAli(const TrackerAlignmentRcd&) {
    return std::auto_ptr<Alignments>(new Alignments);
  }
  std::auto_ptr<AlignmentErrors> produceTkAliErr(const TrackerAlignmentErrorRcd&) { 
    return std::auto_ptr<AlignmentErrors>(new AlignmentErrors);
  }

  /// DT and its APE
  std::auto_ptr<Alignments> produceDTAli(const DTAlignmentRcd&) {
    return std::auto_ptr<Alignments>(new Alignments);
  }
  std::auto_ptr<AlignmentErrors> produceDTAliErr(const DTAlignmentErrorRcd&) {
    return std::auto_ptr<AlignmentErrors>(new AlignmentErrors);
  }

  /// CSC and its APE
  std::auto_ptr<Alignments> produceCSCAli(const CSCAlignmentRcd&) {
    return std::auto_ptr<Alignments>(new Alignments);
  }
  std::auto_ptr<AlignmentErrors> produceCSCAliErr(const CSCAlignmentErrorRcd&) {
    return std::auto_ptr<AlignmentErrors>(new AlignmentErrors);
  }

  /// GlobalPositions
  std::auto_ptr<Alignments> produceGlobals(const GlobalPositionRcd&) {
    return std::auto_ptr<Alignments>(new Alignments);
  }

  /// Tracker surface deformations
  std::auto_ptr<AlignmentSurfaceDeformations>
  produceTrackerSurfaceDeformation(const TrackerSurfaceDeformationRcd&) {
    return std::auto_ptr<AlignmentSurfaceDeformations>(new AlignmentSurfaceDeformations);
  }

 protected:
  /// provide (dummy) IOV
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey& /*dummy*/,
			       const edm::IOVSyncValue& ioSyncVal, edm::ValidityInterval& iov);

 private:

  bool produceTracker_;
  bool produceDT_;
  bool produceCSC_;
  bool produceGlobalPosition_;
  bool produceTrackerSurfaceDeformation_;
};

//________________________________________________________________________________________
//________________________________________________________________________________________
//________________________________________________________________________________________

FakeAlignmentSource::FakeAlignmentSource(const edm::ParameterSet& iConfig)
  :produceTracker_(iConfig.getParameter<bool>("produceTracker")),
   produceDT_(iConfig.getParameter<bool>("produceDT")),
   produceCSC_(iConfig.getParameter<bool>("produceCSC")),
   produceGlobalPosition_(iConfig.getParameter<bool>("produceGlobalPosition")),
   produceTrackerSurfaceDeformation_(iConfig.getParameter<bool>("produceTrackerSurfaceDeformation"))
{
  // This 'appendToDataLabel' is used by the framework to distinguish providers
  // with different settings and to request a special one by e.g.
  // iSetup.get<TrackerDigiGeometryRecord>().get("theLabel", tkGeomHandle);
  
  edm::LogInfo("Alignments") 
    << "@SUB=FakeAlignmentSource" << "Providing data with label '" 
    << iConfig.getParameter<std::string>("appendToDataLabel") << "'.";
  
  // Tell framework what data is produced by which method:
  if (produceTracker_) {
    this->setWhatProduced(this, &FakeAlignmentSource::produceTkAli);
    this->setWhatProduced(this, &FakeAlignmentSource::produceTkAliErr);
  }
  if (produceDT_) {
    this->setWhatProduced(this, &FakeAlignmentSource::produceDTAli);
    this->setWhatProduced(this, &FakeAlignmentSource::produceDTAliErr);
  }
  if (produceCSC_) {
    this->setWhatProduced(this, &FakeAlignmentSource::produceCSCAli);
    this->setWhatProduced(this, &FakeAlignmentSource::produceCSCAliErr);
  }
  if (produceGlobalPosition_) {
    this->setWhatProduced(this, &FakeAlignmentSource::produceGlobals);
  }
  if (produceTrackerSurfaceDeformation_) {
    this->setWhatProduced(this, &FakeAlignmentSource::produceTrackerSurfaceDeformation);
  }

  // Tell framework to provide IOV for the above data:
  if (produceTracker_) {
    this->findingRecord<TrackerAlignmentRcd>();
    this->findingRecord<TrackerAlignmentErrorRcd>();
  }
  if (produceDT_) {
    this->findingRecord<DTAlignmentRcd>();
    this->findingRecord<DTAlignmentErrorRcd>();
  }
  if (produceCSC_) {
    this->findingRecord<CSCAlignmentRcd>();
    this->findingRecord<CSCAlignmentErrorRcd>();
  }
  if (produceGlobalPosition_) {
    this->findingRecord<GlobalPositionRcd>();
  }
  if (produceTrackerSurfaceDeformation_) {
    this->findingRecord<TrackerSurfaceDeformationRcd>();
  }
}

void FakeAlignmentSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& /*dummy*/, 
					    const edm::IOVSyncValue& ioSyncVal, 
					    edm::ValidityInterval& outValidity )
{
  // Implementation copied from SiStripGainFakeESSource: unlimited IOV
  outValidity = edm::ValidityInterval(ioSyncVal.beginOfTime(), ioSyncVal.endOfTime());
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(FakeAlignmentSource);
