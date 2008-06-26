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
// $Id: FakeAlignmentProducer.cc,v 1.5 2008/02/18 20:10:48 pivarski Exp $
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
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"

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

 protected:
  /// provide (dummy) IOV
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey& /*dummy*/,
			       const edm::IOVSyncValue& ioSyncVal, edm::ValidityInterval& iov);
};

//________________________________________________________________________________________
//________________________________________________________________________________________
//________________________________________________________________________________________

FakeAlignmentSource::FakeAlignmentSource(const edm::ParameterSet& iConfig) 
{

  edm::LogInfo("Alignments") 
    << "@SUB=FakeAlignmentSource" << "Providing data with label '" 
    << iConfig.getParameter<std::string>("appendToDataLabel") << "'.";

  // Tell framework what data is produced by which method:
  this->setWhatProduced(this, &FakeAlignmentSource::produceTkAli);
  this->setWhatProduced(this, &FakeAlignmentSource::produceTkAliErr);
  this->setWhatProduced(this, &FakeAlignmentSource::produceDTAli);
  this->setWhatProduced(this, &FakeAlignmentSource::produceDTAliErr);
  this->setWhatProduced(this, &FakeAlignmentSource::produceCSCAli);
  this->setWhatProduced(this, &FakeAlignmentSource::produceCSCAliErr);
  this->setWhatProduced(this, &FakeAlignmentSource::produceGlobals);

  // Tell framework to provide IOV for the above data:
  this->findingRecord<TrackerAlignmentRcd>();
  this->findingRecord<TrackerAlignmentErrorRcd>();
  this->findingRecord<DTAlignmentRcd>();
  this->findingRecord<DTAlignmentErrorRcd>();
  this->findingRecord<CSCAlignmentRcd>();
  this->findingRecord<CSCAlignmentErrorRcd>();
  this->findingRecord<GlobalPositionRcd>();
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
