// -*- C++ -*-
//
// Package:    FakeAlignmentProducer
// Class:      FakeAlignmentProducer
// 
/**\class FakeAlignmentProducer FakeAlignmentProducer.h Alignment/FakeAlignmentProducer/interface/FakeAlignmentProducer.h

Description: Producer of fake alignment data for all geometries (currently: Tracker, DT and CSC)
             (but will not provide IOV as the FakeAlignmentSource)

Implementation: 
The alignment objects are filled with dummy/empty data, 
reconstruction Geometry should notice that and not pass to GeometryAligner.
*/
//
// Original Author:  Frederic Ronga
//         Created:  Fri Feb  9 19:24:38 CET 2007
// $Id: FakeAlignmentProducer.cc,v 1.6 2008/06/26 10:00:35 flucke Exp $
//
//


// System
#include <memory>

// Framework
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"

class FakeAlignmentProducer : public edm::ESProducer {
public:
  FakeAlignmentProducer(const edm::ParameterSet&);
  ~FakeAlignmentProducer() {}

  std::auto_ptr<Alignments>
  produceTkAli(const TrackerAlignmentRcd&) { return std::auto_ptr<Alignments>(new Alignments);}
  std::auto_ptr<Alignments> 
  produceDTAli(const DTAlignmentRcd&) { return std::auto_ptr<Alignments>(new Alignments);}
  std::auto_ptr<Alignments>
  produceCSCAli(const CSCAlignmentRcd&)  { return std::auto_ptr<Alignments>(new Alignments);}
  std::auto_ptr<Alignments>
  produceGlobals(const GlobalPositionRcd&) {return std::auto_ptr<Alignments>(new Alignments);}

  std::auto_ptr<AlignmentErrorsExtended> produceTkAliErr(const TrackerAlignmentErrorExtendedRcd&) {
    return std::auto_ptr<AlignmentErrorsExtended>(new AlignmentErrorsExtended);
  }
  std::auto_ptr<AlignmentErrorsExtended> produceDTAliErr(const DTAlignmentErrorExtendedRcd&) {
    return std::auto_ptr<AlignmentErrorsExtended>(new AlignmentErrorsExtended);
  }
  std::auto_ptr<AlignmentErrorsExtended> produceCSCAliErr(const CSCAlignmentErrorExtendedRcd&) {
    return std::auto_ptr<AlignmentErrorsExtended>(new AlignmentErrorsExtended);
  }

};

FakeAlignmentProducer::FakeAlignmentProducer(const edm::ParameterSet& iConfig) 
{
  // This 'appendToDataLabel' is used by the framework to distinguish providers
  // with different settings and to request a special one by e.g.
  // iSetup.get<TrackerDigiGeometryRecord>().get("theLabel", tkGeomHandle);
  edm::LogInfo("Alignments") 
    << "@SUB=FakeAlignmentProducer" << "Providing data with label '" 
    << iConfig.getParameter<std::string>("appendToDataLabel") << "'.";

  setWhatProduced( this, &FakeAlignmentProducer::produceTkAli );
  setWhatProduced( this, &FakeAlignmentProducer::produceTkAliErr );
  setWhatProduced( this, &FakeAlignmentProducer::produceDTAli );
  setWhatProduced( this, &FakeAlignmentProducer::produceDTAliErr );
  setWhatProduced( this, &FakeAlignmentProducer::produceCSCAli );
  setWhatProduced( this, &FakeAlignmentProducer::produceCSCAliErr );
  setWhatProduced( this, &FakeAlignmentProducer::produceGlobals );

}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(FakeAlignmentProducer);
