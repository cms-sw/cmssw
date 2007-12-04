// -*- C++ -*-
//
// Package:    FakeAlignmentProducer
// Class:      FakeAlignmentProducer
// 
/**\class FakeAlignmentProducer FakeAlignmentProducer.h Alignment/FakeAlignmentProducer/interface/FakeAlignmentProducer.h

Description: Producer of fake alignment data for all geometries (currently: Tracker, DT and CSC)

Implementation: 
The alignement objects are filled with dummy data (not useable by the reconstruction Geometry!)
*/
//
// Original Author:  Frederic Ronga
//         Created:  Fri Feb  9 19:24:38 CET 2007
// $Id: FakeAlignmentProducer.cc,v 1.2 2007/10/03 08:54:12 fronga Exp $
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
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorRcd.h"

class FakeAlignmentProducer : public edm::ESProducer {
public:
  FakeAlignmentProducer(const edm::ParameterSet&);
  ~FakeAlignmentProducer() {}

  std::auto_ptr<Alignments> 
  produceTkAli(const TrackerAlignmentRcd&) { return std::auto_ptr<Alignments>(); }
  std::auto_ptr<Alignments> 
  produceDTAli(const DTAlignmentRcd&) { return std::auto_ptr<Alignments>(); }
  std::auto_ptr<Alignments>
  produceCSCAli(const CSCAlignmentRcd&)  { return std::auto_ptr<Alignments>(); }


  std::auto_ptr<AlignmentErrors> 
  produceTkAliErr(const TrackerAlignmentErrorRcd&) { return std::auto_ptr<AlignmentErrors>(); }
  std::auto_ptr<AlignmentErrors>
  produceDTAliErr(const DTAlignmentErrorRcd&) { return std::auto_ptr<AlignmentErrors>(); }
  std::auto_ptr<AlignmentErrors>
  produceCSCAliErr(const CSCAlignmentErrorRcd&) { return std::auto_ptr<AlignmentErrors>(); }


};

FakeAlignmentProducer::FakeAlignmentProducer(const edm::ParameterSet& iConfig)
{

  edm::LogInfo("Alignments") << "This is a fake alignment producer";

  setWhatProduced( this, &FakeAlignmentProducer::produceTkAli );
  setWhatProduced( this, &FakeAlignmentProducer::produceTkAliErr );
  setWhatProduced( this, &FakeAlignmentProducer::produceDTAli );
  setWhatProduced( this, &FakeAlignmentProducer::produceDTAliErr );
  setWhatProduced( this, &FakeAlignmentProducer::produceCSCAli );
  setWhatProduced( this, &FakeAlignmentProducer::produceCSCAliErr );
}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(FakeAlignmentProducer);
