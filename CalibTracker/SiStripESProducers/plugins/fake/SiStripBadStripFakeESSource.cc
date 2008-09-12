// -*- C++ -*-
//
// Package:    SiStripBadStripFakeESSource
// Class:      SiStripBadStripFakeESSource
// 
/**\class SiStripBadStripFakeESSource  CalibTracker/SiStripBadStripFakeESSource/plugins/fake/SiStripBadStripFakeESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Wed Oct  3 11:46:09 CEST 2007
// $Id: SiStripBadStripFakeESSource.cc,v 1.1 2008/02/06 17:04:16 bainbrid Exp $
//
//

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripBadStripFakeESSource.h"

SiStripBadStripFakeESSource::SiStripBadStripFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<SiStripBadStripRcd>();
}


std::auto_ptr<SiStripBadStrip> SiStripBadStripFakeESSource::produce(const SiStripBadStripRcd& iRecord)
{
  std::auto_ptr<SiStripBadStrip> ptr(new SiStripBadStrip);
  return ptr;
}

void SiStripBadStripFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
							 const edm::IOVSyncValue& iov,
							 edm::ValidityInterval& iValidity){
  edm::ValidityInterval infinity( iov.beginOfTime(), iov.endOfTime() );
  iValidity = infinity;
}


