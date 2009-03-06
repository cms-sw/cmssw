// -*- C++ -*-
//
// Package:    SiStripBadFiberFakeESSource
// Class:      SiStripBadFiberFakeESSource
// 
/**\class SiStripBadFiberFakeESSource  CalibTracker/SiStripBadFiberFakeESSource/plugins/fake/SiStripBadFiberFakeESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Wed Oct  3 11:46:09 CEST 2007
// $Id: SiStripBadFiberFakeESSource.cc,v 1.1 2008/02/06 17:04:16 bainbrid Exp $
//
//

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripBadFiberFakeESSource.h"

SiStripBadFiberFakeESSource::SiStripBadFiberFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<SiStripBadFiberRcd>();
}


std::auto_ptr<SiStripBadStrip> SiStripBadFiberFakeESSource::produce(const SiStripBadFiberRcd& iRecord)
{
  std::auto_ptr<SiStripBadStrip> ptr(new SiStripBadStrip);
  return ptr;
}

void SiStripBadFiberFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
							 const edm::IOVSyncValue& iov,
							 edm::ValidityInterval& iValidity){
  edm::ValidityInterval infinity( iov.beginOfTime(), iov.endOfTime() );
  iValidity = infinity;
}


