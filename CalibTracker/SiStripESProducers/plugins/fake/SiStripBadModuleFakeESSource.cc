// -*- C++ -*-
//
// Package:    SiStripBadModuleFakeESSource
// Class:      SiStripBadModuleFakeESSource
// 
/**\class SiStripBadModuleFakeESSource  CalibTracker/SiStripBadModuleFakeESSource/plugins/fake/SiStripBadModuleFakeESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Wed Oct  3 11:46:09 CEST 2007
// $Id: SiStripBadModuleFakeESSource.cc,v 1.1 2008/02/06 17:04:16 bainbrid Exp $
//
//

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripBadModuleFakeESSource.h"

SiStripBadModuleFakeESSource::SiStripBadModuleFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<SiStripBadModuleRcd>();
}


std::auto_ptr<SiStripBadStrip> SiStripBadModuleFakeESSource::produce(const SiStripBadModuleRcd& iRecord)
{
  std::auto_ptr<SiStripBadStrip> ptr(new SiStripBadStrip);
  return ptr;
}

void SiStripBadModuleFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
							 const edm::IOVSyncValue& iov,
							 edm::ValidityInterval& iValidity){
  edm::ValidityInterval infinity( iov.beginOfTime(), iov.endOfTime() );
  iValidity = infinity;
}


