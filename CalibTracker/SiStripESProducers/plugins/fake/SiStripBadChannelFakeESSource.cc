// -*- C++ -*-
//
// Package:    SiStripBadChannelFakeESSource
// Class:      SiStripBadChannelFakeESSource
// 
/**\class SiStripBadChannelFakeESSource  CalibTracker/SiStripBadChannelFakeESSource/plugins/fake/SiStripBadChannelFakeESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Wed Oct  3 11:46:09 CEST 2007
// $Id: SiStripBadChannelFakeESSource.cc,v 1.1 2008/02/06 17:04:16 bainbrid Exp $
//
//

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripBadChannelFakeESSource.h"

SiStripBadChannelFakeESSource::SiStripBadChannelFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<SiStripBadChannelRcd>();
}


std::auto_ptr<SiStripBadStrip> SiStripBadChannelFakeESSource::produce(const SiStripBadChannelRcd& iRecord)
{
  std::auto_ptr<SiStripBadStrip> ptr(new SiStripBadStrip);
  return ptr;
}

void SiStripBadChannelFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
							 const edm::IOVSyncValue& iov,
							 edm::ValidityInterval& iValidity){
  edm::ValidityInterval infinity( iov.beginOfTime(), iov.endOfTime() );
  iValidity = infinity;
}


