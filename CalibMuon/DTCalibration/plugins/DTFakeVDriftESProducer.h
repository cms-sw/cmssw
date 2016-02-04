#ifndef _DTFAKEVDRIFT_H
#define _DTFAKEVDRIFT_H

/** \class DTFakeVDriftESProducer
 *  ESProducer to store in the EventSetup fake vDrift value read from cfg  
 *
 *  $Date: 2008/09/19 15:56:17 $
 *  $Revision: 1.1 $
 *  \author S. Maselli - INFN Torino
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTMtime;
class DTMtimeRcd;

class DTFakeVDriftESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
public:
  DTFakeVDriftESProducer(const edm::ParameterSet&);
  virtual ~DTFakeVDriftESProducer();
  
  DTMtime* produce(const DTMtimeRcd&);
private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  
  double vDrift;
  double reso;
};

#endif
