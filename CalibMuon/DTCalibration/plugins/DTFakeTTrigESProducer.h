#ifndef _DTFAKETTRIG_H
#define _DTFAKETTRIG_H

/** \class DTFakeTTrigESProducer
 *  ESProducer to store in the EventSetup fake ttrig value read from cfg  
 *
 *  $Date: 2007/06/07 07:55:43 $
 *  $Revision: 1.1 $
 *  \author S. Bolognesi - INFN Torino
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

class DTTtrig;
class DTTtrigRcd;

class DTFakeTTrigESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
public:
  DTFakeTTrigESProducer(const edm::ParameterSet&);
  virtual ~DTFakeTTrigESProducer();
  
  DTTtrig* produce(const DTTtrigRcd&);
private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  
  double tMean;
  double sigma;
};

#endif
