#ifndef _DTFAKETTRIG_H
#define _DTFAKETTRIG_H

/** \class DTFakeTTrigESProducer
 *  ESProducer to store in the EventSetup fake ttrig value read from cfg  
 *
 *  $Date: 2008/12/09 22:44:10 $
 *  $Revision: 1.2 $
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
  double kFact;
};

#endif
