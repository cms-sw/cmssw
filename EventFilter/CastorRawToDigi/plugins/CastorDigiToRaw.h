#ifndef CastorDigiToRaw_h
#define CastorDigiToRaw_h

/** \class CastorDigiToRaw
 *
 * CastorDigiToRaw is the EDProducer subclass which runs 
 * the Castor Unpack algorithm.
 *
 * \author Alan Campbell    
 *
 * \version   1st Version April 18, 2008  
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/CastorRawToDigi/interface/CastorPacker.h"
#include "EventFilter/CastorRawToDigi/interface/CastorCtdcPacker.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

class CastorDigiToRaw : public edm::EDProducer
{
public:
  explicit CastorDigiToRaw(const edm::ParameterSet& ps);
  virtual ~CastorDigiToRaw();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:
  CastorPacker packer_;
  CastorCtdcPacker ctdcpacker_;
  edm::InputTag castorTag_;
  bool usingctdc_;
  edm::EDGetTokenT<CastorDigiCollection> tok_input_;

};

#endif
