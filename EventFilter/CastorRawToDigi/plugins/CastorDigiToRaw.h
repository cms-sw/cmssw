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

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/CastorRawToDigi/interface/CastorPacker.h"
#include "EventFilter/CastorRawToDigi/interface/CastorCtdcPacker.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

class CastorDigiToRaw : public edm::global::EDProducer<>
{
public:
  explicit CastorDigiToRaw(const edm::ParameterSet& ps);
  void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& c) const override;

private:
  const edm::InputTag castorTag_;
  const bool usingctdc_;
  const edm::EDGetTokenT<CastorDigiCollection> tok_input_;
  const edm::EDPutTokenT<FEDRawDataCollection> tok_put_;
};

#endif
