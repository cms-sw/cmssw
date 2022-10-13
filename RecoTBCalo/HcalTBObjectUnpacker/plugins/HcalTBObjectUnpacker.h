#ifndef HcalTBObjectUnpacker_h
#define HcalTBObjectUnpacker_h

/** \class HcalTBObjectUnpacker
 *
 * HcalTBObjectUnpacker is the EDProducer subclass which runs 
 * the Hcal Test Beam Object Unpack algorithm.
 *
 * \author Phil Dudero
      
 *
 * \version   1st Version June 10, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBTriggerDataUnpacker.h"
#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBSlowDataUnpacker.h"
#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBTDCUnpacker.h"
#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBQADCUnpacker.h"
#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBSourcePositionDataUnpacker.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

class HcalTBObjectUnpacker : public edm::stream::EDProducer<> {
public:
  explicit HcalTBObjectUnpacker(const edm::ParameterSet& ps);

  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  int triggerFed_;
  int sdFed_;
  int spdFed_;
  int tdcFed_;
  int qadcFed_;
  std::string calibFile_;
  hcaltb::HcalTBTriggerDataUnpacker tdUnpacker_;
  hcaltb::HcalTBSlowDataUnpacker sdUnpacker_;
  hcaltb::HcalTBTDCUnpacker tdcUnpacker_;
  hcaltb::HcalTBQADCUnpacker qadcUnpacker_;
  hcaltb::HcalTBSourcePositionDataUnpacker spdUnpacker_;
  bool doRunData_, doTriggerData_, doEventPosition_, doTiming_, doSourcePos_, doBeamADC_;

  std::vector<std::vector<std::string> > calibLines_;
  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;

  void parseCalib();
};

#endif
