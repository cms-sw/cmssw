#ifndef HcalHistogramRawToDigi_h
#define HcalHistogramRawToDigi_h

/** \class HcalHistogramRawToDigi
 *
 * HcalHistogramRawToDigi is the EDProducer subclass which runs 
 * the Hcal Unpack algorithm for calibration-mode histograms.
 *
 * \author Jeremiah Mans
      
 *
 * \version   1st Version June 10, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

class HcalHistogramRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit HcalHistogramRawToDigi(const edm::ParameterSet& ps);
  ~HcalHistogramRawToDigi() override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::EDGetTokenT<FEDRawDataCollection> tok_data_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> tok_dbService_;
  HcalUnpacker unpacker_;
  std::vector<int> fedUnpackList_;
  int firstFED_;
};

#endif
