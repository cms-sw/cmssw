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

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

class HcalHistogramRawToDigi : public edm::EDProducer
{
public:
  explicit HcalHistogramRawToDigi(const edm::ParameterSet& ps);
  ~HcalHistogramRawToDigi() override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;
private:
  edm::EDGetTokenT<FEDRawDataCollection> tok_data_;
  HcalUnpacker unpacker_;
  std::vector<int> fedUnpackList_;
  int firstFED_;
};

#endif
