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

class HcalHistogramRawToDigi : public edm::EDProducer
{
public:
  explicit HcalHistogramRawToDigi(const edm::ParameterSet& ps);
  virtual ~HcalHistogramRawToDigi();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
private:
  edm::InputTag dataTag_;
  HcalUnpacker unpacker_;
  std::vector<int> fedUnpackList_;
  int firstFED_;
};

#endif
