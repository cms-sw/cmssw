#ifndef HcalRawToDigi_h
#define HcalRawToDigi_h

/** \class HcalRawToDigi
 *
 * HcalRawToDigi is the EDProducer subclass which runs 
 * the Hcal Unpack algorithm.
 *
 * \author Jeremiah Mans
      
 *
 * \version   1st Version June 10, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDataFrameFilter.h"

class HcalRawToDigi : public edm::EDProducer
{
public:
  explicit HcalRawToDigi(const edm::ParameterSet& ps);
  virtual ~HcalRawToDigi();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
private:
  HcalUnpacker unpacker_;
  std::auto_ptr<HcalMapping> readoutMap_;
  HcalDataFrameFilter filter_;
  std::string readoutMapSource_;
  std::vector<int> fedUnpackList_;
  int firstFED_;
};

#endif
