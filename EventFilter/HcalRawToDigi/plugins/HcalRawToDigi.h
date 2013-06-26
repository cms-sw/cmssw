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
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

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
  edm::InputTag dataTag_;
  HcalUnpacker unpacker_;
  HcalDataFrameFilter filter_;
  std::vector<int> fedUnpackList_;
  int firstFED_;
  bool unpackCalib_, unpackZDC_, unpackTTP_;
  bool silent_,complainEmptyData_;
  int unpackerMode_,expectedOrbitMessageTime_;

  struct Statistics {
    int max_hbhe, ave_hbhe;
    int max_ho, ave_ho;
    int max_hf, ave_hf;
    int max_tp, ave_tp;
    int max_tpho, ave_tpho;
    int max_calib, ave_calib;
    uint64_t n;
  } stats_;
};

#endif
