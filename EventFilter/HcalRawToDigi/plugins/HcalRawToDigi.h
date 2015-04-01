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

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDataFrameFilter.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

class HcalRawToDigi : public edm::stream::EDProducer <>
{
public:
  explicit HcalRawToDigi(const edm::ParameterSet& ps);
  virtual ~HcalRawToDigi();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  virtual void produce(edm::Event& , const edm::EventSetup&) override;
private:
  edm::EDGetTokenT<FEDRawDataCollection> tok_data_;
  HcalUnpacker unpacker_;
  HcalDataFrameFilter filter_;
  std::vector<int> fedUnpackList_;
  const int firstFED_;
  const bool unpackCalib_, unpackZDC_, unpackTTP_;
  const bool silent_, complainEmptyData_;
  const int unpackerMode_, expectedOrbitMessageTime_;
  std::string electronicsMapLabel_;

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
