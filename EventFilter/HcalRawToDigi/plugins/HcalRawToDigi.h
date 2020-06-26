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
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDataFrameFilter.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"

class HcalRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit HcalRawToDigi(const edm::ParameterSet& ps);
  ~HcalRawToDigi() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<FEDRawDataCollection> tok_data_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> tok_dbService_;
  edm::ESGetToken<HcalElectronicsMap, HcalElectronicsMapRcd> tok_electronicsMap_;
  HcalUnpacker unpacker_;
  HcalDataFrameFilter filter_;
  std::vector<int> fedUnpackList_;
  const int firstFED_;
  const bool unpackCalib_, unpackZDC_, unpackTTP_;
  bool unpackUMNio_;

  // input configs for additional QIE10 samples
  std::vector<int> saveQIE10DataNSamples_;
  std::vector<std::string> saveQIE10DataTags_;

  // input configs for additional QIE11 samples
  std::vector<int> saveQIE11DataNSamples_;
  std::vector<std::string> saveQIE11DataTags_;

  const bool silent_, complainEmptyData_;
  const int unpackerMode_, expectedOrbitMessageTime_;
  std::string electronicsMapLabel_;

  // maps to easily associate nSamples to
  // the tag for additional qie10 and qie11 info
  std::unordered_map<int, std::string> saveQIE10Info_;
  std::unordered_map<int, std::string> saveQIE11Info_;

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
