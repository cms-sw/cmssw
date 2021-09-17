#ifndef CondTools_Hcal_BufferedBoostIOESProducer_h
#define CondTools_Hcal_BufferedBoostIOESProducer_h

// -*- C++ -*-
//
// Package:    CondTools/Hcal
// Class:      BufferedBoostIOESProducer
//
/**\class BufferedBoostIOESProducer BufferedBoostIOESProducer.h CondTools/Hcal/interface/BufferedBoostIOESProducer.h

 Description: reads objects serialized by boost from simple string buffers

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Sun Aug 31 22:18:44 CDT 2014
// $Id: BufferedBoostIOESProducer.h,v 1.0 2014/08/31 22:18:44 igv Exp $
//
//

#include <sstream>
#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/Serialization/interface/eos/portable_iarchive.hpp"
#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionBuffer.h"

template <class DataType, class MyRecord>
class BufferedBoostIOESProducer : public edm::ESProducer {
public:
  typedef std::unique_ptr<DataType> ReturnType;

  inline BufferedBoostIOESProducer(const edm::ParameterSet&) : token_(setWhatProduced(this).consumes()) {}

  inline ~BufferedBoostIOESProducer() override {}

  ReturnType produce(const MyRecord&);

private:
  const edm::ESGetToken<OOTPileupCorrectionBuffer, MyRecord> token_;
};

template <class DataType, class MyRecord>
typename BufferedBoostIOESProducer<DataType, MyRecord>::ReturnType
BufferedBoostIOESProducer<DataType, MyRecord>::produce(const MyRecord& iRecord) {
  const auto& buffer = iRecord.get(token_);
  std::istringstream is(buffer.str());
  eos::portable_iarchive ar(is);
  auto ret = std::make_unique<DataType>();
  ar&* ret;
  return ret;
}

#endif  // CondTools_Hcal_BufferedBoostIOESProducer_h
