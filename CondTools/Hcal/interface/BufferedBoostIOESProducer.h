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
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/Serialization/interface/eos/portable_iarchive.hpp"
#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionBuffer.h"

template<class DataType, class MyRecord>
class BufferedBoostIOESProducer : public edm::ESProducer
{
public:
    typedef boost::shared_ptr<DataType> ReturnType;

    inline BufferedBoostIOESProducer(const edm::ParameterSet&)
        {setWhatProduced(this);}

    inline virtual ~BufferedBoostIOESProducer() {}

    ReturnType produce(const MyRecord&);
};

template<class DataType, class MyRecord>
typename BufferedBoostIOESProducer<DataType,MyRecord>::ReturnType
BufferedBoostIOESProducer<DataType,MyRecord>::produce(const MyRecord& iRecord)
{
    edm::ESHandle<OOTPileupCorrectionBuffer> handle;
    iRecord.get(handle);
    std::istringstream is(handle->str());
    eos::portable_iarchive ar(is);
    ReturnType ret(new DataType());
    ar & *ret;
    return ret;
}

#endif // CondTools_Hcal_BufferedBoostIOESProducer_h
