#ifndef CondTools_Hcal_BoostIODBReader_h
#define CondTools_Hcal_BoostIODBReader_h

// -*- C++ -*-
//
// Package:    CondTools/Hcal
// Class:      BoostIODBReader
// 
/**\class BoostIODBReader BoostIODBReader.h CondTools/Hcal/interface/BoostIODBReader.h

 Description: reads an object from a database and puts it into a file using boost I/O

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Fri Apr 25 18:17:13 CDT 2014
//
//

#include <string>
#include <fstream>
#include <cassert>

#include "CondFormats/Serialization/interface/eos/portable_oarchive.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//
template<class DataType, class RecordType>
class BoostIODBReader : public edm::EDAnalyzer
{
public:
    typedef DataType data_type;
    typedef RecordType record_type;

    explicit BoostIODBReader(const edm::ParameterSet&);
    inline ~BoostIODBReader() override {}

private:
    void analyze(const edm::Event&, const edm::EventSetup&) override;

    std::string outputFile_;
};

template<class DataType, class RecordType>
BoostIODBReader<DataType,RecordType>::BoostIODBReader(const edm::ParameterSet& ps)
    : outputFile_(ps.getParameter<std::string>("outputFile"))
{
}

template<class DataType, class RecordType>
void BoostIODBReader<DataType,RecordType>::analyze(const edm::Event& iEvent,
                                                   const edm::EventSetup& iSetup)
{
    edm::ESHandle<DataType> p;
    iSetup.get<RecordType>().get(p);

    std::ofstream of(outputFile_.c_str(), std::ios_base::binary);
    if (!of.is_open())
        throw cms::Exception("InvalidArgument")
            << "Failed to open file \"" << outputFile_ << '"' << std::endl;

    eos::portable_oarchive ar(of);
    ar & *p;
}

#endif // CondTools_Hcal_BoostIODBReader_h
