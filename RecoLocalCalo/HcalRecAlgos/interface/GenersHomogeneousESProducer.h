#ifndef RecoLocalCalo_HcalRecAlgos_GenersHomogeneousESProducer_h
#define RecoLocalCalo_HcalRecAlgos_GenersHomogeneousESProducer_h

// -*- C++ -*-
//
// Package:    RecoLocalCalo/HcalRecAlgos
// Class:      GenersHomogeneousESProducer
// 
/**\class GenersHomogeneousESProducer GenersHomogeneousESProducer.h RecoLocalCalo/HcalRecAlgos/interface/GenersHomogeneousESProducer.h

 Description: producer for homogeneous Geners archives

 Implementation:
     A collection of homogeneous objects in a Geners archive
     (presumably retrieved from a database) is deserialized and
     placed into a double map with const index access (i.e., an
     exception will be thrown if a non-existent index is used
     with operator[]).

     When the objects are retrieved from the archive, they are
     filtered on their name and category with the usual Geners
     search facilities.
*/
//
// Original Author:  Igor Volobouev
//         Created:  Fri Apr  4 05:24:31 CEST 2014
//
//


// system include files
#include <memory>
#include <sstream>
#include <utility>

#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/HcalObjects/interface/HcalOOTPileupCorrectionData.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetDict.h"
#include "Alignment/Geners/interface/Reference.hh"
#include "Alignment/Geners/interface/StringArchive.hh"

//
// class declaration
//
template<typename DataType, class RecordType>
class GenersHomogeneousESProducer : public edm::ESProducer
{
public:
    GenersHomogeneousESProducer(const edm::ParameterSet&);
    inline virtual ~GenersHomogeneousESProducer() {}

    typedef DataType data_type;
    typedef RecordType record_type;
    typedef boost::shared_ptr<DataType> DataPtr;
    typedef FFTJetDict<std::string,FFTJetDict<std::string,DataPtr> > DataMap;
    typedef boost::shared_ptr<DataMap> ReturnType;

    ReturnType produce(const RecordType&);

private:
    // ----------member data ---------------------------
    gs::SearchSpecifier nameSearch_;
    gs::SearchSpecifier categorySearch_;

    static void insertDictItem(DataMap& seq, DataPtr fptr,
                               const std::string& name,
                               const std::string& category);
};

//
// constructor
//
template<typename DataType, class RecordType>
GenersHomogeneousESProducer<DataType,RecordType>::GenersHomogeneousESProducer(
    const edm::ParameterSet& ps)
    : nameSearch_(ps.getParameter<std::string>("name"),
                  ps.getParameter<bool>("nameIsRegex")),
      categorySearch_(ps.getParameter<std::string>("category"),
                      ps.getParameter<bool>("categoryIsRegex"))
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);
}


// ------------ method called to produce the data  ------------
template<typename DataType, class RecordType>
typename GenersHomogeneousESProducer<DataType,RecordType>::ReturnType
GenersHomogeneousESProducer<DataType,RecordType>::produce(const RecordType& iRecord)
{
    edm::ESTransientHandle<HcalOOTPileupCorrectionData> parHandle;
    iRecord.get(parHandle);

    std::istringstream is(parHandle->str());
    CPP11_auto_ptr<gs::StringArchive> ar(gs::read_item<gs::StringArchive>(is));
    gs::Reference<DataType> ref(*ar, nameSearch_, categorySearch_);
    ReturnType result(new DataMap());
    const unsigned long nFound = ref.size();
    for (unsigned long i=0; i<nFound; ++i)
    {
        CPP11_shared_ptr<const gs::CatalogEntry> e(ref.indexedCatalogEntry(i));
        CPP11_auto_ptr<DataType> pdata = ref.get(i);
        DataPtr shared(pdata.release());
        insertDictItem(*result, shared, e->name(), e->category());
    }
    return result;
}


template<typename DataType, class RecordType>
void GenersHomogeneousESProducer<DataType,RecordType>::insertDictItem(
    DataMap& seq, DataPtr fptr,
    const std::string& name, const std::string& category)
{
    typename DataMap::iterator it = seq.find(category);
    if (it == seq.end())
        it = seq.insert(std::make_pair(
             category, FFTJetDict<std::string,DataPtr>())).first;
    it->second.insert(std::make_pair(name, fptr));
}

#endif // RecoLocalCalo_HcalRecAlgos_GenersHomogeneousESProducer_h
