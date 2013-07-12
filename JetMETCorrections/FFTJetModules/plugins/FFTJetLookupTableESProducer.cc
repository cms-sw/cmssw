// -*- C++ -*-
//
// Package:    JetMETCorrections/FFTJetModules
// Class:      FFTJetLookupTableESProducer
// 
/**\class FFTJetLookupTableESProducer FFTJetLookupTableESProducer.cc JetMETCorrections/FFTJetModules/plugins/FFTJetLookupTableESProducer.cc

 Description: produces lookup tables for jet reconstruction

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Thu Aug  2 22:34:02 CDT 2012
// $Id$
//
//


// system include files
#include <sstream>
#include <utility>

#include "boost/shared_ptr.hpp"

#include "Alignment/Geners/interface/CompressedIO.hh"
#include "Alignment/Geners/interface/StringArchive.hh"
#include "Alignment/Geners/interface/Reference.hh"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "CondFormats/JetMETObjects/interface/FFTJetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/FFTJetLUTTypes.h"

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetLookupTableRcd.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetLookupTableSequence.h"

typedef boost::shared_ptr<npstat::StorableMultivariateFunctor> StorableFunctorPtr;

static void insertLUTItem(FFTJetLookupTableSequence& seq, 
                          StorableFunctorPtr fptr,
                          const std::string& name,
                          const std::string& category)
{
    FFTJetLookupTableSequence::iterator it = seq.find(category);
    if (it == seq.end())
        it = seq.insert(std::make_pair(
            category, FFTJetDict<std::string,StorableFunctorPtr>())).first;
    it->second.insert(std::make_pair(name, fptr));
}

static boost::shared_ptr<FFTJetLookupTableSequence>
buildLookupTables(
    const FFTJetCorrectorParameters& tablePars,
    const std::vector<edm::ParameterSet>& tableDefs,
    const bool isArchiveCompressed, const bool verbose)
{
    // Load the archive stored in the FFTJetCorrectorParameters object
    CPP11_auto_ptr<gs::StringArchive> ar;
    {
        std::istringstream is(tablePars.str());
        if (isArchiveCompressed)
            ar = gs::read_compressed_item<gs::StringArchive>(is);
        else
            ar = gs::read_item<gs::StringArchive>(is);
    }

    boost::shared_ptr<FFTJetLookupTableSequence> ptr(
        new FFTJetLookupTableSequence());

    // Avoid loading the same item more than once
    std::set<unsigned long long> loadedSet;

    const unsigned nTables = tableDefs.size();
    for (unsigned itab=0; itab<nTables; ++itab)
    {
        const edm::ParameterSet& ps(tableDefs[itab]);
        gs::SearchSpecifier nameSearch(ps.getParameter<std::string>("name"),
                                       ps.getParameter<bool>("nameIsRegex"));
        gs::SearchSpecifier categorySearch(ps.getParameter<std::string>("category"),
                                           ps.getParameter<bool>("categoryIsRegex"));
        gs::Reference<npstat::StorableMultivariateFunctor> ref(
            *ar, nameSearch, categorySearch);
        const unsigned long nItems = ref.size();
        for (unsigned long item=0; item<nItems; ++item)
        {
            const unsigned long long id = ref.id(item);
            if (loadedSet.insert(id).second)
            {
                CPP11_auto_ptr<npstat::StorableMultivariateFunctor> p(ref.get(item));
                StorableFunctorPtr fptr(p.release());
                CPP11_shared_ptr<const gs::CatalogEntry> e = ar->catalogEntry(id);
                insertLUTItem(*ptr, fptr, e->name(), e->category());
                if (verbose)
                    std::cout << "In buildLookupTables: loaded table with name \""
                              << e->name() << "\" and category \""
                              << e->category() << '"' << std::endl;
            }
        }
    }

    return ptr;
}

//
// class declaration
//
template<typename CT>
class FFTJetLookupTableESProducer : public edm::ESProducer
{
public:
    typedef boost::shared_ptr<FFTJetLookupTableSequence> ReturnType;
    typedef FFTJetLookupTableRcd<CT> MyRecord;
    typedef FFTJetCorrectorParametersRcd<CT> ParentRecord;

    FFTJetLookupTableESProducer(const edm::ParameterSet&);
    virtual ~FFTJetLookupTableESProducer() {}

    ReturnType produce(const MyRecord&);

private:
    inline void doWhenChanged(const ParentRecord&)
        {remakeProduct = true;}

    // Module parameters
    std::vector<edm::ParameterSet> tables;
    bool isArchiveCompressed;
    bool verbose;

    // Other module variables
    bool remakeProduct;
    ReturnType product;
};

//
// constructors and destructor
//
template<typename CT>
FFTJetLookupTableESProducer<CT>::FFTJetLookupTableESProducer(
    const edm::ParameterSet& psIn)
    : tables(psIn.getParameter<std::vector<edm::ParameterSet> >("tables")),
      isArchiveCompressed(psIn.getParameter<bool>("isArchiveCompressed")),
      verbose(psIn.getUntrackedParameter<bool>("verbose")),
      remakeProduct(true)
{
    // The following line is needed to tell the framework what
    // data is being produced
    setWhatProduced(this, dependsOn(&FFTJetLookupTableESProducer::doWhenChanged));
}

// ------------ method called to produce the data  ------------
template<typename CT>
typename FFTJetLookupTableESProducer<CT>::ReturnType
FFTJetLookupTableESProducer<CT>::produce(const MyRecord& iRecord)
{
    if (remakeProduct)
    {
        // According to:
        // https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideHowToGetDependentRecord
        //
        // If ARecord is dependent on BRecord then you can
        // call the method getRecord of ARecord:
        //
        // const BRecord& b = aRecord.getRecord<BRecord>();
        //
        const ParentRecord& rec = iRecord.template getRecord<ParentRecord>();
        edm::ESTransientHandle<FFTJetCorrectorParameters> parHandle;
        rec.get(parHandle);
        product = buildLookupTables(
            *parHandle, tables, isArchiveCompressed, verbose);
        remakeProduct = false;
    }
    return product;
}

//
// define this as a plug-in
//
typedef FFTJetLookupTableESProducer<fftluttypes::EtaFlatteningFactors> FFTEtaFlatteningFactorsTableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::PileupRhoCalibration> FFTPileupRhoCalibrationTableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::PileupRhoEtaDependence> FFTPileupRhoEtaDependenceTableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT0>  FFTLUT0TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT1>  FFTLUT1TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT2>  FFTLUT2TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT3>  FFTLUT3TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT4>  FFTLUT4TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT5>  FFTLUT5TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT6>  FFTLUT6TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT7>  FFTLUT7TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT8>  FFTLUT8TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT9>  FFTLUT9TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT10> FFTLUT10TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT11> FFTLUT11TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT12> FFTLUT12TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT13> FFTLUT13TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT14> FFTLUT14TableESProducer;
typedef FFTJetLookupTableESProducer<fftluttypes::LUT15> FFTLUT15TableESProducer;

// =========================================================

DEFINE_FWK_EVENTSETUP_MODULE(FFTEtaFlatteningFactorsTableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPileupRhoCalibrationTableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPileupRhoEtaDependenceTableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT0TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT1TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT2TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT3TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT4TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT5TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT6TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT7TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT8TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT9TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT10TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT11TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT12TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT13TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT14TableESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTLUT15TableESProducer);
