#ifndef JetMETCorrections_FFTJetModules_plugins_FFTJetCorrectionESProducer_h
#define JetMETCorrections_FFTJetModules_plugins_FFTJetCorrectionESProducer_h

// -*- C++ -*-
//
// Package:    FFTJetModules
// Class:      FFTJetCorrectionESProducer
// 
/**\class FFTJetCorrectionESProducer FFTJetCorrectionESProducer.h JetMETCorrections/FFTJetModules/plugins/FFTJetCorrectionESProducer.h

 Description: produces the correction sequence

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Thu Aug  2 22:34:02 CDT 2012
// $Id: FFTJetCorrectionESProducer.h,v 1.1 2012/11/14 22:34:56 igv Exp $
//
//


// system include files
#include <sstream>

#include "boost/shared_ptr.hpp"

#include "Alignment/Geners/interface/CompressedIO.hh"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "CondFormats/JetMETObjects/interface/FFTJetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/FFTJetCorrTypes.h"

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequenceRcd.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorTransientFromJet.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorResultFromTransient.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequence.h"

#include "JetMETCorrections/FFTJetModules/interface/FFTJetESParameterParser.h"

//
// Don't make the function which builds the sequence
// a member of the producer class -- this way it gets
// instantiated only as many times as there are corrector
// sequence types but not as many times as there are
// record types.
//
template<class CorrectorSequence>
static boost::shared_ptr<CorrectorSequence>
buildCorrectorSequence(
    const FFTJetCorrectorParameters& tablePars,
    const std::vector<edm::ParameterSet>& sequence,
    const bool isArchiveCompressed, const bool verbose)
{
    typedef typename CorrectorSequence::Corrector Corrector;
    typedef typename CorrectorSequence::jet_type jet_type;

    // Load the archive stored in the FFTJetCorrectorParameters object
    CPP11_auto_ptr<gs::StringArchive> ar;
    {
        std::istringstream is(tablePars.str());
        if (isArchiveCompressed)
            ar = gs::read_compressed_item<gs::StringArchive>(is);
        else
            ar = gs::read_item<gs::StringArchive>(is);
    }

    // Create an empty corrector sequence
    boost::shared_ptr<CorrectorSequence> ptr(new CorrectorSequence());

    // Go over the parameter sets in the VPSet and
    // configure all correction levels. Add each new
    // level to the sequence.
    const unsigned nLevels = sequence.size();
    for (unsigned lev=0; lev<nLevels; ++lev)
        ptr->addCorrector(parseFFTJetCorrector<Corrector>(
                              sequence[lev], *ar, verbose));

    // Check for proper level order.
    // Assume that level 0 is special and can happen anywhere.
    unsigned previousLevel = 0;
    for (unsigned lev=0; lev<nLevels; ++lev)
    {
        const unsigned level = (*ptr)[lev].level();
        if (level)
        {
            if (level <= previousLevel)
                throw cms::Exception("FFTJetBadConfig")
                    << "Error in buildCorrectorSequence: "
                    << "correction levels are out of order\n";
            previousLevel = level;
        }
    }

    return ptr;
}

//
// class declaration
//
template<typename CT>
class FFTJetCorrectionESProducer : public edm::ESProducer
{
public:
    // Lookup various types
    typedef typename FFTJetCorrectionsTypemap<CT>::jet_type jet_type;
    typedef FFTJetCorrectorSequence<
        jet_type,
        FFTJetCorrectorTransientFromJet,
        FFTJetCorrectorResultFromTransient
    > CorrectorSequence;
    typedef boost::shared_ptr<CorrectorSequence> ReturnType;
    typedef FFTJetCorrectorSequenceRcd<CT> MyRecord;
    typedef FFTJetCorrectorParametersRcd<CT> ParentRecord;

    FFTJetCorrectionESProducer(const edm::ParameterSet&);
    virtual ~FFTJetCorrectionESProducer() {}

    ReturnType produce(const MyRecord&);

private:
    inline void doWhenChanged(const ParentRecord&)
        {remakeProduct = true;}

    // Module parameters
    std::vector<edm::ParameterSet> sequence;
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
FFTJetCorrectionESProducer<CT>::FFTJetCorrectionESProducer(
    const edm::ParameterSet& psIn)
    : sequence(psIn.getParameter<std::vector<edm::ParameterSet> >("sequence")),
      isArchiveCompressed(psIn.getParameter<bool>("isArchiveCompressed")),
      verbose(psIn.getUntrackedParameter<bool>("verbose", false)),
      remakeProduct(true)
{
    // The following line is needed to tell the framework what
    // data is being produced
    setWhatProduced(this, dependsOn(&FFTJetCorrectionESProducer::doWhenChanged));
}

// ------------ method called to produce the data  ------------
template<typename CT>
typename FFTJetCorrectionESProducer<CT>::ReturnType
FFTJetCorrectionESProducer<CT>::produce(const MyRecord& iRecord)
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
        product = buildCorrectorSequence<CorrectorSequence>(
            *parHandle, sequence, isArchiveCompressed, verbose);
        remakeProduct = false;
    }
    return product;
}

#endif // JetMETCorrections_FFTJetModules_plugins_FFTJetCorrectionESProducer_h
