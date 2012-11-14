// -*- C++ -*-
//
// Package:    FFTJetModules
// Class:      FFTJetCorrectionESProducer
// 
/**\class FFTJetCorrectionESProducer FFTJetCorrectionESProducer.cc JetMETCorrections/FFTJetModules/plugins/FFTJetCorrectionESProducer.cc

 Description: produces the correction sequence

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

//
// define this as a plug-in
//
typedef FFTJetCorrectionESProducer<fftcorrtypes::BasicJet> FFTBasicJetCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::GenJet>   FFTGenJetCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::CaloJet>  FFTCaloJetCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PFJet>    FFTPFJetCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::TrackJet> FFTTrackJetCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::JPTJet>   FFTJPTJetCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PFCHS0>   FFTPFCHS0CorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PFCHS1>   FFTPFCHS1CorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PFCHS2>   FFTPFCHS2CorrectionESProducer;

typedef FFTJetCorrectionESProducer<fftcorrtypes::BasicJetSys> FFTBasicJetSysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::GenJetSys>   FFTGenJetSysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::CaloJetSys>  FFTCaloJetSysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PFJetSys>    FFTPFJetSysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::TrackJetSys> FFTTrackJetSysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::JPTJetSys>   FFTJPTJetSysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PFCHS0Sys>   FFTPFCHS0SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PFCHS1Sys>   FFTPFCHS1SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PFCHS2Sys>   FFTPFCHS2SysCorrectionESProducer;

typedef FFTJetCorrectionESProducer<fftcorrtypes::Gen0> FFTGen0CorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Gen1> FFTGen1CorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Gen2> FFTGen2CorrectionESProducer;

typedef FFTJetCorrectionESProducer<fftcorrtypes::PF0> FFTPF0CorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF1> FFTPF1CorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF2> FFTPF2CorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF3> FFTPF3CorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF4> FFTPF4CorrectionESProducer;

typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo0> FFTCalo0CorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo1> FFTCalo1CorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo2> FFTCalo2CorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo3> FFTCalo3CorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo4> FFTCalo4CorrectionESProducer;

typedef FFTJetCorrectionESProducer<fftcorrtypes::Gen0Sys> FFTGen0SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Gen1Sys> FFTGen1SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Gen2Sys> FFTGen2SysCorrectionESProducer;

typedef FFTJetCorrectionESProducer<fftcorrtypes::PF0Sys> FFTPF0SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF1Sys> FFTPF1SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF2Sys> FFTPF2SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF3Sys> FFTPF3SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF4Sys> FFTPF4SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF5Sys> FFTPF5SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF6Sys> FFTPF6SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF7Sys> FFTPF7SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF8Sys> FFTPF8SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::PF9Sys> FFTPF9SysCorrectionESProducer;

typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo0Sys> FFTCalo0SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo1Sys> FFTCalo1SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo2Sys> FFTCalo2SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo3Sys> FFTCalo3SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo4Sys> FFTCalo4SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo5Sys> FFTCalo5SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo6Sys> FFTCalo6SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo7Sys> FFTCalo7SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo8Sys> FFTCalo8SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::Calo9Sys> FFTCalo9SysCorrectionESProducer;

typedef FFTJetCorrectionESProducer<fftcorrtypes::CHS0Sys> FFTCHS0SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::CHS1Sys> FFTCHS1SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::CHS2Sys> FFTCHS2SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::CHS3Sys> FFTCHS3SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::CHS4Sys> FFTCHS4SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::CHS5Sys> FFTCHS5SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::CHS6Sys> FFTCHS6SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::CHS7Sys> FFTCHS7SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::CHS8Sys> FFTCHS8SysCorrectionESProducer;
typedef FFTJetCorrectionESProducer<fftcorrtypes::CHS9Sys> FFTCHS9SysCorrectionESProducer;

// =========================================================

DEFINE_FWK_EVENTSETUP_MODULE(FFTBasicJetCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTGenJetCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCaloJetCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPFJetCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTTrackJetCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTJPTJetCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPFCHS0CorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPFCHS1CorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPFCHS2CorrectionESProducer);

DEFINE_FWK_EVENTSETUP_MODULE(FFTBasicJetSysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTGenJetSysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCaloJetSysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPFJetSysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTTrackJetSysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTJPTJetSysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPFCHS0SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPFCHS1SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPFCHS2SysCorrectionESProducer);

DEFINE_FWK_EVENTSETUP_MODULE(FFTGen0CorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTGen1CorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTGen2CorrectionESProducer);

DEFINE_FWK_EVENTSETUP_MODULE(FFTPF0CorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF1CorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF2CorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF3CorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF4CorrectionESProducer);

DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo0CorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo1CorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo2CorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo3CorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo4CorrectionESProducer);

DEFINE_FWK_EVENTSETUP_MODULE(FFTGen0SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTGen1SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTGen2SysCorrectionESProducer);

DEFINE_FWK_EVENTSETUP_MODULE(FFTPF0SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF1SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF2SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF3SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF4SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF5SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF6SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF7SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF8SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTPF9SysCorrectionESProducer);

DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo0SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo1SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo2SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo3SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo4SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo5SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo6SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo7SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo8SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCalo9SysCorrectionESProducer);

DEFINE_FWK_EVENTSETUP_MODULE(FFTCHS0SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCHS1SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCHS2SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCHS3SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCHS4SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCHS5SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCHS6SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCHS7SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCHS8SysCorrectionESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FFTCHS9SysCorrectionESProducer);
