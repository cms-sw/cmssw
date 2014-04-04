// -*- C++ -*-
//
// Package:    RecoLocalCalo/HcalRecAlgos
// Class:      OOTPileupCorrectionDBReader
// 
/**\class OOTPileupCorrectionDBReader OOTPileupCorrectionDBReader.cc RecoLocalCalo/HcalRecAlgos/plugins/OOTPileupCorrectionDBReader.cc

 Description: gets a blob from a database and writes it into a file

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Fri Apr  4 05:40:18 CEST 2014
//
//

#include <iostream>
#include <sstream>
#include <fstream>

#include "Alignment/Geners/interface/StringArchive.hh"
#include "Alignment/Geners/interface/CompressedIO.hh"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/HcalObjects/interface/HcalOOTPileupCorrectionData.h"
#include "CondFormats/DataRecord/interface/HcalOOTPileupCorrectionRcd.h"

//
// class declaration
//
class OOTPileupCorrectionDBReader : public edm::EDAnalyzer
{
public:
    explicit OOTPileupCorrectionDBReader(const edm::ParameterSet&);
    virtual ~OOTPileupCorrectionDBReader() {}

private:
    OOTPileupCorrectionDBReader();
    OOTPileupCorrectionDBReader(const OOTPileupCorrectionDBReader&);
    OOTPileupCorrectionDBReader& operator=(const OOTPileupCorrectionDBReader&);

    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

    std::string outputFile_;
    bool dumpMetadata_;
};

OOTPileupCorrectionDBReader::OOTPileupCorrectionDBReader(const edm::ParameterSet& ps)
    : outputFile_(ps.getParameter<std::string>("outputFile")),
      dumpMetadata_(ps.getUntrackedParameter<bool>("dumpMetadata", true))
{
}

void OOTPileupCorrectionDBReader::analyze(const edm::Event& iEvent,
                                          const edm::EventSetup& iSetup)
{
    edm::ESHandle<HcalOOTPileupCorrectionData> p;
    iSetup.get<HcalOOTPileupCorrectionRcd>().get(p);

    if (dumpMetadata_)
    {
        // Dump info about archive contents
        std::istringstream is(p->str());
        CPP11_auto_ptr<gs::StringArchive> par = gs::read_item<gs::StringArchive>(is);
        const unsigned long long idSmall = par->smallestId();
        if (!idSmall)
            std::cout << "++++ No valid records in the archive" << std::endl;
        else
        {
            std::cout << "++++ Archive metadata begins" << std::endl;
            const unsigned long long idLarge = par->largestId();
            unsigned long long count = 0;
            for (unsigned long long id = idSmall; id <= idLarge; ++id)
                if (par->itemExists(id))
                {
                    CPP11_shared_ptr<const gs::CatalogEntry> e = 
                        par->catalogEntry(id);
                    std::cout << '\n';
                    e->humanReadable(std::cout);
                    ++count;
                }
            std::cout << "\n++++ Archive metadata ends, "
                      << count << " items total" << std::endl;
        }
    }

    if (!outputFile_.empty())
    {
        std::ofstream of(outputFile_.c_str(), std::ios_base::binary);
        if (!of.is_open())
            throw cms::Exception("InvalidArgument")
                << "Failed to open file \"" << outputFile_ << '"' << std::endl;
        if (!p->empty())
        {
            of.write(p->getBuffer(), p->length());
            if (of.fail())
                throw cms::Exception("SystemError")
                    << "Output stream failure while writing file \""
                    << outputFile_ << '"' << std::endl;
        }
    }
}

DEFINE_FWK_MODULE(OOTPileupCorrectionDBReader);
