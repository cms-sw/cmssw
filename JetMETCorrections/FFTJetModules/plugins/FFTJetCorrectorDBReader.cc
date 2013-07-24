// -*- C++ -*-
//
// Package:    JetMETCorrections/FFTJetModules
// Class:      FFTJetCorrectorDBReader
// 
/**\class FFTJetCorrectorDBReader FFTJetCorrectorDBReader.cc JetMETCorrections/FFTJetModules/plugins/FFTJetCorrectorDBReader.cc

 Description: writes a blob from a file into a database

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Wed Aug  1 20:59:12 CDT 2012
// $Id: FFTJetCorrectorDBReader.cc,v 1.1 2012/11/14 22:34:56 igv Exp $
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

#include "CondFormats/JetMETObjects/interface/FFTJetCorrectorParameters.h"
#include "CondFormats/DataRecord/interface/FFTJetCorrectorParametersRcdTypes.h"

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorParametersLoader.h"

#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

//
// class declaration
//
class FFTJetCorrectorDBReader : public edm::EDAnalyzer
{
public:
    explicit FFTJetCorrectorDBReader(const edm::ParameterSet&);
    virtual ~FFTJetCorrectorDBReader() {}

private:
    FFTJetCorrectorDBReader();
    FFTJetCorrectorDBReader(const FFTJetCorrectorDBReader&);
    FFTJetCorrectorDBReader& operator=(const FFTJetCorrectorDBReader&);

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    std::string record;
    std::string outputFile;
    bool printAsString;
    bool readArchive;
    bool isArchiveCompressed;
};

FFTJetCorrectorDBReader::FFTJetCorrectorDBReader(const edm::ParameterSet& ps)
    : init_param(std::string, record),
      init_param(std::string, outputFile),
      init_param(bool, printAsString),
      init_param(bool, readArchive),
      init_param(bool, isArchiveCompressed)
{
}

void FFTJetCorrectorDBReader::analyze(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup)
{
    edm::ESHandle<FFTJetCorrectorParameters> JetCorParams;
    StaticFFTJetCorrectorParametersLoader::instance().load(
        iSetup, record, JetCorParams);

    if (printAsString || readArchive)
        std::cout << "++++ FFTJetCorrectorDBReader: info for record \""
                  << record << '"' << std::endl;

    if (printAsString)
        std::cout << "++++ String rep: \"" 
                  << JetCorParams->str() << '"' << std::endl;
    else if (readArchive)
    {
        CPP11_auto_ptr<gs::StringArchive> par;

        {
            std::istringstream is(JetCorParams->str());
            if (isArchiveCompressed)
                par = gs::read_compressed_item<gs::StringArchive>(is);
            else
                par = gs::read_item<gs::StringArchive>(is);
        }

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

    if (!outputFile.empty())
    {
        std::ofstream of(outputFile.c_str(), std::ios_base::binary);
        if (!of.is_open())
            throw cms::Exception("InvalidArgument")
                << "Failed to open file \"" << outputFile << '"' << std::endl;
        if (!JetCorParams->empty())
        {
            of.write(JetCorParams->getBuffer(), JetCorParams->length());
            if (of.fail())
                throw cms::Exception("SystemError")
                    << "Output stream failure while writing file \""
                    << outputFile << '"' << std::endl;
        }
    }
}

DEFINE_FWK_MODULE(FFTJetCorrectorDBReader);
