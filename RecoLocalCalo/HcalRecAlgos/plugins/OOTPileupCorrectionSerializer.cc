// -*- C++ -*-
//
// Package:    RecoLocalCalo/HcalRecAlgos
// Class:      OOTPileupCorrectionSerializer
// 
/**\class OOTPileupCorrectionSerializer OOTPileupCorrectionSerializer.cc RecoLocalCalo/HcalRecAlgos/plugins/OOTPileupCorrectionSerializer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Fri Apr  4 05:42:57 CEST 2014
//
//


// system include files
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/Geners/interface/Record.hh"
#include "Alignment/Geners/interface/Reference.hh"
#include "Alignment/Geners/interface/StringArchive.hh"
#include "Alignment/Geners/interface/IOException.hh"

// Include headers for all subclasses of AbsOOTPileupCorrection
#include "RecoLocalCalo/HcalRecAlgos/interface/DummyOOTPileupCorrection.h"


//
// Parser for the AbsOOTPileupCorrection subclasses.
//
// Do not use parameters called "name" and "category" -- these
// are reserved for the archive catalog.
//
static std::unique_ptr<AbsOOTPileupCorrection> parseOOTPileupCorrection(
    const edm::ParameterSet& ps)
{
    typedef std::unique_ptr<AbsOOTPileupCorrection> Result;

    const std::string& correctionType = ps.getParameter<std::string>("Class");

    if (!correctionType.compare("DummyOOTPileupCorrection"))
        return Result(new DummyOOTPileupCorrection(
                          ps.getParameter<std::string>("description"),
                          ps.getParameter<double>("scale")));

    else
        throw cms::Exception("BadConfig")
            << "In parseOOTPileupCorrection: unknown correction type \""
            << correctionType << "\"\n";
}


//
// class declaration
//
class OOTPileupCorrectionSerializer : public edm::EDAnalyzer 
{
public:
    explicit OOTPileupCorrectionSerializer(const edm::ParameterSet&);
    ~OOTPileupCorrectionSerializer();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    virtual void beginJob() override;
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;
};

//
// constructors and destructor
//
OOTPileupCorrectionSerializer::OOTPileupCorrectionSerializer(const edm::ParameterSet& ps)
{
    const std::vector<edm::ParameterSet>& corrs = ps.getParameter<
        std::vector<edm::ParameterSet> >("corrections");
    const unsigned nCors = corrs.size();
    if (nCors)
    {
        const bool verify = ps.getUntrackedParameter<bool>("verify", true);

        const std::string& outputFile = ps.getParameter<std::string>("outputFile");
        std::ofstream of(outputFile.c_str(), std::ios_base::binary);
        if (!of.is_open())
            throw gs::IOOpeningFailure("OOTPileupCorrectionSerializer", outputFile);

        // Iterate over corrections
        gs::StringArchive ar;
        for (unsigned i=0; i<nCors; ++i)
        {
            const std::string& name = corrs[i].getParameter<std::string>("name");
            const std::string& cat = corrs[i].getParameter<std::string>("category");
            std::unique_ptr<AbsOOTPileupCorrection> p(parseOOTPileupCorrection(corrs[i]));
            ar << gs::Record(*p, name.c_str(), cat.c_str());

            if (verify)                
            {
                const unsigned long long itemId = ar.lastItemId();
                gs::Reference<AbsOOTPileupCorrection> ref(ar, itemId);
                assert(ref.unique());
                CPP11_auto_ptr<AbsOOTPileupCorrection> readback(ref.get(0));
                if (*readback != *p)
                    throw gs::IOInvalidData("In OOTPileupCorrectionSerializer: "
                                            "readback verification failure");
            }
        }
        ar.flush();
        if (!gs::write_item(of, ar))
            throw cms::Exception("SystemError")
                << "In OOTPileupCorrectionSerializer: failed to write into file \""
                << outputFile << "\"\n";
    }
}


OOTPileupCorrectionSerializer::~OOTPileupCorrectionSerializer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
OOTPileupCorrectionSerializer::analyze(const edm::Event& /* iEvent */,
                                       const edm::EventSetup& /* iSetup */)
{
}


// ------------ method called once each job just before starting event loop  ------------
void 
OOTPileupCorrectionSerializer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
OOTPileupCorrectionSerializer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
OOTPileupCorrectionSerializer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
OOTPileupCorrectionSerializer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
OOTPileupCorrectionSerializer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
OOTPileupCorrectionSerializer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
OOTPileupCorrectionSerializer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(OOTPileupCorrectionSerializer);
