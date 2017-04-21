#ifndef CalibratedElectronProducer_h
#define CalibratedElectronProducer_h

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include <TFile.h>


class GBRForestGetterFromDB: public edm::one::EDAnalyzer<>
{
    public:
        explicit GBRForestGetterFromDB( const edm::ParameterSet & ) ;
        virtual ~GBRForestGetterFromDB() ;
        virtual void analyze( const edm::Event &, const edm::EventSetup & ) override ;

    private:
        std::string theGBRForestName;
        std::string theOutputFileName;
        std::string theOutputObjectName;
        edm::ESHandle<GBRForest> theGBRForestHandle;
};

GBRForestGetterFromDB::GBRForestGetterFromDB( const edm::ParameterSet & conf ) :
    theGBRForestName(conf.getParameter<std::string>("grbForestName")),
    theOutputFileName(conf.getUntrackedParameter<std::string>("outputFileName")),
    theOutputObjectName(conf.getUntrackedParameter<std::string>("outputObjectName", theGBRForestName.empty() ? "GBRForest" : theGBRForestName))
{
}

GBRForestGetterFromDB::~GBRForestGetterFromDB()
{
}

void
GBRForestGetterFromDB::analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup ) 
{
    iSetup.get<GBRWrapperRcd>().get(theGBRForestName, theGBRForestHandle);
    TFile *fOut = TFile::Open(theOutputFileName.c_str(), "RECREATE");
    fOut->WriteObject(theGBRForestHandle.product(), theOutputObjectName.c_str());
    fOut->Close();
    std::cout << "Wrote output to " << theOutputFileName << std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GBRForestGetterFromDB);

#endif
