#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBTDCRawInfoDumper.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <TFile.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
    
EcalTBTDCRawInfoDumper::EcalTBTDCRawInfoDumper(edm::ParameterSet const& ps)
{
  rawInfoCollection_ = ps.getParameter<std::string>("rawInfoCollection");
  rawInfoProducer_   = ps.getParameter<std::string>("rawInfoProducer");
  rootfile_          = ps.getUntrackedParameter<std::string>("rootfile","ecalTDCRawInfoPlots.root");
}

EcalTBTDCRawInfoDumper::~EcalTBTDCRawInfoDumper() {
}

//========================================================================
void
EcalTBTDCRawInfoDumper::beginJob() 
{
  //========================================================================
  h_TDCrawValue_ = new TH1F("h_TDCrawValue","TDC raw value",2048,-0.5,2047.5);
}

//========================================================================
void
EcalTBTDCRawInfoDumper::endJob() {
//========================================================================
  TFile f(rootfile_.c_str(),"RECREATE");
  h_TDCrawValue_->Write();
  f.Close();
}

void EcalTBTDCRawInfoDumper::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  // Get input
  edm::Handle<EcalTBTDCRawInfo> ecalRawTDC;  
  const EcalTBTDCRawInfo* tdcRawInfo = 0;
  //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
  e.getByLabel( rawInfoProducer_, ecalRawTDC);
  if (!ecalRawTDC.isValid()) {
    edm::LogError("EcalTBTDCRecInfoError") << "Error! can't get the product " << rawInfoCollection_.c_str() ;
  } else {
    tdcRawInfo = ecalRawTDC.product();
  }

  
  if (tdcRawInfo)
    {
      int tdcd = (*tdcRawInfo)[0].tdcValue();
      h_TDCrawValue_->Fill(tdcd);
    }
  // Create empty output
} 


