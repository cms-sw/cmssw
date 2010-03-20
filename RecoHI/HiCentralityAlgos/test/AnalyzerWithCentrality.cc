// -*- C++ -*-
//
// Package:    AnalyzerWithCentrality
// Class:      AnalyzerWithCentrality
// 
/**\class AnalyzerWithCentrality AnalyzerWithCentrality.cc RecoHI/AnalyzerWithCentrality/src/AnalyzerWithCentrality.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Mon Mar  1 17:18:04 EST 2010
// $Id: AnalyzerWithCentrality.cc,v 1.3 2010/03/20 14:42:44 yilmaz Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TH1D.h"

//
// class declaration
//

class AnalyzerWithCentrality : public edm::EDAnalyzer {
   public:
      explicit AnalyzerWithCentrality(const edm::ParameterSet&);
      ~AnalyzerWithCentrality();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

   const CentralityBins * cbins_;
   edm::Service<TFileService> fs;
   TH1D* h1;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
AnalyzerWithCentrality::AnalyzerWithCentrality(const edm::ParameterSet& iConfig) : 
cbins_(0)
{
   //now do what ever initialization is needed
   h1 = fs->make<TH1D>("h1","histogram",100,0,100);
}


AnalyzerWithCentrality::~AnalyzerWithCentrality()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
AnalyzerWithCentrality::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   if(!cbins_) cbins_ = getCentralityBinsFromDB(iSetup);

   edm::Handle<reco::Centrality> cent;
   iEvent.getByLabel(edm::InputTag("hiCentrality"),cent);

   double hf = cent->EtHFhitSum();
   double hftp = cent->EtHFtowerSumPlus();
   double hftm = cent->EtHFtowerSumMinus();
   double eb = cent->EtEBSum();
   double eep = cent->EtEESumPlus();
   double eem = cent->EtEESumMinus();

   hftp = hftp+hftm;
   eb = eb/(eep+eem);
 
   int bin = cbins_->getBin(hf);

   double npartMean = cbins_->NpartMean(hf);
   double npartSigma = cbins_->NpartSigma(hf);

   // or, alternatively,
   npartMean = cbins_->NpartMeanOfBin(bin);
   npartSigma = cbins_->NpartSigmaOfBin(bin);

   double ncollMean = cbins_->NcollMean(hf);
   double ncollSigma = cbins_->NcollSigma(hf);

   // or, alternatively,
   ncollMean = cbins_->NcollMeanOfBin(bin);
   ncollSigma = cbins_->NcollSigmaOfBin(bin);

   double bMean = cbins_->bMean(hf);
   double bSigma = cbins_->bSigma(hf);

   bMean = cbins_->bMeanOfBin(bin);
   bSigma = cbins_->bSigmaOfBin(bin);


}


// ------------ method called once each job just before starting event loop  ------------
void 
AnalyzerWithCentrality::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
AnalyzerWithCentrality::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(AnalyzerWithCentrality);
