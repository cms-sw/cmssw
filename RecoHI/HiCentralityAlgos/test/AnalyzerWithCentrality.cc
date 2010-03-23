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
// $Id: AnalyzerWithCentrality.cc,v 1.4 2010/03/20 21:17:51 yilmaz Exp $
//
//


// system include files
#include <memory>
#include <iostream>

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
using namespace std;

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

   cout<<"Centrality variables in the event:"<<endl;
   cout<<"Total energy in HF hits : "<<hf<<endl;
   cout<<"Asymmetry of HF towers : "<<fabs(hftp-hftm)/(hftp+hftm)<<endl;
   cout<<"Total energy in EE basic clusters : "<<eep+eem<<endl;
   cout<<"Total energy in EB basic clusters : "<<eb<<endl;
   
   int bin = cbins_->getBin(hf);
   int nbins = cbins_->getNbins(); 
   int binsize = 100/nbins;
   char* binName = Form("%d to % d",bin*binsize,(bin+1)*binsize);
   cout<<"The event falls into centrality bin : "<<binName<<" id : "<<bin<<endl;

   double npartMean = cbins_->NpartMean(hf);
   double npartSigma = cbins_->NpartSigma(hf);
   cout<<"Npart Mean : "<<npartMean<<"   Variance : "<<npartSigma<<endl;

   // or, alternatively,
   npartMean = cbins_->NpartMeanOfBin(bin);
   npartSigma = cbins_->NpartSigmaOfBin(bin);

   double ncollMean = cbins_->NcollMean(hf);
   double ncollSigma = cbins_->NcollSigma(hf);
   cout<<"Ncoll Mean : "<<ncollMean<<"   Variance : "<<ncollSigma<<endl;

   // or, alternatively,
   ncollMean = cbins_->NcollMeanOfBin(bin);
   ncollSigma = cbins_->NcollSigmaOfBin(bin);


   double nhardMean = cbins_->NhardMean(hf);
   double nhardSigma = cbins_->NhardSigma(hf);
   cout<<"Nhard Mean : "<<nhardMean<<"   Variance : "<<nhardSigma<<endl;

   double bMean = cbins_->bMean(hf);
   double bSigma = cbins_->bSigma(hf);
   cout<<"b Mean : "<<bMean<<"   Variance : "<<bSigma<<endl;

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
