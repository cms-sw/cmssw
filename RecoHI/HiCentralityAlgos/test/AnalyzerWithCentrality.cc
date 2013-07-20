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
// $Id: AnalyzerWithCentrality.cc,v 1.10 2010/11/02 16:15:59 yilmaz Exp $
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

#include "DataFormats/HeavyIonEvent/interface/CentralityProvider.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TNtuple.h"
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

   CentralityProvider * centrality_;
   edm::Service<TFileService> fs;
   TH1D* h1;
   TNtuple* nt;
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
centrality_(0)
{
   //now do what ever initialization is needed
   h1 = fs->make<TH1D>("h1","histogram",100,0,100);
   nt = fs->make<TNtuple>("hi","hi","hf:hft:hftp:hftm:eb:ee:eep:eem:npix:et:zdc:zdcp:zdcm:bin:trig");
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
   if(!centrality_) centrality_ = new CentralityProvider(iSetup);
   centrality_->newEvent(iEvent,iSetup);

   double hf = centrality_->raw()->EtHFhitSum();
   double hft = centrality_->raw()->EtHFtowerSum();
   double hftp = centrality_->raw()->EtHFtowerSumPlus();
   double hftm = centrality_->raw()->EtHFtowerSumMinus();
   double eb = centrality_->raw()->EtEBSum();
   double ee = centrality_->raw()->EtEESum();
   double eep = centrality_->raw()->EtEESumPlus();
   double eem = centrality_->raw()->EtEESumMinus();
   double zdc = centrality_->raw()->zdcSum();
   double zdcm = centrality_->raw()->zdcSumMinus();
   double zdcp = centrality_->raw()->zdcSumPlus();
   double npix = centrality_->raw()->multiplicityPixel();
   double et = centrality_->raw()->EtMidRapiditySum();

   cout<<"Centrality variables in the event:"<<endl;
   cout<<"Total energy in HF hits : "<<hf<<endl;
   cout<<"Asymmetry of HF towers : "<<fabs(hftp-hftm)/(hftp+hftm)<<endl;
   cout<<"Total energy in EE basic clusters : "<<eep+eem<<endl;
   cout<<"Total energy in EB basic clusters : "<<eb<<endl;
   
   centrality_->print();
   
   cout<<"Centrality of the event : "<<centrality_->centralityValue()<<endl;

   int bin = centrality_->getBin();
   cout<<"a"<<endl;
   nt->Fill(hf,hft,hftp,hftm,eb,ee,eep,eem,npix,et,zdc,zdcp,zdcm,bin,1);
   cout<<"b"<<endl;

   h1->Fill(bin);
   cout<<"c"<<endl;

   int nbins = centrality_->getNbins(); 
   cout<<"d"<<endl;
   int binsize = 100./nbins;
   cout<<"e"<<endl;
   char* binName = Form("%d to % d",bin*binsize,(bin+1)*binsize);
   cout<<"The event falls into centrality bin : "<<binName<<" id : "<<bin<<endl;

   double npartMean = centrality_->NpartMean();
   double npartSigma = centrality_->NpartSigma();
   cout<<"Npart Mean : "<<npartMean<<"   Variance : "<<npartSigma<<endl;

   // or, alternatively,
   npartMean = centrality_->NpartMeanOfBin(bin);
   npartSigma = centrality_->NpartSigmaOfBin(bin);

   double ncollMean = centrality_->NcollMean();
   double ncollSigma = centrality_->NcollSigma();
   cout<<"Ncoll Mean : "<<ncollMean<<"   Variance : "<<ncollSigma<<endl;

   // or, alternatively,
   ncollMean = centrality_->NcollMeanOfBin(bin);
   ncollSigma = centrality_->NcollSigmaOfBin(bin);


   double nhardMean = centrality_->NhardMean();
   double nhardSigma = centrality_->NhardSigma();
   cout<<"Nhard Mean : "<<nhardMean<<"   Variance : "<<nhardSigma<<endl;

   double bMean = centrality_->bMean();
   double bSigma = centrality_->bSigma();
   cout<<"b Mean : "<<bMean<<"   Variance : "<<bSigma<<endl;

   bMean = centrality_->bMeanOfBin(bin);
   bSigma = centrality_->bSigmaOfBin(bin);


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
