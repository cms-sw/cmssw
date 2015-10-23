// -*- C++ -*-
//
// Package:    HiCentralityBiasFilter
// Class:      HiCentralityBiasFilter
// 
/**\class HiCentralityBiasFilter HiCentralityBiasFilter.cc yetkin/HiCentralityBiasFilter/src/HiCentralityBiasFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Tue Aug 11 12:42:25 EDT 2009
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "CLHEP/Random/RandomEngine.h"
#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"
#include "TF1.h"

using namespace std;

//
// class declaration
//

class HiCentralityBiasFilter : public edm::EDFilter {
   public:
      explicit HiCentralityBiasFilter(const edm::ParameterSet&);
      ~HiCentralityBiasFilter();

   private:
      virtual void beginJob() override ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;
      
  edm::InputTag hepmcSrc_;

   edm::Service<edm::RandomNumberGenerator> rng_;

   TF1* fBias_;
   string func_;
   vector<double> par_;

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
HiCentralityBiasFilter::HiCentralityBiasFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   hepmcSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("src", edm::InputTag("generatorSmeared"));
   func_ = iConfig.getParameter<string>("function");
   par_ = iConfig.getParameter<vector<double> >("parameters");
}


HiCentralityBiasFilter::~HiCentralityBiasFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HiCentralityBiasFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;



   CLHEP::HepRandomEngine& engine = rng_->getEngine(iEvent.streamID());

   Handle<HepMCProduct> mc;
   iEvent.getByLabel(hepmcSrc_,mc);
   const HepMC::GenEvent* evt = mc->GetEvent();

   const HepMC::HeavyIon* hi = evt->heavy_ion();
   if(!hi) return false;

   double b = hi->impact_parameter();
   double bound = fBias_->Eval(b);
   double rand = engine.flat();
   if(rand > bound) return false;

   return true;

}

// ------------ method called once each job just before starting event loop  ------------
void 
HiCentralityBiasFilter::beginJob()
{

   fBias_ = new TF1("fBias",func_.data(),0,20);

   for(size_t ip = 0; ip < par_.size(); ++ip){
      fBias_->SetParameter(ip,par_[ip]);
   }

   double maxpoint = fBias_->GetMaximum(-0.1,20);
   if(maxpoint < 0.9) throw cms::Exception("HeavyIonCentralityBias")<<"Input bias function is not optimized. Peak value is "<<maxpoint<<" which is required to be close to 1. Please fix the parameters before production."<<endl;
   
   
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HiCentralityBiasFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiCentralityBiasFilter);
