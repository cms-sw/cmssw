// -*- C++ -*-
//
// Package:    PileupWeightProducer
// Class:      PileupWeightProducer
// 
/**\class PileupWeightProducer PileupWeightProducer.cc 

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Ricardo Vasquez Sierra,6 R-025,+41227672274,
//         Created:  Mon Nov 21 15:05:26 CET 2011
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/Utilities/interface/Lumi3DReWeighting.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h" 
#include <vector>

//
// class declaration
//

class PileupWeightProducer : public edm::EDProducer {
   public:
      explicit PileupWeightProducer(const edm::ParameterSet&);
      ~PileupWeightProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      virtual void endRun(edm::Run&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------

  bool firsttime_;
  std::string pileupMC_;
  std::string pileupData_;
  edm::Lumi3DReWeighting LumiWeightsNominal_;
  edm::Lumi3DReWeighting LumiWeightsUp_;
  edm::Lumi3DReWeighting LumiWeightsDown_;
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
PileupWeightProducer::PileupWeightProducer(const edm::ParameterSet& iConfig)
{
//   firsttime_= iConfig.existsAs<bool>("FirstTime") ? iConfig.getParameter<bool>("FirstTime") : true ;
//   pileupMC_ = iConfig.existsAs<std::string>("PileupMCFile") ? iConfig.getParameter<std::string>("PileupMCFile") : "PUMC_dist.root" ;
//   pileupData_ = iConfig.existsAs<std::string>("PileupDataFile") ? iConfig.getParameter<std::string>("PileupDataFile") : "PUData_dist.root" ;

  firsttime_ =  iConfig.getUntrackedParameter<bool>("FirstTime");
  pileupMC_ =  iConfig.getUntrackedParameter<std::string>("PileupMCFile");
  pileupData_ = iConfig.getUntrackedParameter<std::string>("PileupDataFile");

  //register your products
  
  produces<std::vector<float> >( "pileupWeights" ).setBranchAlias( "pileupWeights" );
  
  

  if ( firsttime_ )
    {
      std::cout<< " Initializing with the following files MC: " << pileupMC_ << " data: " << pileupData_ << std::endl;
      LumiWeightsNominal_.weight3D_set( pileupMC_, pileupData_, "pileup", "pileup");
      LumiWeightsUp_.weight3D_set( pileupMC_, pileupData_, "pileup", "pileup");
      LumiWeightsDown_.weight3D_set( pileupMC_, pileupData_, "pileup", "pileup");

      LumiWeightsNominal_.weight3D_init(1.0);
      LumiWeightsUp_.weight3D_init(1.08);
      LumiWeightsDown_.weight3D_init(0.92);
    }
  else 
    {
      std::cout<< " Initializing with Weight3D.root " << std::endl; 
      LumiWeightsNominal_.weight3D_init("Weight3D.root");
      LumiWeightsUp_.weight3D_init("Weight3DscaleUp.root");
      LumiWeightsDown_.weight3D_init("Weight3DscaleDown.root");
    }



  
}


PileupWeightProducer::~PileupWeightProducer()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}

// ------------ method called to produce the data  ------------
void
PileupWeightProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   using namespace edm;
   std::auto_ptr<std::vector<float> > pileupWeights( new std::vector<float> );

   edm::EventBase* iEventB = dynamic_cast<edm::EventBase*>(&iEvent);
   double nominalWeight3D = LumiWeightsNominal_.weight3D( (*iEventB) );
   double weight3DUp = LumiWeightsUp_.weight3D( (*iEventB) );
   double weight3DDown = LumiWeightsDown_.weight3D( (*iEventB) );

   pileupWeights->push_back( nominalWeight3D );
   pileupWeights->push_back( weight3DUp );
   pileupWeights->push_back( weight3DDown );

   iEvent.put(pileupWeights, "pileupWeights");
   
}

// ------------ method called once each job just before starting event loop  ------------
void 
PileupWeightProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PileupWeightProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
PileupWeightProducer::beginRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
PileupWeightProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
PileupWeightProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
PileupWeightProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PileupWeightProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PileupWeightProducer);
