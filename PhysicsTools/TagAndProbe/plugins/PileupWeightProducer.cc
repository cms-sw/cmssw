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
#include <vector>

#include "TFile.h"
#include "TH1F.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h" 

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
      
   // ----------member data ---------------------------
   std::string pileupData_;
   TH1F* h_pileupData_;
   TFile* dataFile_;
   TH1F* h_pileupMC_;
   TH1F* h_pileupRatio_;

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
  double Summer2012[60] = {
    //Pile Up For S10
    2.560E-06,
    5.239E-06,
    1.420E-05,
    5.005E-05,
    1.001E-04,
    2.705E-04,
    1.999E-03,
    6.097E-03,
    1.046E-02,
    1.383E-02,
    1.685E-02,
    2.055E-02,
    2.572E-02,
    3.262E-02,
    4.121E-02,
    4.977E-02,
    5.539E-02,
    5.725E-02,
    5.607E-02,
    5.312E-02,
    5.008E-02,
    4.763E-02,
    4.558E-02,
    4.363E-02,
    4.159E-02,
    3.933E-02,
    3.681E-02,
    3.406E-02,
    3.116E-02,
    2.818E-02,
    2.519E-02,
    2.226E-02,
    1.946E-02,
    1.682E-02,
    1.437E-02,
    1.215E-02,
    1.016E-02,
    8.400E-03,
    6.873E-03,
    5.564E-03,
    4.457E-03,
    3.533E-03,
    2.772E-03,
    2.154E-03,
    1.656E-03,
    1.261E-03,
    9.513E-04,
    7.107E-04,
    5.259E-04,
    3.856E-04,
    2.801E-04,
    2.017E-04,
    1.439E-04,
    1.017E-04,
    7.126E-05,
    4.948E-05,
    3.405E-05,
    2.322E-05,
    1.570E-05,
    5.005E-06
  };   

  
  pileupData_ = iConfig.getUntrackedParameter<std::string>("PileupDataFile");
  TFile* dataFile_ = new TFile(pileupData_.c_str(), "READ");
  h_pileupData_ = new TH1F(  *(static_cast<TH1F*>(dataFile_->Get( "pileup" )->Clone() )) );
  
  h_pileupMC_ = new TH1F("pileupMC","Generated pileup distribution (i.e., MC)",60,0.,60);
  for (int i=1;i<=60;i++)  {
    h_pileupMC_->SetBinContent(i,Summer2012[i-1]);
  }
  
  h_pileupData_->Scale( 1.0/ h_pileupData_->Integral() );
  h_pileupMC_->Scale( 1.0/ h_pileupMC_->Integral() );
  h_pileupRatio_ = new TH1F( *(h_pileupData_)) ;
  h_pileupRatio_->Divide(h_pileupMC_);
  
  //register your products
  
  produces<std::vector<float> >( "pileupWeights" ).setBranchAlias( "pileupWeights" );

}


PileupWeightProducer::~PileupWeightProducer()
{
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)
  delete h_pileupData_;
  delete h_pileupMC_;
  delete h_pileupRatio_;
//   dataFile_ -> Close();
//   delete dataFile_;
}

// ------------ method called to produce the data  ------------
void
PileupWeightProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   using namespace edm;
   std::auto_ptr<std::vector<float> > pileupWeights( new std::vector<float> );
   
   // MC Pileup Summary Info
   float mcPUtrueInteractions = 0;
   const edm::InputTag PileupSrc("addPileupInfo");
   edm::Handle<std::vector< PileupSummaryInfo > >  PupInfo;
   iEvent.getByLabel(PileupSrc, PupInfo);
   std::vector<PileupSummaryInfo>::const_iterator PVI;
   int ctid = 0;
   for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
     if (ctid>2) break;
     if(PVI->getBunchCrossing() == 0) mcPUtrueInteractions = PVI->getTrueNumInteractions();
     ctid++;
   }
   
   float nominalWeight = h_pileupRatio_->GetBinContent(int(mcPUtrueInteractions+0.01)+1); 
   pileupWeights->push_back( nominalWeight );

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
