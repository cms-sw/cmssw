// -*- C++ -*-
//
// Package:    ElectronIsoAnalyzer
// Class:      ElectronIsoAnalyzer
// 
/**\class ElectronIsoAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Daniele Benedetti



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEffectiveArea.h"

#include <iostream>
#include <string>
#include <map>

//
// class decleration
//

using namespace edm;
using namespace reco;
using namespace std;
class ElectronIsoAnalyzer : public edm::EDAnalyzer {
   public:
      explicit ElectronIsoAnalyzer(const edm::ParameterSet&);
      ~ElectronIsoAnalyzer();
  //

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
 

  ParameterSet conf_;

 
  unsigned int ev;
      // ----------member data ---------------------------
  bool verbose_;
  edm::InputTag inputTagGsfElectrons_;
  edm::InputTag rhoIsoInputTag_;
  std::vector<edm::InputTag> inputTagIsoValElectrons_;   
  typedef std::vector< edm::Handle< edm::ValueMap<double> > > IsoValues; 
  ElectronEffectiveArea::ElectronEffectiveAreaTarget effAreaTarget_;
  ElectronEffectiveArea::ElectronEffectiveAreaType   effAreaGammaPlusNeutralHad_;
  std::string rho_; 
  std::string deltaR_;


  // Control histos
  TH1F* chargedBarrel_   ; 
  TH1F* photonBarrel_    ; 
  TH1F* neutralBarrel_   ; 
    
  TH1F* chargedEndcaps_  ; 
  TH1F* photonEndcaps_   ; 
  TH1F* neutralEndcaps_  ; 

  TH1F* sumBarrel_       ;
  TH1F* sumEndcaps_      ;

  TH1F* missHitsBarrel_  ;
  TH1F* missHitsEndcap_  ;

  TH1F* sumCorrBarrel_   ;
  TH1F* sumCorrEndcaps_  ;

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
ElectronIsoAnalyzer::ElectronIsoAnalyzer(const edm::ParameterSet& iConfig):
  conf_(iConfig)

{

  verbose_                    = iConfig.getUntrackedParameter<bool>("verbose", false);
  inputTagGsfElectrons_       = iConfig.getParameter<edm::InputTag>("Electrons");
  inputTagIsoValElectrons_    = iConfig.getParameter< std::vector<edm::InputTag> >("IsoValElectrons");   
  deltaR_                     = iConfig.getParameter<std::string>("deltaR");  
  std::string eaTarget        = iConfig.getParameter<std::string>("effectiveAreaTarget");
  rhoIsoInputTag_             = iConfig.getParameter<edm::InputTag>("rhoIsoInputTag");

  if      (eaTarget == "NoCorr")     effAreaTarget_ = ElectronEffectiveArea::kEleEANoCorr;
  else if (eaTarget == "Data2011")   effAreaTarget_ = ElectronEffectiveArea::kEleEAData2011;   // default for HZZ 
  else if (eaTarget == "Data2012")   effAreaTarget_ = ElectronEffectiveArea::kEleEAData2012;   // default for HWW
  else if (eaTarget == "Summer11MC") effAreaTarget_ = ElectronEffectiveArea::kEleEASummer11MC;
  else if (eaTarget == "Fall11MC")   effAreaTarget_ = ElectronEffectiveArea::kEleEAFall11MC;
  else throw cms::Exception("Configuration") << "Unknown effective area " << eaTarget << "\n";

  if (deltaR_ == "03") {
    effAreaGammaPlusNeutralHad_ = ElectronEffectiveArea::kEleGammaAndNeutralHadronIso03;
  } else if (deltaR_ == "04") {
    effAreaGammaPlusNeutralHad_ = ElectronEffectiveArea::kEleGammaAndNeutralHadronIso04;
  } else throw cms::Exception("Configuration") << "Unsupported deltaR " << deltaR_ << "\n";

  
  edm::Service<TFileService> fs;
  chargedBarrel_    = fs->make<TH1F>("chargedBarrel",";Sum pT/pT" ,100,0,4);
  photonBarrel_     = fs->make<TH1F>("photonBarrel",";Sum pT/pT", 100,0,4);
  neutralBarrel_    = fs->make<TH1F>("neutralBarrel",";Sum pT/pT", 100,0,4);
  		      
  chargedEndcaps_   = fs->make<TH1F>("chargedEndcaps",";Sum pT/pT",100,0,4);
  photonEndcaps_    = fs->make<TH1F>("photonEndcaps",";Sum pT/pT",100,0,4);
  neutralEndcaps_   = fs->make<TH1F>("neutralEndcaps",";Sum pT/pT",100,0,4);
  		      
  sumBarrel_        = fs->make<TH1F>("allbarrel",";Sum pT/pT",100,0,4);
  sumEndcaps_       = fs->make<TH1F>("allendcaps",";Sum pT/pT",100,0,4);

  sumCorrBarrel_        = fs->make<TH1F>("allcorrbarrel",";Sum pT/pT",100,0,4);
  sumCorrEndcaps_       = fs->make<TH1F>("allcorrendcaps",";Sum pT/pT",100,0,4);


  missHitsBarrel_   = fs->make<TH1F>("missHitsBarrel_","",10,0,10);
  missHitsEndcap_   = fs->make<TH1F>("missHitsEndcap_","",10,0,10);

}


ElectronIsoAnalyzer::~ElectronIsoAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
ElectronIsoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  

 
  Handle<GsfElectronCollection> theEGammaCollection;
  iEvent.getByLabel(inputTagGsfElectrons_,theEGammaCollection);
  const GsfElectronCollection theEGamma = *(theEGammaCollection.product());

  // rho for isolation
  edm::Handle<double> rhoIso_h;
  iEvent.getByLabel(rhoIsoInputTag_, rhoIso_h);
  double rhoIso = *(rhoIso_h.product());


  unsigned nTypes=3;
  IsoValues  electronIsoValues(nTypes);
  
  for (size_t j = 0; j<inputTagIsoValElectrons_.size(); ++j) {
    iEvent.getByLabel(inputTagIsoValElectrons_[j], electronIsoValues[j]);
  }

  unsigned nele=theEGammaCollection->size();
  
  for(unsigned iele=0; iele<nele;++iele) {
    reco::GsfElectronRef myElectronRef(theEGammaCollection,iele);
    
    const IsoValues * myIsoValues = &electronIsoValues;
    
    double charged =  (*(*myIsoValues)[0])[myElectronRef];
    double photon = (*(*myIsoValues)[1])[myElectronRef];
    double neutral = (*(*myIsoValues)[2])[myElectronRef];
    
    float abseta = fabs(myElectronRef->superCluster()->eta());
    
    float eff_area_phnh = ElectronEffectiveArea::GetElectronEffectiveArea(effAreaGammaPlusNeutralHad_, abseta, effAreaTarget_);
    
    double myRho = max<double>(0.d,rhoIso);
    
    float myPfIsoPuCorr = charged + max<float>(0.f, (photon+neutral) - eff_area_phnh*myRho);


    if(verbose_) { 
      
      std::cout << " run " << iEvent.id().run() << " lumi " << iEvent.id().luminosityBlock() << " event " << iEvent.id().event();
      std::cout << " pt " <<  myElectronRef->pt() << " eta " << myElectronRef->eta() << " phi " << myElectronRef->phi() 
		<< " charge " << myElectronRef->charge()<< " : " << std::endl;;
      
      // print values also from alternate code
      std::cout << " ChargedIso " << charged << std::endl;
      std::cout << " PhotonIso " <<  photon << std::endl;
      std::cout << " NeutralHadron Iso " << neutral << std::endl;

    }
    if(myElectronRef->isEB()) {
      chargedBarrel_ ->Fill(charged/myElectronRef->pt());
      photonBarrel_->Fill(photon/myElectronRef->pt());
      neutralBarrel_->Fill(neutral/myElectronRef->pt());
      sumBarrel_->Fill((charged+photon+neutral)/myElectronRef->pt());
      sumCorrBarrel_->Fill(myPfIsoPuCorr/myElectronRef->pt());
      missHitsBarrel_->Fill(myElectronRef->gsfTrack()->trackerExpectedHitsInner().numberOfHits());
      
    } else {
      chargedEndcaps_ ->Fill(charged/myElectronRef->pt());
      photonEndcaps_->Fill(photon/myElectronRef->pt());
      neutralEndcaps_->Fill(neutral/myElectronRef->pt());
      sumEndcaps_->Fill((charged+photon+neutral)/myElectronRef->pt());
      sumCorrEndcaps_->Fill(myPfIsoPuCorr/myElectronRef->pt());
      missHitsEndcap_->Fill(myElectronRef->gsfTrack()->trackerExpectedHitsInner().numberOfHits());
    }
  }

}
// ------------ method called once each job just before starting event loop  ------------
void 
ElectronIsoAnalyzer::beginJob(const edm::EventSetup&)
{

  ev = 0;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ElectronIsoAnalyzer::endJob() {
  cout << " endJob:: #events " << ev << endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronIsoAnalyzer);
