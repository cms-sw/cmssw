#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "CommonTools/ParticleFlow/test/PFIsoReaderDemo.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

PFIsoReaderDemo::PFIsoReaderDemo(const edm::ParameterSet& iConfig)
{
  inputTagGsfElectrons_ = iConfig.getParameter<edm::InputTag>("Electrons");
  inputTagPFCandidateMap_ = iConfig.getParameter< edm::InputTag>("PFCandidateMap");   
  inputTagIsoDepElectrons_ = iConfig.getParameter< std::vector<edm::InputTag> >("IsoDepElectron");
  inputTagIsoValElectronsNoPFId_ = iConfig.getParameter< std::vector<edm::InputTag> >("IsoValElectronNoPF");
  inputTagIsoValElectronsPFId_   = iConfig.getParameter< std::vector<edm::InputTag> >("IsoValElectronPF");   

  // Control plots
  TFileDirectory dir = fileservice_->mkdir("PFISO");
  chargedBarrel_    = dir.make<TH1F>("chargedBarrel",";Sum pT/pT" ,100,0,4);
  photonBarrel_     = dir.make<TH1F>("photonBarrel",";Sum pT/pT", 100,0,4);
  neutralBarrel_    = dir.make<TH1F>("neutralBarrel",";Sum pT/pT", 100,0,4);
  		      
  chargedEndcaps_   = dir.make<TH1F>("chargedEndcaps",";Sum pT/pT",100,0,4);
  photonEndcaps_    = dir.make<TH1F>("photonEndcaps",";Sum pT/pT",100,0,4);
  neutralEndcaps_   = dir.make<TH1F>("neutralEndcaps",";Sum pT/pT",100,0,4);
  		      
  sumBarrel_        = dir.make<TH1F>("allbarrel",";Sum pT/pT",100,0,4);
  sumEndcaps_       = dir.make<TH1F>("allendcaps",";Sum pT/pT",100,0,4);
}

PFIsoReaderDemo::~PFIsoReaderDemo(){;}

void 
PFIsoReaderDemo::beginRun(edm::Run const&, edm::EventSetup const& ){;}

void PFIsoReaderDemo::analyze(const edm::Event & iEvent,const edm::EventSetup & c)
{
  edm::Handle<reco::GsfElectronCollection> gsfElectronH;
  bool found=iEvent.getByLabel(inputTagGsfElectrons_,gsfElectronH);
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get GsfElectrons: "
       <<inputTagGsfElectrons_<<std::endl;
    edm::LogError("PFIsoReaderDemo")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }  

  // get the iso deposits. 3 (charged hadrons, photons, neutral hadrons)
  unsigned nTypes=3;
  IsoDepositMaps electronIsoDep(nTypes);

  for (size_t j = 0; j<inputTagIsoDepElectrons_.size(); ++j) {
    iEvent.getByLabel(inputTagIsoDepElectrons_[j], electronIsoDep[j]);
  }

  IsoDepositVals electronIsoValPFId(nTypes);
  IsoDepositVals electronIsoValNoPFId(nTypes);

  for (size_t j = 0; j<inputTagIsoValElectronsPFId_.size(); ++j) {
    iEvent.getByLabel(inputTagIsoValElectronsPFId_[j], electronIsoValPFId[j]);
  }
  for (size_t j = 0; j<inputTagIsoValElectronsPFId_.size(); ++j) {
    iEvent.getByLabel(inputTagIsoValElectronsNoPFId_[j], electronIsoValNoPFId[j]);
  }

  // PFCandidateMap
  edm::Handle<edm::ValueMap<reco::PFCandidatePtr> >ValMapH;
  iEvent.getByLabel(inputTagPFCandidateMap_,ValMapH);
  const edm::ValueMap<reco::PFCandidatePtr> & myValMap(*ValMapH); 


  // Electrons - from reco 
  unsigned nele=gsfElectronH->size();

  for(unsigned iele=0; iele<nele;++iele) {
    reco::GsfElectronRef myElectronRef(gsfElectronH,iele);

    // Get the PFCandidate
    const reco::PFCandidatePtr pfElePtr(myValMap[myElectronRef]); 

    unsigned pfId= pfElePtr.isNonnull();

    const IsoDepositVals * electronIsoVals =  (pfId) ? &electronIsoValPFId : &electronIsoValNoPFId ;
    double charged =  (*(*electronIsoVals)[0])[myElectronRef];
    double photon = (*(*electronIsoVals)[1])[myElectronRef];
    double neutral = (*(*electronIsoVals)[2])[myElectronRef];
    std::cout << " GsfElectron pT, eta, phi, charge " << myElectronRef->pt() << " " << myElectronRef->eta() << " " << myElectronRef->phi() << " " << myElectronRef->charge() << std::endl; 
    std::cout << " Charged Iso " << charged << std::endl;
    std::cout << " Photon Iso " <<  photon << std::endl;
    std::cout << " Neutral Hadron Iso " << neutral << std::endl;
    
    if(myElectronRef->isEB()) {
      chargedBarrel_ ->Fill(charged/myElectronRef->pt());
      photonBarrel_->Fill(photon/myElectronRef->pt());
      neutralBarrel_->Fill(neutral/myElectronRef->pt());
      sumBarrel_->Fill((charged+photon+neutral)/myElectronRef->pt());
    } else {
      chargedEndcaps_ ->Fill(charged/myElectronRef->pt());
      photonEndcaps_->Fill(photon/myElectronRef->pt());
      neutralEndcaps_->Fill(neutral/myElectronRef->pt());
      sumEndcaps_->Fill((charged+photon+neutral)/myElectronRef->pt());
    }
      
  }
}
  

DEFINE_FWK_MODULE(PFIsoReaderDemo);


