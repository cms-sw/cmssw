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

  // No longer needed. e/g recommendation (04/04/12)
  //  inputTagIsoValElectronsNoPFId_ = iConfig.getParameter< std::vector<edm::InputTag> >("IsoValElectronNoPF");
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

  // No longer needed. e/g recommendation (04/04/12)
  //  IsoDepositVals electronIsoValNoPFId(nTypes);

  for (size_t j = 0; j<inputTagIsoValElectronsPFId_.size(); ++j) {
    iEvent.getByLabel(inputTagIsoValElectronsPFId_[j], electronIsoValPFId[j]);
  }


  // No longer needed. e/g recommendation (04/04/12)

  //for (size_t j = 0; j<inputTagIsoValElectronsPFId_.size(); ++j) {
  //  iEvent.getByLabel(inputTagIsoValElectronsNoPFId_[j], electronIsoValNoPFId[j]);
  //}
  //
  //
  // PFCandidateMap
  // edm::Handle<edm::ValueMap<reco::PFCandidatePtr> >ValMapH;
  //  iEvent.getByLabel(inputTagPFCandidateMap_,ValMapH);
  //  const edm::ValueMap<reco::PFCandidatePtr> & myValMap(*ValMapH); 
  //  for (size_t j = 0; j<inputTagIsoValElectronsPFId_.size(); ++j) {
  //    iEvent.getByLabel(inputTagIsoValElectronsNoPFId_[j], electronIsoValNoPFId[j]);
  //  }


  // Electrons - from reco 
  unsigned nele=gsfElectronH->size();

  for(unsigned iele=0; iele<nele;++iele) {
    reco::GsfElectronRef myElectronRef(gsfElectronH,iele);


    //    No longer needed. e/g recommendation (04/04/12)
    // Get the PFCandidate
    //    const reco::PFCandidatePtr pfElePtr(myValMap[myElectronRef]); 
    //    unsigned pfId= pfElePtr.isNonnull();
    //    unsigned pfId=(myElectronRef->passingMvaPreselection()) ? 1 : 0 ;
    // Get the PFCandidate
    //    const reco::PFCandidatePtr pfElePtr(myValMap[myElectronRef]); 
   //    unsigned pfId= pfElePtr.isNonnull();
   //    unsigned pfId=(myElectronRef->passingMvaPreselection()) ? 1 : 0 ;
   // const IsoDepositVals * electronIsoVals =  (pfId) ? &electronIsoValPFId : &electronIsoValNoPFId ;

    const IsoDepositVals * electronIsoVals = &electronIsoValPFId;

    double charged =  (*(*electronIsoVals)[0])[myElectronRef];
    double photon = (*(*electronIsoVals)[1])[myElectronRef];
    double neutral = (*(*electronIsoVals)[2])[myElectronRef];

    std::cout << " run " << iEvent.id().run() << " lumi " << iEvent.id().luminosityBlock() << " event " << iEvent.id().event();
    std::cout << " pt " <<  myElectronRef->pt() << " eta " << myElectronRef->eta() << " phi " << myElectronRef->phi() << " charge " << myElectronRef->charge()<< " : ";
    std::cout << " ChargedIso " << charged ;
    std::cout << " PhotonIso " <<  photon ;
    std::cout << " NeutralHadron Iso " << neutral << std::endl;
    
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


