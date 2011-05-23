#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoParticleFlow/PFProducer/test/PFIsoReader.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

PFIsoReader::PFIsoReader(const edm::ParameterSet& iConfig)
{
  inputTagGsfElectrons_ = iConfig.getParameter<edm::InputTag>("Electrons");
  inputTagPhotons_ = iConfig.getParameter<edm::InputTag>("Photons");
  inputTagPFCandidates_ = iConfig.getParameter<edm::InputTag>("PFCandidates");

  useValueMaps_ = iConfig.getParameter<bool>("useEGPFValueMaps");
  inputTagValueMapPhotons_   = iConfig.getParameter<edm::InputTag>("ElectronValueMap");  
  inputTagValueMapElectrons_  = iConfig.getParameter<edm::InputTag>("PhotonValueMap");  

  inputTagElectronIsoDeposits_ = iConfig.getParameter<std::vector<edm::InputTag> >("ElectronIsoDeposits");
  inputTagPhotonIsoDeposits_ = iConfig.getParameter<std::vector<edm::InputTag> >("PhotonIsoDeposits");

}

PFIsoReader::~PFIsoReader(){;}

void 
PFIsoReader::beginRun(edm::Run const&, edm::EventSetup const& ){;}

void PFIsoReader::analyze(const edm::Event & iEvent,const edm::EventSetup & c)
{
  edm::Handle<reco::PFCandidateCollection> pfCandidatesH;
  bool found=iEvent.getByLabel(inputTagPFCandidates_,pfCandidatesH);
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get PFCandidates: "
       <<inputTagPFCandidates_<<std::endl;
    edm::LogError("PFIsoReader")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }  


//  edm::Handle<reco::GsfElectronCollection> gsfElectronH;
//  found=iEvent.getByLabel(inputTagGsfElectrons_,gsfElectronH);
//  if(!found ) {
//    std::ostringstream  err;
//    err<<" cannot get GsfElectrons: "
//       <<inputTagGsfElectrons_<<std::endl;
//    edm::LogError("PFIsoReader")<<err.str();
//    throw cms::Exception( "MissingProduct", err.str());
//  }  

  edm::Handle<reco::PhotonCollection> photonH;
  found=iEvent.getByLabel(inputTagPhotons_,photonH);
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get Photonss: "
       <<inputTagPhotons_<<std::endl;
    edm::LogError("PFIsoReader")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }  

  // Get the value maps

  //  edm::Handle<edm::ValueMap<reco::PFCandidatePtr> > electronValMapH;
  //  const edm::ValueMap<reco::PFCandidatePtr> & myElecValMap(*electronValMapH);
  
  edm::Handle<edm::ValueMap<reco::PFCandidatePtr> > photonValMapH;
  iEvent.getByLabel(inputTagValueMapPhotons_,photonValMapH);
  const edm::ValueMap<reco::PFCandidatePtr> & myPhotonValMap(*photonValMapH);  

  // get the iso deposits
  //  IsoDepositMaps electronIsoDep(inputTagElectronIsoDeposits_.size());
  IsoDepositMaps photonIsoDep(inputTagPhotonIsoDeposits_.size());   
  
//  for (size_t j = 0; j<inputTagElectronIsoDeposits_.size(); ++j) {
//    std::cout << iEvent.getByLabel(inputTagElectronIsoDeposits_[j], electronIsoDep[j]) << std::endl;
//  }
  for (size_t j = 0; j<inputTagPhotonIsoDeposits_.size(); ++j) {
    iEvent.getByLabel(inputTagPhotonIsoDeposits_[j], photonIsoDep[j]) ;
  }

  unsigned nphot=photonH->size();
  for(unsigned iphot=0; iphot<nphot;++iphot) {
    reco::PhotonRef myPhotRef(photonH,iphot);
    const reco::PFCandidatePtr & pfPhotPtr(myPhotonValMap[myPhotRef]);
    std::cout << "TEST " << pfPhotPtr->sourceCandidatePtr(0).key() << " " << pfPhotPtr.key() << std::endl;
  }

  
  unsigned ncandidates= pfCandidatesH->size();

  for(unsigned icand=0;icand<ncandidates;++icand) {
    const reco::PFCandidate & cand((*pfCandidatesH)[icand]);
    //    std::cout << " Pdg " << cand.pdgId() << " mva " << cand.mva_nothing_gamma() << std::endl;
    if(!(cand.pdgId()==22 && cand.mva_nothing_gamma()>0)) continue;
    //    reco::PhotonRef myPhotonRef(photonH,iphot);
//    if(!myPhotonValMap.contains(myPhotonRef))     {
//      std::cout << " Did not find the PFCandidate "  << std::endl;
//      continue;
//    }
//    const reco::PFCandidatePtr myPFCandidatePtr((myPhotonValMap[myPhotonRef]));
    reco::PFCandidatePtr myPFCandidatePtr(pfCandidatesH,icand);
    std::cout << icand << std::endl;
    std::cout << myPFCandidatePtr->sourceCandidatePtr(0).key() << " " << myPFCandidatePtr->sourceCandidatePtr(0).get()->eta() << std::endl;
    std::cout << myPFCandidatePtr.key() << " " << myPFCandidatePtr->eta() << std::endl;
    unsigned nIsoDepTypes=photonIsoDep.size(); // should be 3 (charged hadrons, photons, neutral hadrons)
    for(unsigned ideptype=0; ideptype<nIsoDepTypes;++ideptype) {
      const reco::IsoDeposit & isoDep((*photonIsoDep[ideptype])[myPFCandidatePtr]);
      typedef reco::IsoDeposit::const_iterator IM;
      for(IM im=isoDep.begin(); im != isoDep.end(); ++im) {
	std::cout << "dR " << im->dR() << " val " << im->value() << std::endl;
      }
    }   
  }  
}


DEFINE_FWK_MODULE(PFIsoReader);


