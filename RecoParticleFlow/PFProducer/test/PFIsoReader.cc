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
  inputTagValueMapPhotons_   = iConfig.getParameter<edm::InputTag>("PhotonValueMap");  
  inputTagValueMapElectrons_  = iConfig.getParameter<edm::InputTag>("ElectronValueMap");  
  inputTagValueMapMerged_  = iConfig.getParameter<edm::InputTag>("MergedValueMap");  

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


  edm::Handle<reco::GsfElectronCollection> gsfElectronH;
  found=iEvent.getByLabel(inputTagGsfElectrons_,gsfElectronH);
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get GsfElectrons: "
       <<inputTagGsfElectrons_<<std::endl;
    edm::LogError("PFIsoReader")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }  

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

  edm::Handle<edm::ValueMap<reco::PFCandidatePtr> > electronValMapH;
  found = iEvent.getByLabel(inputTagValueMapElectrons_,electronValMapH);
  const edm::ValueMap<reco::PFCandidatePtr> & myElectronValMap(*electronValMapH);
  
  std::cout << " Read Electron Value Map " << myElectronValMap.size() << std::endl;

//  edm::Handle<edm::ValueMap<reco::PFCandidatePtr> > photonValMapH;
//  found = iEvent.getByLabel(inputTagValueMapPhotons_,photonValMapH);
//   const edm::ValueMap<reco::PFCandidatePtr> & myPhotonValMap(*photonValMapH);  

  edm::Handle<edm::ValueMap<reco::PFCandidatePtr> > mergedValMapH;
  found = iEvent.getByLabel(inputTagValueMapMerged_,mergedValMapH);
  const edm::ValueMap<reco::PFCandidatePtr> & myMergedValMap(*mergedValMapH);  
  
  // get the iso deposits
  IsoDepositMaps electronIsoDep(inputTagElectronIsoDeposits_.size());
  IsoDepositMaps photonIsoDep(inputTagPhotonIsoDeposits_.size());   
  
  for (size_t j = 0; j<inputTagElectronIsoDeposits_.size(); ++j) {
    iEvent.getByLabel(inputTagElectronIsoDeposits_[j], electronIsoDep[j]);
  }
  for (size_t j = 0; j<inputTagPhotonIsoDeposits_.size(); ++j) {
    iEvent.getByLabel(inputTagPhotonIsoDeposits_[j], photonIsoDep[j]) ;
  }

  // Photons - from reco 
  unsigned nphot=photonH->size();
  std::cout<<"Photon: "<<nphot<<std::endl;
  for(unsigned iphot=0; iphot<nphot;++iphot) {
    reco::PhotonRef myPhotRef(photonH,iphot);
    //    const reco::PFCandidatePtr & pfPhotPtr(myPhotonValMap[myPhotRef]);
    const reco::PFCandidatePtr & pfPhotPtr(myMergedValMap[myPhotRef]);
    printIsoDeposits(photonIsoDep,pfPhotPtr);
  }

  // Photons - from PF Candidates 
  unsigned ncandidates= pfCandidatesH->size();
  std::cout<<"Candidates: "<<ncandidates<<std::endl;
  for(unsigned icand=0;icand<ncandidates;++icand) {
    const reco::PFCandidate & cand((*pfCandidatesH)[icand]);
    //    std::cout << " Pdg " << cand.pdgId() << " mva " << cand.mva_nothing_gamma() << std::endl;
    if(!(cand.pdgId()==22 && cand.mva_nothing_gamma()>0)) continue;

    reco::PFCandidatePtr myPFCandidatePtr(pfCandidatesH,icand);
    printIsoDeposits(photonIsoDep,myPFCandidatePtr);
  }  


  // Electrons - from reco 
  unsigned nele=gsfElectronH->size();
  std::cout<<"Electron: "<<nele<<std::endl;
  for(unsigned iele=0; iele<nele;++iele) {
    reco::GsfElectronRef myElectronRef(gsfElectronH,iele);

    if(myElectronRef->mva_e_pi()<-1) continue;
    //const reco::PFCandidatePtr & pfElePtr(myElectronValMap[myElectronRef]);
    const reco::PFCandidatePtr pfElePtr(myElectronValMap[myElectronRef]);
    printIsoDeposits(electronIsoDep,pfElePtr);
  }

  // Electrons - from PFCandidate
  nele=gsfElectronH->size();
  std::cout<<"Candidates: "<<nele<<std::endl;
  for(unsigned icand=0; icand<ncandidates;++icand) {
    const reco::PFCandidate & cand((*pfCandidatesH)[icand]);
    
    if (!(abs(cand.pdgId())==11)) continue;

    reco::PFCandidatePtr myPFCandidatePtr(pfCandidatesH,icand);
    printIsoDeposits(electronIsoDep,myPFCandidatePtr);
  }
  
  
}
  
void PFIsoReader::printIsoDeposits(const IsoDepositMaps & isodepmap, const reco::PFCandidatePtr & ptr) const {
  std::cout << " Isodeposits for " << ptr.id() << " " << ptr.key() << std::endl;
  unsigned nIsoDepTypes=isodepmap.size(); // should be 3 (charged hadrons, photons, neutral hadrons)
    for(unsigned ideptype=0; ideptype<nIsoDepTypes;++ideptype) {
      const reco::IsoDeposit & isoDep((*isodepmap[ideptype])[ptr]);
      typedef reco::IsoDeposit::const_iterator IM;
      std::cout << " Iso deposits type " << ideptype << std::endl;
      for(IM im=isoDep.begin(); im != isoDep.end(); ++im) {
	std::cout << "dR " << im->dR() << " val " << im->value() << std::endl;
      }
    }   
}


DEFINE_FWK_MODULE(PFIsoReader);


