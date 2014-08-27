////////////////////////////////////////////////////////////////////////////////
//
// L6SLBCorrector
// --------------
//
//           25/10/2009   Hauke Held             <hauke.held@cern.ch>
//                        Philipp Schieferdecker <philipp.schieferdecker@cern.ch
////////////////////////////////////////////////////////////////////////////////

#include "JetMETCorrections/Algorithms/interface/L6SLBCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TrackReco/interface/Track.h"


#include <string>


using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
L6SLBCorrector::L6SLBCorrector(const JetCorrectorParameters& fParam, const edm::ParameterSet& fConfig)
  : addMuonToJet_(fConfig.getParameter<bool>("addMuonToJet"))
  , srcBTagInfoElec_(fConfig.getParameter<edm::InputTag>("srcBTagInfoElectron"))
  , srcBTagInfoMuon_(fConfig.getParameter<edm::InputTag>("srcBTagInfoMuon"))
  , corrector_(0)
{
  vector<JetCorrectorParameters> vParam;
  vParam.push_back(fParam);
  corrector_ = new FactorizedJetCorrector(vParam);
}

//______________________________________________________________________________
L6SLBCorrector::~L6SLBCorrector()
{
  delete corrector_;
} 


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
double L6SLBCorrector::correction(const LorentzVector& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(LorentzVector), event required!";
  return 1.0;
}


//______________________________________________________________________________
double L6SLBCorrector::correction(const reco::Jet& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(reco::Jet), event required!";
  return 1.0;
}


//______________________________________________________________________________
double L6SLBCorrector::correction(const reco::Jet& fJet,
				  const edm::RefToBase<reco::Jet>& refToRawJet,
				  const edm::Event& fEvent, 
				  const edm::EventSetup& fSetup) const
{
  corrector_->setJetPt(fJet.pt());
  corrector_->setJetEta(fJet.eta());
  corrector_->setJetPhi(fJet.phi());
  corrector_->setJetE(fJet.energy());
  
  edm::Handle< vector<reco::SoftLeptonTagInfo> > muoninfos;
  fEvent.getByLabel(srcBTagInfoMuon_,muoninfos);
  
  const reco::SoftLeptonTagInfo& sltMuon =
    (*muoninfos)[getBTagInfoIndex(refToRawJet,*muoninfos)];
  if (sltMuon.leptons()>0) {
    edm::RefToBase<reco::Track> trackRef = sltMuon.lepton(0);
    corrector_->setLepPx(trackRef->px());
    corrector_->setLepPy(trackRef->py());
    corrector_->setLepPz(trackRef->pz());
    corrector_->setAddLepToJet(addMuonToJet_);
    return corrector_->getCorrection();
  }
  else {
    edm::Handle< vector<reco::SoftLeptonTagInfo> > elecinfos;
    fEvent.getByLabel(srcBTagInfoElec_,elecinfos);
    const reco::SoftLeptonTagInfo& sltElec =
      (*elecinfos)[getBTagInfoIndex(refToRawJet,*elecinfos)];
    if (sltElec.leptons()>0) {
      edm::RefToBase<reco::Track> trackRef = sltElec.lepton(0);
      corrector_->setLepPx(trackRef->px());
      corrector_->setLepPy(trackRef->py());
      corrector_->setLepPz(trackRef->pz());
      corrector_->setAddLepToJet(false);
      return corrector_->getCorrection();
    }
  }
  return 1.0;
}


////////////////////////////////////////////////////////////////////////////////
// implementation of private member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
int L6SLBCorrector::getBTagInfoIndex(const edm::RefToBase<reco::Jet>& refToRawJet,
				     const vector<reco::SoftLeptonTagInfo>& tags)
  const
{
  for (unsigned int i=0;i<tags.size();i++)
    if (tags[i].jet().get()==refToRawJet.get()) return i;
  return -1;
}
