////////////////////////////////////////////////////////////////////////////////
//
// L6SLBCorrectorImpl
// --------------
//
//           25/10/2009   Hauke Held             <hauke.held@cern.ch>
//                        Philipp Schieferdecker <philipp.schieferdecker@cern.ch
////////////////////////////////////////////////////////////////////////////////

#include "JetMETCorrections/Algorithms/interface/L6SLBCorrectorImpl.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/TrackReco/interface/Track.h"


#include <string>


using namespace std;

L6SLBCorrectorImplMaker::L6SLBCorrectorImplMaker(edm::ParameterSet const& fConfig,
						 edm::ConsumesCollector fCollector)
  : JetCorrectorImplMakerBase(fConfig)
  , elecToken_(fCollector.consumes<std::vector<reco::SoftLeptonTagInfo>>(fConfig.getParameter<edm::InputTag>("srcBTagInfoElectron")))
  , muonToken_(fCollector.consumes<std::vector<reco::SoftLeptonTagInfo>>(fConfig.getParameter<edm::InputTag>("srcBTagInfoMuon")))
  , addMuonToJet_(fConfig.getParameter<bool>("addMuonToJet"))
{
}

std::unique_ptr<reco::JetCorrectorImpl> 
L6SLBCorrectorImplMaker::make(edm::Event const&fEvent, edm::EventSetup const& fSetup)
{
  edm::Handle< std::vector<reco::SoftLeptonTagInfo> > muoninfos;
  fEvent.getByToken(muonToken_,muoninfos);
  edm::RefProd<std::vector<reco::SoftLeptonTagInfo>> muonProd(muoninfos);

  edm::Handle< std::vector<reco::SoftLeptonTagInfo> > elecinfos;
  fEvent.getByToken(elecToken_,elecinfos);
  edm::RefProd<std::vector<reco::SoftLeptonTagInfo>> elecProd(elecinfos);

  auto calculator = getCalculator(fSetup,
				  [](std::string const& ) {});
  return std::unique_ptr<reco::JetCorrectorImpl>(new L6SLBCorrectorImpl(calculator,
									muonProd,
									elecProd,
									addMuonToJet_));
}

void 
L6SLBCorrectorImplMaker::fillDescriptions(edm::ConfigurationDescriptions& iDescriptions)
{
  edm::ParameterSetDescription desc;
  addToDescription(desc);
  desc.add<edm::InputTag>("srcBTagInfoElectron");
  desc.add<edm::InputTag>("srcBTagInfoMuon");
  desc.add<bool>("addMuonToJet");
  iDescriptions.addDefault(desc);
}

////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
L6SLBCorrectorImpl::L6SLBCorrectorImpl(std::shared_ptr<FactorizedJetCorrectorCalculator const> corrector,
				       edm::RefProd<std::vector<reco::SoftLeptonTagInfo>> const& bTagInfoMuon,
				       edm::RefProd<std::vector<reco::SoftLeptonTagInfo>> const& bTagInfoElec,
				       bool addMuonToJet):
	       corrector_(corrector),
	       bTagInfoMuon_(bTagInfoMuon),
	       bTagInfoElec_(bTagInfoElec),
	       addMuonToJet_(addMuonToJet)
{
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
double L6SLBCorrectorImpl::correction(const LorentzVector& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(LorentzVector), event required!";
  return 1.0;
}


//______________________________________________________________________________
double L6SLBCorrectorImpl::correction(const reco::Jet& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(reco::Jet), event required!";
  return 1.0;
}


//______________________________________________________________________________
double L6SLBCorrectorImpl::correction(const reco::Jet& fJet,
				      const edm::RefToBase<reco::Jet>& refToRawJet) const
{
  FactorizedJetCorrectorCalculator::VariableValues values;
  values.setJetPt(fJet.pt());
  values.setJetEta(fJet.eta());
  values.setJetPhi(fJet.phi());
  values.setJetE(fJet.energy());
  
  const reco::SoftLeptonTagInfo& sltMuon =
    (*bTagInfoMuon_)[getBTagInfoIndex(refToRawJet,*bTagInfoMuon_)];
  if (sltMuon.leptons()>0) {
    edm::RefToBase<reco::Track> trackRef = sltMuon.lepton(0);
    values.setLepPx(trackRef->px());
    values.setLepPy(trackRef->py());
    values.setLepPz(trackRef->pz());
    values.setAddLepToJet(addMuonToJet_);
    return corrector_->getCorrection(values);
  }
  else {
    const reco::SoftLeptonTagInfo& sltElec =
      (*bTagInfoElec_)[getBTagInfoIndex(refToRawJet,*bTagInfoElec_)];
    if (sltElec.leptons()>0) {
      edm::RefToBase<reco::Track> trackRef = sltElec.lepton(0);
      values.setLepPx(trackRef->px());
      values.setLepPy(trackRef->py());
      values.setLepPz(trackRef->pz());
      values.setAddLepToJet(false);
      return corrector_->getCorrection(values);
    }
  }
  return 1.0;
}


////////////////////////////////////////////////////////////////////////////////
// implementation of private member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
int L6SLBCorrectorImpl::getBTagInfoIndex(const edm::RefToBase<reco::Jet>& refToRawJet,
				     const vector<reco::SoftLeptonTagInfo>& tags)
  const
{
  for (unsigned int i=0;i<tags.size();i++)
    if (tags[i].jet().get()==refToRawJet.get()) return i;
  return -1;
}
