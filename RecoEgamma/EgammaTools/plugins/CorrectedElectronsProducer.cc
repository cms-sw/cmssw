/**\class CorrectedElectrons CorrectedElectrons.cc RecoEgamma/CorrectedElectrons/src/CorrectedElectrons.cc

// Original Author:  Matteo Sani,40 3-A02,+41227671577,
//         Created:  Mon Jun  7 09:34:28 CEST 2010
*/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"

class CorrectedElectronsProducer : public edm::EDProducer {
public:
  explicit CorrectedElectronsProducer(const edm::ParameterSet&);
  ~CorrectedElectronsProducer();

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ; 
  float correct_phi(float dphi);
  
  edm::InputTag electronCollection_;
  double energyScale_;
  std::vector<double> scPositionCorrectionEBP_;
  std::vector<double> scPositionCorrectionEBM_;
  std::vector<double> scPositionCorrectionEEP_;
  std::vector<double> scPositionCorrectionEEM_;
};


CorrectedElectronsProducer::CorrectedElectronsProducer(const edm::ParameterSet& iConfig) {
  
  electronCollection_ = iConfig.getParameter<edm::InputTag>("electronCollection");
  scPositionCorrectionEBP_ = iConfig.getParameter<std::vector<double> >("scPositionCorrectionEBPlus");
  scPositionCorrectionEBM_ = iConfig.getParameter<std::vector<double> >("scPositionCorrectionEBMinus");
  scPositionCorrectionEEP_ = iConfig.getParameter<std::vector<double> >("scPositionCorrectionEEPlus");
  scPositionCorrectionEEM_ = iConfig.getParameter<std::vector<double> >("scPositionCorrectionEEMinus");

  produces<reco::GsfElectronCollection>();
}


CorrectedElectronsProducer::~CorrectedElectronsProducer()
{}

float CorrectedElectronsProducer::correct_phi(float dphi) {
  if (fabs(dphi) > CLHEP::pi) {
    if (dphi < 0)
      dphi = CLHEP::twopi+dphi;
    else
      dphi = dphi-CLHEP::twopi; 
  }

  return dphi;
}

void CorrectedElectronsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  using namespace edm;
  using namespace reco;

  std::auto_ptr<GsfElectronCollection> pOutEle(new GsfElectronCollection);

  edm::Handle<reco::GsfElectronCollection> elH;
  iEvent.getByLabel(electronCollection_, elH);

  for(reco::GsfElectronCollection::const_iterator igsf = elH->begin(); igsf != elH->end(); igsf++) {
    
    reco::GsfElectron* uncorEl = new reco::GsfElectron(*igsf);

    Candidate::LorentzVector momentum = Candidate::LorentzVector(uncorEl->px(),
                                                                 uncorEl->py(),
                                                                 uncorEl->pz(),
                                                                 uncorEl->energy());
    int charge = uncorEl->charge();
    GsfElectron::ChargeInfo chargeInfo = uncorEl->chargeInfo();
    GsfElectronCoreRef coreRef = uncorEl->core();

    GsfElectron::TrackExtrapolations tkExtra =  uncorEl->trackExtrapolations();    

    GsfElectron::TrackClusterMatching tcMatching = uncorEl->trackClusterMatching();
    
    math::XYZVector scNewPos(0,0,0);
    if(uncorEl->isEB() and uncorEl->eta() > 0) {
      scNewPos = math::XYZVector(uncorEl->superCluster()->x()+scPositionCorrectionEBP_[0],
                                 uncorEl->superCluster()->y()+scPositionCorrectionEBP_[1],
                                 uncorEl->superCluster()->z()+scPositionCorrectionEBP_[2]);
    } else if(uncorEl->isEB() and uncorEl->eta() < 0) {
      scNewPos = math::XYZVector(uncorEl->superCluster()->x()+scPositionCorrectionEBM_[0],
                                 uncorEl->superCluster()->y()+scPositionCorrectionEBM_[1],
                                 uncorEl->superCluster()->z()+scPositionCorrectionEBM_[2]);
    } else if(uncorEl->isEE() and uncorEl->eta() > 0) { 
      scNewPos = math::XYZVector(uncorEl->superCluster()->x()+scPositionCorrectionEEP_[0],
                                 uncorEl->superCluster()->y()+scPositionCorrectionEEP_[1],
                                 uncorEl->superCluster()->z()+scPositionCorrectionEEP_[2]);
    } else if(uncorEl->isEE() and uncorEl->eta() < 0) {
      scNewPos = math::XYZVector(uncorEl->superCluster()->x()+scPositionCorrectionEEM_[0],
                                 uncorEl->superCluster()->y()+scPositionCorrectionEEM_[1],
                                 uncorEl->superCluster()->z()+scPositionCorrectionEEM_[2]);
    }

    float deta_sc = scNewPos.eta() - uncorEl->superCluster()->eta();
    float dphi_sc = correct_phi(scNewPos.phi() - uncorEl->superCluster()->phi());
    
    tcMatching.deltaEtaSuperClusterAtVtx = deta_sc + uncorEl->deltaEtaSuperClusterTrackAtVtx();
    tcMatching.deltaEtaEleClusterAtCalo = deta_sc + uncorEl->deltaEtaEleClusterTrackAtCalo();
    tcMatching.deltaEtaSeedClusterAtCalo =  deta_sc + uncorEl->deltaEtaSeedClusterTrackAtCalo();

    tcMatching.deltaPhiEleClusterAtCalo = correct_phi(dphi_sc + uncorEl->deltaPhiEleClusterTrackAtCalo());
    tcMatching.deltaPhiSuperClusterAtVtx = correct_phi(dphi_sc + uncorEl->deltaPhiSuperClusterTrackAtVtx());
    tcMatching.deltaPhiSeedClusterAtCalo = correct_phi(dphi_sc + uncorEl->deltaPhiSeedClusterTrackAtCalo());


    GsfElectron::ClosestCtfTrack ctfInfo = uncorEl->closestCtfTrack();
    
    GsfElectron::FiducialFlags fiducialFlags = uncorEl-> fiducialFlags();
    GsfElectron::ShowerShape showerShape = uncorEl->showerShape();
    float fbrem = uncorEl->fbrem();
    float mva = uncorEl->mva();
    
    GsfElectron * corEl = new GsfElectron(momentum,charge,
                                          chargeInfo,coreRef,
                                          tcMatching, tkExtra, ctfInfo,
                                          fiducialFlags,showerShape, fbrem, mva);

    reco::GsfElectron::IsolationVariables dr03, dr04;
    corEl->setIsolation03(uncorEl->isolationVariables03());
    corEl->setIsolation04(uncorEl->isolationVariables04());
    
    pOutEle->push_back(*corEl) ;
  }
  
  // put result into the Event
  iEvent.put(pOutEle);
}


void CorrectedElectronsProducer::beginJob()
{}

void CorrectedElectronsProducer::endJob() {}

DEFINE_FWK_MODULE(CorrectedElectronsProducer);
