// -*- C++ -*-
//
// Package:    DQMOffline/ECALMultifitAnalyzer_HI
// Class:      ECALMultifitAnalyzer_HI
//
/**\class ECALMultifitAnalyzer_HI ECALMultifitAnalyzer_HI.cc DQMOffline/ECALMultifitAnalyzer_HI/plugins/ECALMultifitAnalyzer_HI.cc
   Description: [one line class summary]
   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  R. Alex Barbieri
//         Created:  Tue, 17 Nov 2015 17:18:50 GMT
// Modified for DQM: Raghav Kunnawalkam Elayavalli
//                   Wednesday 18 Nov 2015 9:45:AM GVA time
//
//


// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "TH2F.h"

//
// class declaration
//

class ECALMultifitAnalyzer_HI : public DQMEDAnalyzer  {
public:
  explicit ECALMultifitAnalyzer_HI(const edm::ParameterSet&);
  ~ECALMultifitAnalyzer_HI() {}

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<std::vector<reco::Photon> > recoPhotonsCollection_;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetToken_;
  edm::EDGetTokenT<EcalRecHitCollection> RecHitCollection_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> RecHitCollection_EE_;

  double mRechitEnergyThreshold;
  double mRecoPhotonPtThreshold;
  double mRecoJetPtThreshold;
  double mDeltaRPhotonThreshold;
  double mDeltaRJetThreshold;
  
  edm::ESHandle<CaloGeometry> geomH;

  MonitorElement * eb_chi2;
  MonitorElement * eb_chi2_eta;
  MonitorElement * eb_chi2_e5;
  MonitorElement * eb_chi2_e5_eta;
  MonitorElement * eb_errors;
  MonitorElement * eb_errors_eta;
  MonitorElement * eb_errors_e5;
  MonitorElement * eb_errors_e5_eta;
  MonitorElement * eb_chi2_photon15;
  MonitorElement * eb_errors_photon15;
  MonitorElement * eb_chi2_jet30;
  MonitorElement * eb_errors_jet30;

  MonitorElement * ee_chi2;
  MonitorElement * ee_chi2_eta;
  MonitorElement * ee_chi2_e5;
  MonitorElement * ee_chi2_e5_eta;
  MonitorElement * ee_errors;
  MonitorElement * ee_errors_eta;
  MonitorElement * ee_errors_e5;
  MonitorElement * ee_errors_e5_eta;
  MonitorElement * ee_chi2_photon15;
  MonitorElement * ee_errors_photon15;
  MonitorElement * ee_chi2_jet30;
  MonitorElement * ee_errors_jet30;
};

//
// constructors and destructor
//
ECALMultifitAnalyzer_HI::ECALMultifitAnalyzer_HI(const edm::ParameterSet& iConfig):

  recoPhotonsCollection_(consumes<std::vector<reco::Photon> > (iConfig.getParameter<edm::InputTag>("recoPhotonSrc"))),
  caloJetToken_(consumes<reco::CaloJetCollection>             (iConfig.getParameter<edm::InputTag> ("recoJetSrc"))),
  RecHitCollection_EB_(consumes<EcalRecHitCollection>         (iConfig.getParameter<edm::InputTag>("RecHitCollection_EB"))),
  RecHitCollection_EE_(consumes<EcalRecHitCollection>         (iConfig.getParameter<edm::InputTag>("RecHitCollection_EE"))),
  mRechitEnergyThreshold                                      (iConfig.getParameter<double> ("rechitEnergyThreshold")),
  mRecoPhotonPtThreshold                                      (iConfig.getParameter<double> ("recoPhotonPtThreshold")),
  mRecoJetPtThreshold                                         (iConfig.getParameter<double> ("recoJetPtThreshold")),
  mDeltaRPhotonThreshold                                      (iConfig.getParameter<double> ("deltaRPhotonThreshold")),
  mDeltaRJetThreshold                                         (iConfig.getParameter<double> ("deltaRJetThreshold"))
{
  
}

//
// member functions
//

// ------------ method called for each event  ------------
void ECALMultifitAnalyzer_HI::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  
  iSetup.get<CaloGeometryRecord>().get(geomH);

  Handle<std::vector<reco::Photon> > recoPhotonsHandle;
  iEvent.getByToken(recoPhotonsCollection_, recoPhotonsHandle);

  Handle<reco::CaloJetCollection> recoJetHandle;
  iEvent.getByToken(caloJetToken_, recoJetHandle);

  Handle<EcalRecHitCollection> ebHandle;
  iEvent.getByToken(RecHitCollection_EB_, ebHandle);

  Handle<EcalRecHitCollection> eeHandle;
  iEvent.getByToken(RecHitCollection_EE_, eeHandle);

  for(EcalRecHitCollection::const_iterator hit = ebHandle->begin(); hit != ebHandle->end(); ++hit) {
    eb_chi2->Fill(hit->chi2() );
    eb_errors->Fill(hit->energyError() );
    double eta = geomH->getGeometry(hit->detid())->getPosition().eta();
    double phi = geomH->getGeometry(hit->detid())->getPosition().phi();
    eb_chi2_eta->Fill(eta, hit->chi2() );
    eb_errors_eta->Fill(eta, hit->energyError() );
    if(hit->energy() > mRechitEnergyThreshold)
    {
      eb_chi2_e5->Fill(hit->chi2() );
      eb_errors_e5->Fill(hit->energyError() );
      eb_chi2_e5_eta->Fill(eta, hit->chi2() );
      eb_errors_e5_eta->Fill(eta, hit->energyError() );
    }

    for (std::vector<reco::Photon>::const_iterator pho = recoPhotonsHandle->begin(); pho != recoPhotonsHandle->end(); ++pho) {
      if(pho->et() < mRecoPhotonPtThreshold ) continue;
      double dr = reco::deltaR(eta, phi, pho->eta(), pho->phi());
      if(dr < mDeltaRPhotonThreshold)
      {
	eb_chi2_photon15->Fill(hit->chi2() );
	eb_errors_photon15->Fill(hit->energyError() );
      }
    }
    for (std::vector<reco::CaloJet>::const_iterator jet = recoJetHandle->begin(); jet != recoJetHandle->end(); ++jet) {
      if(jet->pt() < mRecoJetPtThreshold ) continue;
      double dr = reco::deltaR(eta, phi, jet->eta(), jet->phi());
      if(dr < mDeltaRJetThreshold)
      {
	eb_chi2_jet30->Fill(hit->chi2() );
	eb_errors_jet30->Fill(hit->energyError() );
      }
    }
  }

  for(EcalRecHitCollection::const_iterator hit = eeHandle->begin(); hit != eeHandle->end(); ++hit) {
    ee_chi2->Fill(hit->chi2() );
    ee_errors->Fill(hit->energyError() );
    double eta = geomH->getGeometry(hit->detid())->getPosition().eta();
    double phi = geomH->getGeometry(hit->detid())->getPosition().phi();
    ee_chi2_eta->Fill(eta, hit->chi2() );
    ee_errors_eta->Fill(eta, hit->energyError() );
    if(hit->energy() > mRechitEnergyThreshold)
    {
      ee_chi2_e5->Fill(hit->chi2() );
      ee_errors_e5->Fill(hit->energyError() );
      ee_chi2_e5_eta->Fill(eta, hit->chi2() );
      ee_errors_e5_eta->Fill(eta, hit->energyError() );
    }

    for (std::vector<reco::Photon>::const_iterator pho = recoPhotonsHandle->begin(); pho != recoPhotonsHandle->end(); ++pho) {
      if(pho->et() < mRecoPhotonPtThreshold ) continue;
      double dr = reco::deltaR(eta, phi, pho->eta(), pho->phi());
      if(dr < mDeltaRPhotonThreshold)
      {
	ee_chi2_photon15->Fill(hit->chi2() );
	ee_errors_photon15->Fill(hit->energyError() );
      }
    }
    for (std::vector<reco::CaloJet>::const_iterator jet = recoJetHandle->begin(); jet != recoJetHandle->end(); ++jet) {
      if(jet->pt() < mRecoJetPtThreshold ) continue;
      double dr = reco::deltaR(eta, phi, jet->eta(), jet->phi());
      if(dr < mDeltaRJetThreshold)
      {
	ee_chi2_jet30->Fill(hit->chi2() );
	ee_errors_jet30->Fill(hit->energyError() );
      }
    }
  }
}


// ------------ method called once each job just before starting event loop  ------------
void
ECALMultifitAnalyzer_HI::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&)
{

  iBooker.setCurrentFolder ("EcalCalibration/HIN_MultiFitAnalyzer");

  eb_chi2 = 0;
  eb_chi2_eta = 0;
  eb_chi2_e5 = 0;
  eb_chi2_e5_eta = 0;
  eb_errors = 0;
  eb_errors_eta = 0;
  eb_errors_e5 = 0;
  eb_errors_e5_eta = 0;
  eb_chi2_photon15 = 0;
  eb_errors_photon15 = 0;
  eb_chi2_jet30 = 0;
  eb_errors_jet30 = 0;

  ee_chi2 = 0;
  ee_chi2_eta = 0;
  ee_chi2_e5 = 0;
  ee_chi2_e5_eta = 0;
  ee_errors = 0;
  ee_errors_eta = 0;
  ee_errors_e5 = 0;
  ee_errors_e5_eta = 0;
  ee_chi2_photon15 = 0;
  ee_errors_photon15 = 0;
  ee_chi2_jet30 = 0;
  ee_errors_jet30 = 0;

  const int nBins = 500;
  const float maxChi2 = 70;
  const float maxError = 0.5;

  TH2F * hProfile_Chi2 = new TH2F("hProfile_Chi2","",nBins, -5, 5, nBins, 0, maxChi2);
  TH2F * hProfile_Err = new TH2F("hProfile_Err","",nBins, -5, 5, nBins, 0, maxError);

  eb_chi2 = iBooker.book1D("rechit_eb_chi2","Rechit eb_chi2;chi2 fit value;",nBins,0,maxChi2);
  eb_chi2_eta = iBooker.book2D("rechit_eb_eta_Vs_mean_Chi2", hProfile_Chi2);
  eb_chi2_e5 = iBooker.book1D(Form("rechit_eb_chi2_e%2.0f", mRechitEnergyThreshold),Form("Rechit eb_chi2, e>%2.0fGeV;chi2 fit value;", mRechitEnergyThreshold),nBins,0,maxChi2);
  eb_chi2_e5_eta = iBooker.book2D(Form("rechit_eb_chi2_e%2.0f_eta", mRechitEnergyThreshold),hProfile_Chi2);
  eb_errors = iBooker.book1D("rechit_eb_errors","Rechit eb_errors;error on the energy;",nBins,0,maxError);
  eb_errors_eta = iBooker.book2D("rechit_eb_errors_eta",hProfile_Err);
  eb_errors_e5 = iBooker.book1D(Form("rechit_eb_errors_e%2.0f", mRechitEnergyThreshold),"Rechit eb_errors, e>5GeV;error on the energy;",nBins,0,maxError);
  eb_errors_e5_eta = iBooker.book2D(Form("rechit_eb_errors_e%2.0f_eta", mRechitEnergyThreshold),hProfile_Err);
  eb_chi2_photon15 = iBooker.book1D(Form("rechit_eb_chi2_photon%2.0f", mRecoPhotonPtThreshold),"Rechit eb_chi2 near photons;chi2 fit value;",nBins,0,maxChi2);
  eb_errors_photon15 = iBooker.book1D(Form("rechit_eb_errors_photon%2.0f", mRecoPhotonPtThreshold),"Rechit eb_errors near photons;error on the energy;",nBins,0,maxError);
  eb_chi2_jet30 = iBooker.book1D(Form("rechit_eb_chi2_jet%2.0f", mRecoJetPtThreshold),"Rechit eb_chi2 near jets;chi2 fit value;",nBins,0,maxChi2);
  eb_errors_jet30 = iBooker.book1D(Form("rechit_eb_errors_jet%2.0f", mRecoJetPtThreshold),"Rechit eb_errors near jets;error on the energy;",nBins,0,maxError);

  ee_chi2 = iBooker.book1D("rechit_ee_chi2","Rechit ee_chi2;chi2 fit value;",nBins,0,maxChi2);
  ee_chi2_eta = iBooker.book2D("rechit_ee_chi2_eta",hProfile_Chi2);
  ee_chi2_e5 = iBooker.book1D(Form("rechit_ee_chi2_e%2.0f", mRechitEnergyThreshold),"Rechit ee_chi2, e>5GeV;chi2 fit value;",nBins,0,maxChi2);
  ee_chi2_e5_eta = iBooker.book2D(Form("rechit_ee_chi2_e%2.0f_eta", mRechitEnergyThreshold),hProfile_Chi2);
  ee_errors = iBooker.book1D("rechit_ee_errors","Rechit ee_errors;error on the energy;",nBins,0,maxError);
  ee_errors_eta = iBooker.book2D("rechit_ee_errors_eta",hProfile_Err);
  ee_errors_e5 = iBooker.book1D(Form("rechit_ee_errors_e%2.0f", mRechitEnergyThreshold),"Rechit ee_errors, e>5GeV;error on the energy;",nBins,0,maxError);
  ee_errors_e5_eta = iBooker.book2D(Form("rechit_ee_errors_e%2.0f_eta", mRechitEnergyThreshold),hProfile_Err);
  ee_chi2_photon15 = iBooker.book1D(Form("rechit_ee_chi2_photon%2.0f", mRecoPhotonPtThreshold),"Rechit ee_chi2 near photons;chi2 fit value;",nBins,0,maxChi2);
  ee_errors_photon15 = iBooker.book1D(Form("rechit_ee_errors_photon%2.0f", mRecoPhotonPtThreshold),"Rechit ee_errors near photons;error on the energy;",nBins,0,maxError);
  ee_chi2_jet30 = iBooker.book1D(Form("rechit_ee_chi2_jet%2.0f", mRecoJetPtThreshold),"Rechit ee_chi2 near jets;chi2 fit value;",nBins,0,maxChi2);
  ee_errors_jet30 = iBooker.book1D(Form("rechit_ee_errors_jet%2.0f", mRecoJetPtThreshold),"Rechit ee_errors near jets;error on the energy;",nBins,0,maxError);

  delete hProfile_Chi2;
  delete hProfile_Err;

}


//define this as a plug-in
DEFINE_FWK_MODULE(ECALMultifitAnalyzer_HI);
