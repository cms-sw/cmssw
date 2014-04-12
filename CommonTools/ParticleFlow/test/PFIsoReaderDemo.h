#ifndef PFISOREADERDEMO_H
#define PFISOREADERDEMO_H
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <iostream>
#include <string>
#include <map>

#include "TH2F.h"


class PFIsoReaderDemo : public edm::EDAnalyzer
{
 public:
  explicit PFIsoReaderDemo(const edm::ParameterSet&);
  ~PFIsoReaderDemo();
  virtual void beginRun(edm::Run const&, edm::EventSetup const& );
  virtual void analyze(const edm::Event & iEvent,const edm::EventSetup & c);

  typedef std::vector< edm::Handle< edm::ValueMap<reco::IsoDeposit> > > IsoDepositMaps;
  typedef std::vector< edm::Handle< edm::ValueMap<double> > > IsoDepositVals;

 private:

  bool printPhotons_;
  bool printElectrons_;

  void plotIsoDeposits(const IsoDepositMaps & depmap, const reco::GsfElectronRef & ref,
		       double&, double&, double&) ;

 //This analyzer produces a file with a tree so we need,
  edm::Service<TFileService> fileservice_;

  edm::InputTag inputTagGsfElectrons_;
  edm::EDGetTokenT<reco::GsfElectronCollection> tokenGsfElectrons_;
  edm::InputTag inputTagPhotons_;
  edm::EDGetTokenT<reco::PhotonCollection> tokenPhotons_;
//   edm::InputTag inputTagPFCandidateMap_;
//   std::vector<edm::InputTag> inputTagIsoDepElectrons_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<reco::IsoDeposit> > > tokensIsoDepElectrons_;
//   std::vector<edm::InputTag> inputTagIsoDepPhotons_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<reco::IsoDeposit> > > tokensIsoDepPhotons_;
  //  std::vector<edm::InputTag> inputTagIsoValElectronsNoPFId_;
//   std::vector<edm::InputTag> inputTagIsoValElectronsPFId_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<double> > > tokensIsoValElectronsPFId_;
//   std::vector<edm::InputTag> inputTagIsoValPhotonsPFId_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<double> > > tokensIsoValPhotonsPFId_;

  // Control histos
  TH1F* chargedBarrelElectrons_   ;
  TH1F* photonBarrelElectrons_    ;
  TH1F* neutralBarrelElectrons_   ;

  TH1F* chargedEndcapsElectrons_  ;
  TH1F* photonEndcapsElectrons_   ;
  TH1F* neutralEndcapsElectrons_  ;

  TH1F* sumBarrelElectrons_       ;
  TH1F* sumEndcapsElectrons_      ;

  TH1F* chargedBarrelPhotons_   ;
  TH1F* photonBarrelPhotons_    ;
  TH1F* neutralBarrelPhotons_   ;

  TH1F* chargedEndcapsPhotons_  ;
  TH1F* photonEndcapsPhotons_   ;
  TH1F* neutralEndcapsPhotons_  ;

  TH1F* sumBarrelPhotons_       ;
  TH1F* sumEndcapsPhotons_      ;
};
#endif
