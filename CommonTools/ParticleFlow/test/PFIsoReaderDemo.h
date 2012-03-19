#ifndef PFISOREADERDEMO_H
#define PFISOREADERDEMO_H
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
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

  void plotIsoDeposits(const IsoDepositMaps & depmap, const reco::GsfElectronRef & ref,
		       double&, double&, double&) ;
  
 //This analyzer produces a file with a tree so we need,
  edm::Service<TFileService> fileservice_;

  edm::InputTag inputTagGsfElectrons_;
  std::vector<edm::InputTag> inputTagIsoDepElectrons_;
  std::vector<edm::InputTag> inputTagIsoValElectronsNoPFId_;
  std::vector<edm::InputTag> inputTagIsoValElectronsPFId_;   

  // Control histos
  TH1F* chargedBarrel_   ; 
  TH1F* photonBarrel_    ; 
  TH1F* neutralBarrel_   ; 
    
  TH1F* chargedEndcaps_  ; 
  TH1F* photonEndcaps_   ; 
  TH1F* neutralEndcaps_  ; 

  TH1F* sumBarrel_       ;
  TH1F* sumEndcaps_      ;
};
#endif
