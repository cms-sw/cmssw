// -*- C++ -*-
//
// Package:    HeavyIonsAnalysis/HiMixValidation
// Class:      HiMixValidation
// 
/**\class HiMixValidation HiMixValidation.cc HeavyIonsAnalysis/HiMixValidation/plugins/HiMixValidation.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Wed, 15 Oct 2014 15:12:55 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "TH1D.h"
#include "TH2D.h"


//
// class declaration
//

class HiMixValidation : public edm::EDAnalyzer {
   public:
      explicit HiMixValidation(const edm::ParameterSet&);
      ~HiMixValidation();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------

   edm::InputTag genParticleSrc_;
   edm::InputTag genHIsrc_;

   TH1D *hGenParticleEtaSignal, 
      *hGenParticleEtaBkg, 
      *hGenParticleEtaBkgNpart2, 
      *hGenParticleEtaMixed,
      *hGenJetEtaSignal,
      *hGenJetEtaBkg,
      *hGenJetEtaBkgNpart2,
      *hGenJetEtaMixed,
      *hGenPhotonEtaSignal,
      *hGenPhotonEtaBkg,
      *hGenPhotonEtaBkgNpart2,
      *hGenPhotonEtaMixed,
      *hGenMuonEtaSignal,
      *hGenMuonEtaBkg,
      *hGenMuonEtaBkgNpart2,
      *hGenMuonEtaMixed,
      *hGenElectronEtaSignal,
      *hGenElectronEtaBkg,
      *hGenElectronEtaBkgNpart2,
      *hGenElectronEtaMixed,
      *hCentralityBin,
      *hNrecoVertex,
      *hZrecoVertex0,
      *hZrecoVertex1;

   TH2D *hZrecoVertices,
      *hNtrackHF;

   edm::Service<TFileService> f;


};

HiMixValidation::HiMixValidation(const edm::ParameterSet& iConfig)
{
   genParticleSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("genpSrc",edm::InputTag("hiGenParticles"));
   genHIsrc_ = iConfig.getUntrackedParameter<edm::InputTag>("genHiSrc",edm::InputTag("heavyIon"));
}


HiMixValidation::~HiMixValidation()
{
 

}

void
HiMixValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   edm::Handle<GenHIEvent> higen;
   iEvent.getByLabel(genHIsrc_,higen);
   double npart = higen->Npart();

   edm::Handle<reco::GenParticleCollection> parts;
   iEvent.getByLabel(genParticleSrc_,parts);
   for(UInt_t i = 0; i < parts->size(); ++i){
      const reco::GenParticle& p = (*parts)[i];
      if (p.status()!=1) continue;
      int sube = p.collisionId();
      int pdg = abs(p.pdgId());
      double eta = p.eta();
      if(sube == 0){
	 if(p.charge() != 0) hGenParticleEtaSignal->Fill(eta);
	 if(pdg == 22) hGenPhotonEtaSignal->Fill(eta);	 
	 if(pdg == 11) hGenElectronEtaSignal->Fill(eta);	 
	 if(pdg == 13) hGenMuonEtaSignal->Fill(eta);

      }else{
	 if(p.charge() != 0) hGenParticleEtaBkg->Fill(eta);
         if(pdg == 22) hGenPhotonEtaBkg->Fill(eta);
         if(pdg == 11) hGenElectronEtaBkg->Fill(eta);
         if(pdg == 13) hGenMuonEtaBkg->Fill(eta);
	 if(npart == 2){
	    if(p.charge() != 0) hGenParticleEtaBkgNpart2->Fill(eta);
	    if(pdg == 22) hGenPhotonEtaBkgNpart2->Fill(eta);
	    if(pdg == 11) hGenElectronEtaBkgNpart2->Fill(eta);
	    if(pdg == 13) hGenMuonEtaBkgNpart2->Fill(eta);
	 }
      }
      
      if(p.charge() != 0) hGenParticleEtaMixed->Fill(eta);
      if(pdg == 22) hGenPhotonEtaMixed->Fill(eta);
      if(pdg == 11) hGenElectronEtaMixed->Fill(eta);
      if(pdg == 13) hGenMuonEtaMixed->Fill(eta);

   }

}


void 
HiMixValidation::beginJob()
{

   double histEtaMax = 10;
   int histEtaBins = 200;
   int NcentralityBins = 100;
   int NvtxMax = 20;

   hGenParticleEtaSignal = f->make<TH1D>("hGenParticleEtaSignal",";#eta;Particles",histEtaBins,-histEtaMax,histEtaMax);
   hGenParticleEtaBkg = f->make<TH1D>("hGenParticleEtaBkg",";#eta;Particles",histEtaBins,-histEtaMax,histEtaMax);
   hGenParticleEtaBkgNpart2 = f->make<TH1D>("hGenParticleEtaBkgNpart2",";#eta;Particles",histEtaBins,-histEtaMax,histEtaMax);
   hGenParticleEtaMixed = f->make<TH1D>("hGenParticleEtaMixed",";#eta;Particles",histEtaBins,-histEtaMax,histEtaMax);
   hGenJetEtaSignal = f->make<TH1D>("hGenJetEtaSignal",";#eta;GenJets",histEtaBins,-histEtaMax,histEtaMax);
   hGenJetEtaBkg = f->make<TH1D>("hGenJetEtaBkg",";#eta;GenJets",histEtaBins,-histEtaMax,histEtaMax);
   hGenJetEtaBkgNpart2 = f->make<TH1D>("hGenJetEtaBkgNpart2",";#eta;GenJets",histEtaBins,-histEtaMax,histEtaMax);
   hGenJetEtaMixed = f->make<TH1D>("hGenJetEtaMixed",";#eta;GenJets",histEtaBins,-histEtaMax,histEtaMax);
   hGenPhotonEtaSignal = f->make<TH1D>("hGenPhotonEtaSignal",";#eta;Photons",histEtaBins,-histEtaMax,histEtaMax);
   hGenPhotonEtaBkg = f->make<TH1D>("hGenPhotonEtaBkg",";#eta;Photons",histEtaBins,-histEtaMax,histEtaMax);
   hGenPhotonEtaBkgNpart2 = f->make<TH1D>("hGenPhotonEtaBkgNpart2",";#eta;Photons",histEtaBins,-histEtaMax,histEtaMax);
   hGenPhotonEtaMixed = f->make<TH1D>("hGenPhotonEtaMixed",";#eta;Photons",histEtaBins,-histEtaMax,histEtaMax);
   hGenMuonEtaSignal = f->make<TH1D>("hGenMuonEtaSignal",";#eta;Muons",histEtaBins,-histEtaMax,histEtaMax);
   hGenMuonEtaBkg = f->make<TH1D>("hGenMuonEtaBkg",";#eta;Muons",histEtaBins,-histEtaMax,histEtaMax);
   hGenMuonEtaBkgNpart2 = f->make<TH1D>("hGenMuonEtaBkgNpart2",";#eta;Muons",histEtaBins,-histEtaMax,histEtaMax);
   hGenMuonEtaMixed = f->make<TH1D>("hGenMuonEtaMixed",";#eta;Muons",histEtaBins,-histEtaMax,histEtaMax);
   hGenElectronEtaSignal = f->make<TH1D>("hGenElectronEtaSignal",";#eta;Electrons",histEtaBins,-histEtaMax,histEtaMax);
   hGenElectronEtaBkg = f->make<TH1D>("hGenElectronEtaBkg",";#eta;Electrons",histEtaBins,-histEtaMax,histEtaMax);
   hGenElectronEtaBkgNpart2 = f->make<TH1D>("hGenElectronEtaBkgNpart2",";#eta;Electrons",histEtaBins,-histEtaMax,histEtaMax);
   hGenElectronEtaMixed = f->make<TH1D>("hGenElectronEtaMixed",";#eta;Electrons",histEtaBins,-histEtaMax,histEtaMax);

   hCentralityBin = f->make<TH1D>("hCentralityBin",";Bin;Events",NcentralityBins,0,NcentralityBins);
   hNrecoVertex = f->make<TH1D>("hNrecoVertex",";Number of reconstructed vertices;Events",NvtxMax,0,NvtxMax);
   hZrecoVertex0 = f->make<TH1D>("hZrecoVertex0",";z-position of first vertex [cm];Events",60,-30,30);
   hZrecoVertex1 = f->make<TH1D>("hZrecoVertex1",";z-position of second vertex [cm];Events",60,-30,30);

   hZrecoVertices = f->make<TH2D>("hZrecoVertices",";z-position of first vertex;z-position of second vertex;Events",60,-30,30,60,-30,30);
   hNtrackHF = f->make<TH2D>("hNtrackHF",";E_{T}^{HF};N_{tracks};Events",50,0,5000,50,0,5000);
}

void 
HiMixValidation::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
HiMixValidation::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
HiMixValidation::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
HiMixValidation::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
HiMixValidation::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HiMixValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiMixValidation);
