// -*- C++ -*-
//
// Package:    PPSValidation/Validation
// Class:      Validation
// 
/**\class Validation Validation.cc PPSValidation/Validation/plugins/Validation.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Sandro Fonseca De Souza
//         Created:  Tue, 01 Dec 2015 12:10:07 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//Central Detector (CMS)
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/PFJet.h"
//PPS
#include "FastSimulation/PPSFastObjects/interface/PPSSpectrometer.h"
#include "FastSimulation/PPSFastObjects/interface/PPSGenData.h"
#include "FastSimulation/PPSFastObjects/interface/PPSSimData.h"
#include "FastSimulation/PPSFastObjects/interface/PPSRecoData.h"
#include "FastSimulation/PPSFastObjects/interface/PPSGenVertex.h"
#include "FastSimulation/PPSFastObjects/interface/PPSRecoVertex.h"
//
//
//
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TTree.h"
#include "TCanvas.h"
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <map>
//

using namespace edm;
using namespace std;

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

class Validation : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
	public:
		explicit Validation(const edm::ParameterSet&);
		~Validation();

		static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


	private:
		virtual void beginJob() override;
		virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
		virtual void endJob() override;
		
		//
		void GenPPSInfo(const edm::Event&, const edm::EventSetup&);
                void GenInfo(const edm::Event&, const edm::EventSetup&);
		void PPSResolutionInfo(const edm::Event&, const edm::EventSetup&);
		// ----------member data ---------------------------
		typedef edm::View<reco::Vertex> VertexCollection;
		typedef edm::View<reco::GsfElectron> ElectronCollection;
		typedef edm::View<reco::Muon> MuonCollection;
		typedef edm::View<reco::PFJet> JetCollection;
		typedef PPSGenData  PPSGen; 
		typedef PPSSimData  PPSSim; 
		typedef PPSRecoData PPSReco; 

		edm::Service<TFileService> fs;
		TStyle* style;
		TTree* tree;

		double pv_multiplicity;
		std::vector<double> electron_eta
			, electron_phi
			, electron_pt
			, muon_eta
			, muon_phi
			, muon_pt
			, jet_eta
			, jet_phi
			, jet_pt; 

		// distributions pointers
		std::vector<double> * p_electron_eta
			, * p_electron_phi
			, * p_electron_pt
			, * p_muon_eta
			, * p_muon_phi
			, * p_muon_pt
			, * p_jet_eta
			, * p_jet_phi
			, * p_jet_pt; 

		TH1F * h_pv_multiplicity  // primary vertex multiplicity per event
			, *h_pv_x
			, *h_pv_y
			, *h_pv_z
			, * h_electron_eta     // electron eta
			, * h_electron_phi     // electron phi
			, * h_electron_pt      // electron pt
			, * h_muon_eta         // muon eta
			, * h_muon_phi         // muon phi
			, * h_muon_pt          // muon pt
			, * h_jet_eta          // jet eta
			, * h_jet_phi          // jet phi
			, * h_jet_pt;           // jet pt

		//PPS histograms for Det inside tracks, which means coincidence between Det1, Det2 and ToF
		TH1F *h_PPS_xiARMB, *h_PPS_xiARMF, *h_PPS_tofARMF;
		TH2F *h_PPS_xiVstARMB, *h_PPS_xiVstARMF;  
		TH1F *h_PPS_tARMB, *h_PPS_tARMF, *h_PPS_tofARMB; 
		TH2F *h_PPS_xVsy_ARMFDt1, *h_PPS_xVsy_ARMFDt2, *h_PPS_xVsy_ARMBDt1, *h_PPS_xVsy_ARMBDt2; 
		TH2F *h_PPS_xVsy_ARMFToF, *h_PPS_xVsy_ARMBToF; 
		TH1F *h_PPS_ToF_ARMFToFDet, *h_PPS_ToF_ARMBToFDet; 
		TH2F *h_PPS_xVsy_ARMFTrkDet1, *h_PPS_xVsy_ARMFTrkDet2, *h_PPS_xVsy_ARMFToFDet;
		TH2F *h_PPS_xVsy_ARMBTrkDet1, *h_PPS_xVsy_ARMBTrkDet2, *h_PPS_xVsy_ARMBToFDet;
		TH1F *h_PPS_etaARMF, *h_PPS_etaARMB, *h_PPS_phiARMF, *h_PPS_phiARMB, *h_PPS_thetaXARMF, *h_PPS_thetaYARMF, *h_PPS_thetaXARMB, *h_PPS_thetaYARMB;  
		TH1F *h_PPS_xVertices, *h_PPS_yVertices, *h_PPS_zVertices; 

		// PPS Gen Histograms
		TH1F* h_PPS_Gen_xiARMF,*h_PPS_Gen_tARMF;
		TH1F* h_PPS_Gen_etaARMF,*h_PPS_Gen_phiARMF;	
		TH2F* h_PPS_Gen_xiVs_Gen_tARMF;

		TH1F* h_PPS_Gen_xiARMB,*h_PPS_Gen_tARMB;
		TH1F* h_PPS_Gen_etaARMB,*h_PPS_Gen_phiARMB;
		TH2F* h_PPS_Gen_xiVs_Gen_tARMB;

		//
		edm::Handle<VertexCollection> vertices;
		edm::EDGetTokenT<VertexCollection> verticesToken;
		edm::Handle<ElectronCollection> electrons;
		edm::EDGetTokenT<ElectronCollection> electronsToken;
		edm::Handle<MuonCollection> muons;
		edm::EDGetTokenT<MuonCollection> muonsToken;
		edm::Handle<JetCollection> jets;
		edm::EDGetTokenT<JetCollection> jetsToken;

		//PPS Tokens
		edm::Handle<PPSSpectrometer<PPSGen> > ppsGen;
		edm::EDGetTokenT<PPSSpectrometer<PPSGen> > ppsGenToken;
		edm::Handle<PPSSpectrometer<PPSSim> > ppsSim;
		edm::EDGetTokenT<PPSSpectrometer<PPSSim> > ppsSimToken;
		edm::Handle<PPSSpectrometer<PPSReco> > ppsReco;
		edm::EDGetTokenT<PPSSpectrometer<PPSReco> > ppsRecoToken;

		double EBeam_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
Validation::Validation(const edm::ParameterSet& iConfig):
	p_electron_eta    (&electron_eta)
	, p_electron_phi    (&electron_phi)
	, p_electron_pt     (&electron_pt)
	, p_muon_eta        (&muon_eta)
	, p_muon_phi        (&muon_phi)
	, p_muon_pt         (&muon_pt)
	, p_jet_eta         (&jet_eta)
	, p_jet_phi         (&jet_phi)
	, p_jet_pt          (&jet_pt)
	, verticesToken (consumes<edm::View<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("vertices")))
	, electronsToken (consumes<edm::View<reco::GsfElectron>>(iConfig.getParameter<edm::InputTag>("electrons")))
	, muonsToken (consumes<edm::View<reco::Muon>>(iConfig.getParameter<edm::InputTag>("muons")))
	, jetsToken (consumes<edm::View<reco::PFJet>>(iConfig.getParameter<edm::InputTag>("jets")))
	, ppsGenToken (consumes<PPSSpectrometer<PPSGen>>(iConfig.getParameter<edm::InputTag>("ppsGen")))
	, ppsSimToken (consumes<PPSSpectrometer<PPSSim>>(iConfig.getParameter<edm::InputTag>("ppsSim")))
	, ppsRecoToken (consumes<PPSSpectrometer<PPSReco>>(iConfig.getParameter<edm::InputTag>("ppsReco")))
{
	//now do what ever initialization is needed
	usesResource("TFileService");


}


Validation::~Validation()
{

	// do anything here that needs to be done at desctruction time
	// (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

void Validation::GenInfo(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}
///
void Validation::GenPPSInfo(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	using namespace edm; 
	iEvent.getByToken(ppsGenToken, ppsGen);

	//Arm F 
	for(size_t i=0;i<ppsGen->ArmF.genParticles.size();++i) {
		h_PPS_Gen_xiARMF->Fill(ppsGen->ArmF.genParticles.at(i).xi);
		h_PPS_Gen_tARMF->Fill(ppsGen->ArmF.genParticles.at(i).t);
		h_PPS_Gen_xiVs_Gen_tARMF->Fill(ppsGen->ArmF.genParticles.at(i).xi,ppsGen->ArmF.genParticles.at(i).t);
		h_PPS_Gen_etaARMF->Fill(ppsGen->ArmF.genParticles.at(i).eta);
		h_PPS_Gen_phiARMF->Fill(ppsGen->ArmF.genParticles.at(i).phi);
	}


	//Arm B
	for(size_t i=0;i<ppsGen->ArmB.genParticles.size();++i) {
		h_PPS_Gen_xiARMB->Fill(ppsGen->ArmB.genParticles.at(i).xi);
		h_PPS_Gen_tARMB->Fill(ppsGen->ArmB.genParticles.at(i).t);
		h_PPS_Gen_xiVs_Gen_tARMB->Fill(ppsGen->ArmB.genParticles.at(i).xi,ppsGen->ArmB.genParticles.at(i).t);
		h_PPS_Gen_etaARMB->Fill(ppsGen->ArmB.genParticles.at(i).eta);
		h_PPS_Gen_phiARMB->Fill(ppsGen->ArmB.genParticles.at(i).phi);
	}



}
//
void Validation::PPSResolutionInfo(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}

// ------------ method called for each event  ------------
	void
Validation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	GenPPSInfo(iEvent, iSetup);


	using namespace edm;
	iEvent.getByToken(jetsToken, jets);
	typename JetCollection::const_iterator jet;
	iEvent.getByToken(electronsToken, electrons);
	typename ElectronCollection::const_iterator electron;
	iEvent.getByToken(muonsToken, muons);
	typename MuonCollection::const_iterator muon;
	iEvent.getByToken(verticesToken, vertices);
	typename VertexCollection::const_iterator vertex;

	if(vertices.isValid()) {
		h_pv_multiplicity->Fill(vertices->size());
		pv_multiplicity = vertices->size();
		for (int v = 0 ; v < (int)vertices->size(); v++) {
			h_pv_x->Fill(vertices->at(v).x());
			h_pv_y->Fill(vertices->at(v).y());
			h_pv_z->Fill(vertices->at(v).z());
		}
	}

	if(jets.isValid()) {
		for(jet = jets->begin(); jet != jets->end(); jet++) {
			h_jet_eta->Fill(jet->eta());
			h_jet_phi->Fill(jet->phi());
			h_jet_pt->Fill(jet->pt());
			p_jet_eta->push_back(jet->eta());
			p_jet_phi->push_back(jet->phi());
			p_jet_pt->push_back(jet->pt());
		}
	}
	if(electrons.isValid()) {
		for(electron = electrons->begin(); electron != electrons->end(); electron++) {
			h_electron_eta->Fill(electron->eta());
			h_electron_phi->Fill(electron->phi());
			h_electron_pt->Fill(electron->pt());
			p_electron_eta->push_back(electron->eta());
			p_electron_phi->push_back(electron->phi());
			p_electron_pt->push_back(electron->pt());
		}
	}
	if(muons.isValid()) {
		for(muon = muons->begin(); muon != muons->end(); muon++) {
			h_muon_eta->Fill(muon->eta());
			h_muon_phi->Fill(muon->phi());
			h_muon_pt->Fill(muon->pt());
			p_muon_eta->push_back(muon->eta());
			p_muon_phi->push_back(muon->phi());
			p_muon_pt->push_back(muon->pt());
		}
	}

	iEvent.getByToken(ppsRecoToken, ppsReco);
	for(size_t k=0;k<ppsReco->ArmF.TrkDet1.size();++k) {
		h_PPS_xVsy_ARMFTrkDet1->Fill(ppsReco->ArmF.TrkDet1.at(k).X,ppsReco->ArmF.TrkDet1.at(k).Y);
	}
	for(size_t k=0;k<ppsReco->ArmF.TrkDet2.size();++k) {
		h_PPS_xVsy_ARMFTrkDet2->Fill(ppsReco->ArmF.TrkDet2.at(k).X,ppsReco->ArmF.TrkDet2.at(k).Y);
	}
	for(size_t k=0;k<ppsReco->ArmB.TrkDet1.size();++k) {
		h_PPS_xVsy_ARMBTrkDet1->Fill(ppsReco->ArmB.TrkDet1.at(k).X,ppsReco->ArmB.TrkDet1.at(k).Y);
	}
	for(size_t k=0;k<ppsReco->ArmB.TrkDet2.size();++k) {
		h_PPS_xVsy_ARMBTrkDet2->Fill(ppsReco->ArmB.TrkDet2.at(k).X,ppsReco->ArmB.TrkDet2.at(k).Y);
	}
	for(size_t k=0;k<ppsReco->ArmF.ToFDet.size();++k) {
		h_PPS_xVsy_ARMFToFDet->Fill(ppsReco->ArmF.ToFDet.at(k).X,ppsReco->ArmF.ToFDet.at(k).Y);
		h_PPS_ToF_ARMFToFDet->Fill(ppsReco->ArmF.ToFDet.at(k).ToF);
	}
	for(size_t k=0;k<ppsReco->ArmB.ToFDet.size();++k) {
		h_PPS_xVsy_ARMBToFDet->Fill(ppsReco->ArmB.ToFDet.at(k).X,ppsReco->ArmB.ToFDet.at(k).Y);
		h_PPS_ToF_ARMBToFDet->Fill(ppsReco->ArmB.ToFDet.at(k).ToF);
	}


	for(size_t k=0;k<ppsReco->Vertices->size();++k) {
		h_PPS_xVertices->Fill(ppsReco->Vertices->at(k).x);
		h_PPS_yVertices->Fill(ppsReco->Vertices->at(k).y);
		h_PPS_zVertices->Fill(ppsReco->Vertices->at(k).z);
	}
	for(size_t i=0;i<ppsReco->ArmF.Tracks.size();++i) {
		h_PPS_xiARMF->Fill(ppsReco->ArmF.Tracks.at(i).xi);
		h_PPS_tARMF->Fill(ppsReco->ArmF.Tracks.at(i).t);
		h_PPS_xiVstARMF->Fill(ppsReco->ArmF.Tracks.at(i).xi,ppsReco->ArmF.Tracks.at(i).t);
		h_PPS_etaARMF->Fill(ppsReco->ArmF.Tracks.at(i).eta);
		h_PPS_phiARMF->Fill(ppsReco->ArmF.Tracks.at(i).phi);
		h_PPS_thetaXARMF->Fill(ppsReco->ArmF.Tracks.at(i).thetaX);
		h_PPS_thetaYARMF->Fill(ppsReco->ArmF.Tracks.at(i).thetaY);
		h_PPS_tofARMF->Fill(ppsReco->ArmF.Tracks.at(i).ToF.ToF);
		h_PPS_xVsy_ARMFDt1->Fill(ppsReco->ArmF.Tracks.at(i).Det1.X,ppsReco->ArmF.Tracks.at(i).Det1.Y);
		h_PPS_xVsy_ARMFDt2->Fill(ppsReco->ArmF.Tracks.at(i).Det2.X,ppsReco->ArmF.Tracks.at(i).Det2.Y);
		if(ppsReco->ArmF.Tracks.at(i).ToF.ToF!=0)h_PPS_xVsy_ARMFToF->Fill(ppsReco->ArmF.Tracks.at(i).ToF.X,ppsReco->ArmF.Tracks.at(i).ToF.Y); 
	}
	for(size_t j=0;j<ppsReco->ArmB.Tracks.size();++j) {
		h_PPS_xiARMB->Fill(ppsReco->ArmB.Tracks.at(j).xi);
		h_PPS_tARMB->Fill(ppsReco->ArmB.Tracks.at(j).t);
		h_PPS_xiVstARMB->Fill(ppsReco->ArmB.Tracks.at(j).xi,ppsReco->ArmB.Tracks.at(j).t);
		h_PPS_etaARMB->Fill(ppsReco->ArmB.Tracks.at(j).eta);
		if(ppsReco->ArmB.Tracks.at(j).phi<0.) h_PPS_phiARMB->Fill(ppsReco->ArmB.Tracks.at(j).phi+TMath::Pi());
		else if(ppsReco->ArmB.Tracks.at(j).phi>=0.) h_PPS_phiARMB->Fill(ppsReco->ArmB.Tracks.at(j).phi-TMath::Pi());
		h_PPS_thetaXARMB->Fill(ppsReco->ArmB.Tracks.at(j).thetaX);
		h_PPS_thetaYARMB->Fill(ppsReco->ArmB.Tracks.at(j).thetaY);
		h_PPS_tofARMB->Fill(ppsReco->ArmB.Tracks.at(j).ToF.ToF);
		h_PPS_xVsy_ARMBDt1->Fill(ppsReco->ArmB.Tracks.at(j).Det1.X,ppsReco->ArmB.Tracks.at(j).Det1.Y);
		h_PPS_xVsy_ARMBDt2->Fill(ppsReco->ArmB.Tracks.at(j).Det2.X,ppsReco->ArmB.Tracks.at(j).Det2.Y);
		if(ppsReco->ArmB.Tracks.at(j).ToF.ToF!=0)h_PPS_xVsy_ARMBToF->Fill(ppsReco->ArmB.Tracks.at(j).ToF.X,ppsReco->ArmB.Tracks.at(j).ToF.Y);  
	}

}


// ------------ method called once each job just before starting event loop  ------------
	void 
Validation::beginJob()
{
	h_PPS_xiARMF = fs->make<TH1F>( "PPS_xiARMF" , "PPS_xiARMF; #xi; nEvents" , 100, 0., .50 ); 
	h_PPS_tARMF = fs->make<TH1F>( "PPS_tARMF" , "PPS_tARMF; t [GeV^{2}]; nEvents" , 100, .0, 3.0 );
	h_PPS_xiVstARMF = fs->make<TH2F>( "PPS_xiVstARMF" , "PPS_xiVstARMF; #xi ; t [GeV^{2}]" , 100, 0., .50 , 100, .0, 3.0 );
	h_PPS_etaARMF = fs->make<TH1F>( "PPS_etaARMF" , "PPS_etaARMF; #eta; nEvents" , 100, 8., 15.); 
	h_PPS_phiARMF = fs->make<TH1F>( "PPS_phiARMF" , "PPS_phiARMF; #phi; nEvents" , 100, -3.15, 3.15 ); 
	h_PPS_thetaXARMF = fs->make<TH1F>( "PPS_thetaXARMF" , "PPS_thetaXARMF;  #theta_{X}[#murad]; nEvents" , 100, -300., 300. ); 
	h_PPS_thetaYARMF = fs->make<TH1F>( "PPS_thetaYARMF" , "PPS_thetaYARMF;  #theta_{Y}[#murad]; nEvents" , 100, -300., 300. ); 
	h_PPS_tofARMF = fs->make<TH1F>( "PPS_tofARMF" , "Hits in tracks - PPS_tofARMF;  ToF [#mus]; nEvents" , 100, 715., 722. ); 
	h_PPS_xVsy_ARMFDt1 = fs->make<TH2F>( "PPS_xVsy_ARMFDt1" , "Hits in tracks - PPS x_vs_y_{ARMFDt1}; x [mm]; y [mm]" , 22, -25.0, -3.0,18,-9.0,9.0 );
	h_PPS_xVsy_ARMFDt2 = fs->make<TH2F>( "PPS_xVsy_ARMFDt2" , "Hits in tracks - PPS x_vs_y_{ARMFDt2}; x [mm]; y [mm]" , 22, -25.0, -3.0,18,-9.0,9.0 );
	h_PPS_xVsy_ARMFToF = fs->make<TH2F>( "PPS_xVsy_ARMFToF" , "Hits in tracks - PPS x_vs_y_{ARMFToF}; x [mm]; y [mm]" ,  5, -16.75, -1.75,4,-6.0,6.0);
	h_PPS_xiARMB = fs->make<TH1F>( "PPS_xiARMB" , "PPS_xiARMB; #xi; nEvents" , 100, 0., .50 ); 
	h_PPS_tARMB = fs->make<TH1F>( "PPS_tARMB" , "PPS_tARMB; t [GeV^{2}]; nEvents" , 100, .0, 3.0 ); 
	h_PPS_xiVstARMB = fs->make<TH2F>( "PPS_xiVstARMB" , "PPS_xiVstARMB; #xi ; t [GeV^{2}]" , 100, 0., .50 , 100, .0, 3.0 );
	h_PPS_etaARMB = fs->make<TH1F>( "PPS_etaARMB" , "PPS_etaARMB; #eta; nEvents" , 100, -15., -8. ); 
	h_PPS_phiARMB = fs->make<TH1F>( "PPS_phiARMB" , "PPS_phiARMB;|#phi - #pi|; nEvents" , 100, -3.15, 3.15 ); 
	h_PPS_thetaXARMB = fs->make<TH1F>( "PPS_thetaXARMB" , "PPS_thetaXARMB;  #theta_{X}[#murad]; nEvents" , 100, -300., 300. ); 
	h_PPS_thetaYARMB = fs->make<TH1F>( "PPS_thetaYARMB" , "PPS_thetaYARMB;  #theta_{Y}[#murad]; nEvents" , 100, -300., 300. ); 
	h_PPS_tofARMB = fs->make<TH1F>( "PPS_tofARMB" , "Hits in tracks - PPS_tofARMB; ToF [#mus]; nEvents" , 100, 715., 722. ); 
	h_PPS_xVsy_ARMBDt1 = fs->make<TH2F>( "PPS_xVsy_ARMBDt1" , "Hits in tracks - PPS x_vs_y_{ARMBDt1}; x [mm]; y [mm]" , 22, -25.0, -3.0,18,-9.0,9.0 );
	h_PPS_xVsy_ARMBDt2 = fs->make<TH2F>( "PPS_xVsy_ARMBDt2" , "Hits in tracks - PPS x_vs_y_{ARMBDt2}; x [mm]; y [mm]" , 22, -25.0, -3.0,18,-9.0,9.0 );
	h_PPS_xVsy_ARMBToF = fs->make<TH2F>( "PPS_xVsy_ARMBToF" , "Hits in tracks - PPS x_vs_y_{ARMBToF}; x [mm]; y [mm]" , 5, -16.75, -1.75,4,-6.0,6.0);

	h_PPS_xVertices = fs->make<TH1F>( "PPS_xVertices" , "PPS_xVertices; x_{vPPS} [cm]; nEvents" , 100, -.40, .20 ); 
	h_PPS_yVertices = fs->make<TH1F>( "PPS_yVertices" , "PPS_yVertices; y_{vPPS} [cm]; nEvents" , 100, -.10, .10 ); 
	h_PPS_zVertices = fs->make<TH1F>( "PPS_zVertices" , "PPS_zVertices; z_{vPPS} [cm]; nEvents" , 100, -20., 20. ); 

	h_PPS_xVsy_ARMFTrkDet1 = fs->make<TH2F>( "PPS_xVsy_ARMFTrkDet1" , "All hits - PPS x_vs_y_{TrkDet1_ARMF}; x [mm]; y [mm]" , 22, -25.0, -3.0,18,-9.0,9.0 );
	h_PPS_xVsy_ARMFTrkDet2 = fs->make<TH2F>( "PPS_xVsy_ARMFTrkDet2" , "All hits - PPS x_vs_y_{TrkDet2_ARMF}; x [mm]; y [mm]" , 22, -25.0, -3.0,18,-9.0,9.0 );
	h_PPS_xVsy_ARMFToFDet = fs->make<TH2F>( "PPS_xVsy_ARMFToFDet" , "All hits - PPS x_vs_y_{ARMFToFDet}; x [mm]; y [mm]" , 5, -16.75, -1.75,4,-6.0,6.0);
	h_PPS_ToF_ARMFToFDet = fs->make<TH1F>( "PPS_ToF_ARMFToFDet" , "All hits - PPS_ToF_ARMFToFDet;  ToF [#mus]; nEvents" , 100, 700., 740. );
	h_PPS_xVsy_ARMBTrkDet1 = fs->make<TH2F>( "PPS_xVsy_ARMBTrkDet1" , "All hits - PPS x_vs_y_{TrkDet1_ARMB}; x [mm]; y [mm]" , 22, -25.0, -3.0,18,-9.0,9.0 );
	h_PPS_xVsy_ARMBTrkDet2 = fs->make<TH2F>( "PPS_xVsy_ARMBTrkDet2" , "All hits - PPS x_vs_y_{TrkDet2_ARMB}; x [mm]; y [mm]" , 22, -25.0, -3.0,18,-9.0,9.0 );
	h_PPS_xVsy_ARMBToFDet = fs->make<TH2F>( "PPS_xVsy_ARMBToFDet" , "All hits - PPS x_vs_y_{ARMBToFDet}; x [mm]; y [mm]" , 5, -16.75, -1.75,4,-6.0,6.0);
	h_PPS_ToF_ARMBToFDet = fs->make<TH1F>( "PPS_ToF_ARMBToFDet" , "All hits - PPS_ToF_ARMBToFDet;  ToF [#mus]; nEvents" , 100, 700., 740. );

	//Gen PPS histo
	h_PPS_Gen_xiARMB = fs->make<TH1F>( "PPS_Gen_xiARMB" , "PPS_Gen_xiARMB; #xi; nEvents" , 100, 0., .50 );
	h_PPS_Gen_tARMB = fs->make<TH1F>( "PPS_Gen_tARMB" , "PPS_Gen_tARMB; t [GeV^{2}]; nEvents" , 100, .0, 3.0 );
	h_PPS_Gen_xiVs_Gen_tARMB = fs->make<TH2F>( "PPS_Gen_xiVs_Gen_tARMB" , "PPS_Gen_xiVs_Gen_tARMB; #xi ; t [GeV^{2}]" , 100, 0., .50 , 100, .0, 3.0 );
	h_PPS_Gen_etaARMB = fs->make<TH1F>( "PPS_Gen_etaARMB" , "PPS_Gen_etaARMB; #eta; nEvents" , 100, -15., -8. );
	h_PPS_Gen_phiARMB = fs->make<TH1F>( "PPS_Gen_phiARMB" , "PPS_Gen_phiARMB;|#phi - #pi|; nEvents" , 100, -3.15, 3.15 );

	h_PPS_Gen_xiARMF = fs->make<TH1F>( "PPS_Gen_xiARMF" , "PPS_Gen_xiARMF; #xi; nEvents" , 100, 0., .50 ); 
	h_PPS_Gen_tARMF = fs->make<TH1F>( "PPS_Gen_tARMF" , "PPS_Gen_tARMF; t [GeV^{2}]; nEvents" , 100, .0, 3.0 );
	h_PPS_Gen_xiVs_Gen_tARMF = fs->make<TH2F>( "PPS_Gen_xiVs_Gen_tARMF" , "PPS_Gen_xiVs_Gen_tARMF; #xi ; t [GeV^{2}]" , 100, 0., .50 , 100, .0, 3.0 ); 
	h_PPS_Gen_etaARMF = fs->make<TH1F>( "PPS_Gen_etaARMF" , "PPS_Gen_etaARMF; #eta; nEvents" , 100, 8., 15.);  
	h_PPS_Gen_phiARMF = fs->make<TH1F>( "PPS_Gen_phiARMF" , "PPS_Gen_phiARMF;|#phi - #pi|; nEvents" , 100, -3.15, 3.15 );

	h_pv_multiplicity  = fs->make<TH1F>( "h_pv_multiplicity",  "h_pv_multiplicity", 50,   -0.5,   50.5 );
	h_pv_x             = fs->make<TH1F>( "h_pv_x",             "h_pv_x", 100,   -0.5,  0.5 );
	h_pv_y             = fs->make<TH1F>( "h_pv_y",             "h_pv_y", 100,   -0.5,  0.5 );
	h_pv_z             = fs->make<TH1F>( "h_pv_z",             "h_pv_z", 120,   -30.,  30. );
	h_electron_eta     = fs->make<TH1F>( "h_electron_eta",     "h_electron_eta", 100, -3.4,    3.4 );
	h_electron_phi     = fs->make<TH1F>( "h_electron_phi",     "h_electron_phi", 100, -3.2,    3.2 );
	h_electron_pt      = fs->make<TH1F>( "h_electron_pt",      "h_electron_pt", 100,  0.0,    50. );
	h_muon_eta         = fs->make<TH1F>( "h_muon_eta",         "h_muon_eta", 100, -3.4,    3.4 );
	h_muon_phi         = fs->make<TH1F>( "h_muon_phi",         "h_muon_phi", 100, -3.2,    3.2 );
	h_muon_pt          = fs->make<TH1F>( "h_muon_pt",          "h_muon_pt", 100,  0.0,    10. );
	h_jet_eta          = fs->make<TH1F>( "h_jet_eta",          "h_jet_eta", 100,  -6.,    6.0 );
	h_jet_phi          = fs->make<TH1F>( "h_jet_phi",          "h_jet_phi", 100, -3.2,    3.2 );
	h_jet_pt           = fs->make<TH1F>( "h_jet_pt",           "h_jet_pt", 200,  0.0, 100. );

	// set histos parameters
	h_PPS_xVsy_ARMFDt1->SetOption("COLZ");
	h_PPS_xVsy_ARMFDt2->SetOption("COLZ");
	h_PPS_xVsy_ARMFToF->SetOption("COLZ");
	h_PPS_xiVstARMF->SetOption("COLZ");
	h_PPS_xVsy_ARMFDt1->SetStats(kFALSE);
	h_PPS_xVsy_ARMFDt2->SetStats(kFALSE);
	h_PPS_xVsy_ARMFToF->SetStats(kFALSE);
	h_PPS_xiVstARMF->SetStats(kFALSE);

	h_PPS_Gen_xiVs_Gen_tARMF->SetOption("COLZ"); 
	h_PPS_Gen_xiVs_Gen_tARMB->SetOption("COLZ");
	h_PPS_Gen_xiVs_Gen_tARMF->SetStats(kFALSE);  
	h_PPS_Gen_xiVs_Gen_tARMB->SetStats(kFALSE);

	h_PPS_xVsy_ARMBDt1->SetOption("COLZ");
	h_PPS_xVsy_ARMBDt2->SetOption("COLZ");
	h_PPS_xVsy_ARMBToF->SetOption("COLZ");
	h_PPS_xiVstARMB->SetOption("COLZ");
	h_PPS_xVsy_ARMBDt1->SetStats(kFALSE);
	h_PPS_xVsy_ARMBDt2->SetStats(kFALSE);
	h_PPS_xVsy_ARMBToF->SetStats(kFALSE);
	h_PPS_xiVstARMB->SetStats(kFALSE);
	h_PPS_xVsy_ARMFTrkDet1->SetOption("COLZ");
	h_PPS_xVsy_ARMFTrkDet2->SetOption("COLZ");
	h_PPS_xVsy_ARMFToFDet->SetOption("COLZ");
	h_PPS_xVsy_ARMFTrkDet1->SetStats(kFALSE);
	h_PPS_xVsy_ARMFTrkDet2->SetStats(kFALSE);
	h_PPS_xVsy_ARMFToFDet->SetStats(kFALSE);
	h_PPS_xVsy_ARMBTrkDet1->SetOption("COLZ");
	h_PPS_xVsy_ARMBTrkDet2->SetOption("COLZ");
	h_PPS_xVsy_ARMBToFDet->SetOption("COLZ");
	h_PPS_xVsy_ARMBTrkDet1->SetStats(kFALSE);
	h_PPS_xVsy_ARMBTrkDet2->SetStats(kFALSE);
	h_PPS_xVsy_ARMBToFDet->SetStats(kFALSE);
	style->SetPalette(1);
	h_pv_multiplicity->GetXaxis()->SetTitle("Primary vertex multiplicity");
	h_pv_x->GetXaxis()->SetTitle("Primary vertex x [cm]");
	h_pv_y->GetXaxis()->SetTitle("Primary vertex y [cm]");
	h_pv_x->GetXaxis()->SetTitle("Primary vertex z [cm]");
	h_electron_eta->GetXaxis()->SetTitle("#eta");
	h_electron_phi->GetXaxis()->SetTitle("#phi [rad]");
	h_electron_pt->GetXaxis()->SetTitle("pT [GeV]");
	h_muon_eta->GetXaxis()->SetTitle("#eta");
	h_muon_phi->GetXaxis()->SetTitle("#phi [rad]");
	h_muon_pt->GetXaxis()->SetTitle("pT [GeV]");
	h_jet_eta->GetXaxis()->SetTitle("#eta");
	h_jet_phi->GetXaxis()->SetTitle("#phi [rad]");
	h_jet_pt->GetXaxis()->SetTitle("pT [GeV]");
	h_pv_multiplicity->GetYaxis()->SetTitle("Events");
	h_pv_x->GetYaxis()->SetTitle("Events");
	h_pv_y->GetYaxis()->SetTitle("Events");
	h_pv_z->GetYaxis()->SetTitle("Events");
	h_electron_eta->GetYaxis()->SetTitle("Events");
	h_electron_phi->GetYaxis()->SetTitle("Events");
	h_electron_pt->GetYaxis()->SetTitle("Events");
	h_muon_eta->GetYaxis()->SetTitle("Events");
	h_muon_phi->GetYaxis()->SetTitle("Events");
	h_muon_pt->GetYaxis()->SetTitle("Events");
	h_jet_eta->GetYaxis()->SetTitle("Events");
	h_jet_phi->GetYaxis()->SetTitle("Events");
	h_jet_pt->GetYaxis()->SetTitle("Events");

	// initialize variables
	pv_multiplicity = 99999.;

	electron_eta.clear();
	electron_phi.clear();
	electron_pt.clear();
	muon_eta.clear();
	muon_phi.clear();
	muon_pt.clear();
	jet_eta.clear();
	jet_phi.clear();

	// set tree parameters
	tree = new TTree("Events","Events");
	tree->Branch("pv_multiplicity",&pv_multiplicity,"pv_multiplicity/D");
	tree->Branch("p_electron_eta","std::vector<double>",&p_electron_eta);
	tree->Branch("p_electron_phi","std::vector<double>",&p_electron_phi);
	tree->Branch("p_electron_pt","std::vector<double>",&p_electron_pt);
	tree->Branch("p_muon_eta","std::vector<double>",&p_muon_eta);
	tree->Branch("p_muon_phi","std::vector<double>",&p_muon_phi);
	tree->Branch("p_muon_pt","std::vector<double>",&p_muon_pt);
	tree->Branch("p_jet_eta","std::vector<double>",&p_jet_eta);
	tree->Branch("p_jet_phi","std::vector<double>",&p_jet_phi);
	tree->Branch("p_jet_pt","std::vector<double>",&p_jet_pt);
}

// ------------ method called once each job just after ending the event loop  ------------
	void 
Validation::endJob() 
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
Validation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	//The following says we do not know what parameters are allowed so do no Validation
	// Please change this to state exactly what you do use, even if it is no parameters
	edm::ParameterSetDescription desc;
	desc.setUnknown();
	descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Validation);
