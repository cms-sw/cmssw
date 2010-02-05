// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
using namespace std;

#include "JetMETCorrections/TauJet/interface/TCTauCorrector.h"
#include "JetMETCorrections/TauJet/interface/TauJetCorrector.h"

#include "DataFormats/TauReco/interface/CaloTau.h"
#include "RecoTauTag/TauTagTools/interface/CaloTauElementsOperators.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminator.h"


#include "DataFormats/TauReco/interface/PFTau.h"
#include "RecoTauTag/TauTagTools/interface/PFTauElementsOperators.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "TFile.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TLatex.h"
#include "TLine.h"
#include "TTree.h"

//#include "JetMETCorrections/TauJet/test/visibleTaus.h"

//typedef math::XYZTLorentzVectorD LorentzVector;
//typedef std::vector<LorentzVector> LorentzVectorCollection;

class TCTauAnalysis : public edm::EDAnalyzer {
  public:
  	explicit TCTauAnalysis(const edm::ParameterSet&);
  	~TCTauAnalysis();

  	virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  	virtual void beginJob();
  	virtual void endJob();

  private:
	void resetNtuple();

	TFile* outFile;
	TTree* tauTree;

	float MCTau_pt,MCTau_eta,MCTau_phi;
	float PFTau_pt,PFTau_eta,PFTau_phi,PFTau_nProngs,PFTau_ltrackPt,PFTau_d_isol,PFTau_d_1,PFTau_d_2;
	float CaloTau_pt,CaloTau_eta,CaloTau_phi,CaloTau_nProngs,CaloTau_ltrackPt,CaloTau_d_isol,CaloTau_d_1,CaloTau_d_2;
	float TCTau_pt,TCTau_eta,TCTau_phi,TCTau_nProngs,TCTau_ltrackPt,TCTau_d_isol,TCTau_d_1,TCTau_d_2;

	int nMCTaus,
            nCaloTaus,
            nTCTaus,
	    nPFTaus;

	double tauEtCut,tauEtaCut;

	edm::InputTag CaloTaus;
	edm::InputTag TCTaus;
	edm::InputTag PFTaus;
	edm::InputTag MCTaus;
	edm::InputTag Discriminator;
};

TCTauAnalysis::TCTauAnalysis(const edm::ParameterSet& iConfig){

	outFile = TFile::Open("tctau.root", "RECREATE");
        tauTree = new TTree("tauTree","tauTree");

	nMCTaus   = 0;
	nCaloTaus = 0;
	nTCTaus   = 0;
	nPFTaus   = 0;

        MCTau_pt  = 0; tauTree->Branch("MCTau_pt",  &MCTau_pt,  "MCTau_pt/F");
        MCTau_eta = 0; tauTree->Branch("MCTau_eta", &MCTau_eta, "MCTau_eta/F");
        MCTau_phi = 0; tauTree->Branch("MCTau_phi", &MCTau_phi, "MCTau_phi/F");

        PFTau_pt  = 0; tauTree->Branch("PFTau_pt",  &PFTau_pt,  "PFTau_pt/F");
        PFTau_eta = 0; tauTree->Branch("PFTau_eta", &PFTau_eta, "PFTau_eta/F");
        PFTau_phi = 0; tauTree->Branch("PFTau_phi", &PFTau_phi, "PFTau_phi/F");
	PFTau_nProngs = 0;  tauTree->Branch("PFTau_nProngs",  &PFTau_nProngs,  "PFTau_nProngs/F");
	PFTau_ltrackPt = 0; tauTree->Branch("PFTau_ltrackPt",  &PFTau_ltrackPt,  "PFTau_ltrackPt/F");
	PFTau_d_isol = 0;   tauTree->Branch("PFTau_d_isol",  &PFTau_d_isol,  "PFTau_d_isol/F"); //DiscriminationByIsolation
	PFTau_d_1 = 0;      tauTree->Branch("PFTau_d_1",  &PFTau_d_1,  "PFTau_d_1/F");
	PFTau_d_2 = 0;      tauTree->Branch("PFTau_d_2",  &PFTau_d_2,  "PFTau_d_2/F");

        CaloTau_pt  = 0; tauTree->Branch("CaloTau_pt",  &CaloTau_pt,  "CaloTau_pt/F");
        CaloTau_eta = 0; tauTree->Branch("CaloTau_eta", &CaloTau_eta, "CaloTau_eta/F");
        CaloTau_phi = 0; tauTree->Branch("CaloTau_phi", &CaloTau_phi, "CaloTau_phi/F");
        CaloTau_nProngs = 0;  tauTree->Branch("CaloTau_nProngs",  &CaloTau_nProngs,  "CaloTau_nProngs/F");
        CaloTau_ltrackPt = 0; tauTree->Branch("CaloTau_ltrackPt",  &CaloTau_ltrackPt,  "CaloTau_ltrackPt/F");
	CaloTau_d_isol = 0;   tauTree->Branch("CaloTau_d_isol",  &CaloTau_d_isol,  "CaloTau_d_isol/F");
        CaloTau_d_1 = 0;      tauTree->Branch("CaloTau_d_1",  &CaloTau_d_1,  "CaloTau_d_1/F");
        CaloTau_d_2 = 0;      tauTree->Branch("CaloTau_d_2",  &CaloTau_d_2,  "CaloTau_d_2/F");

        TCTau_pt  = 0; tauTree->Branch("TCTau_pt",  &TCTau_pt,  "TCTau_pt/F");
        TCTau_eta = 0; tauTree->Branch("TCTau_eta", &TCTau_eta, "TCTau_eta/F");
        TCTau_phi = 0; tauTree->Branch("TCTau_phi", &TCTau_phi, "TCTau_phi/F");
        TCTau_nProngs = 0;  tauTree->Branch("TCTau_nProngs",  &TCTau_nProngs,  "TCTau_nProngs/F");
        TCTau_ltrackPt = 0; tauTree->Branch("TCTau_ltrackPt",  &TCTau_ltrackPt,  "TCTau_ltrackPt/F");
        TCTau_d_isol = 0;   tauTree->Branch("TCTau_d_isol",  &TCTau_d_isol,  "TCTau_d_isol/F");
        TCTau_d_1 = 0;      tauTree->Branch("TCTau_d_1",  &TCTau_d_1,  "TCTau_d_1/F");
        TCTau_d_2 = 0;      tauTree->Branch("TCTau_d_2",  &TCTau_d_2,  "TCTau_d_2/F");

        tauEtCut  	= iConfig.getParameter<double>("TauJetEt");
        tauEtaCut 	= iConfig.getParameter<double>("TauJetEta");
        CaloTaus  	= iConfig.getParameter<edm::InputTag>("CaloTauCollection");
        TCTaus    	= iConfig.getParameter<edm::InputTag>("TCTauCollection");
        PFTaus    	= iConfig.getParameter<edm::InputTag>("PFTauCollection");
	MCTaus    	= iConfig.getParameter<edm::InputTag>("MCTauCollection");
	Discriminator 	= iConfig.getParameter<edm::InputTag>("Discriminator");
}

TCTauAnalysis::~TCTauAnalysis(){

        cout << endl << endl;
        cout << "MCTaus     " << nMCTaus << endl;
        cout << "CaloTaus   " << nCaloTaus << endl;
        cout << "TCTaus     " << nTCTaus << endl;
        cout << "PFTaus     " << nPFTaus << endl;

/*
        TH1F* h_tcTauEff = new TH1F("h_tcTauEff","",4,0,4);
        h_tcTauEff->SetBinContent(1,nMCTaus);
        h_tcTauEff->GetXaxis()->SetBinLabel(1,"nMCTaus");
        h_tcTauEff->SetBinContent(2,nCaloTaus);
        h_tcTauEff->GetXaxis()->SetBinLabel(2,"nCaloTaus");
        h_tcTauEff->SetBinContent(3,nTCTaus);
        h_tcTauEff->GetXaxis()->SetBinLabel(3,"nTCTaus");
        h_tcTauEff->SetBinContent(4,nPFTaus);
        h_tcTauEff->GetXaxis()->SetBinLabel(4,"nPFTaus");
        h_tcTauEff->Write();
*/
	outFile->cd();
        tauTree->Write();
        outFile->Close();

}


void TCTauAnalysis::resetNtuple(){
        MCTau_pt  = 0;
        MCTau_eta = 0;
        MCTau_phi = 0;

        PFTau_pt  = 0;
        PFTau_eta = 0;
        PFTau_phi = 0;
        PFTau_nProngs = 0;
        PFTau_ltrackPt = 0;
        PFTau_d_isol = 0;
        PFTau_d_1 = 0;
        PFTau_d_2 = 0;

        CaloTau_pt  = 0;
        CaloTau_eta = 0;
        CaloTau_phi = 0;
        CaloTau_nProngs = 0;
        CaloTau_ltrackPt = 0;
        CaloTau_d_isol = 0;
        CaloTau_d_1 = 0;
        CaloTau_d_2 = 0;

        TCTau_pt  = 0;
        TCTau_eta = 0;
        TCTau_phi = 0;
        TCTau_nProngs = 0;
        TCTau_ltrackPt = 0;
	TCTau_d_isol = 0;
        TCTau_d_1 = 0;
        TCTau_d_2 = 0;
}

void TCTauAnalysis::beginJob(){}

void TCTauAnalysis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

        double matchingConeSize         = 0.1,
               signalConeSize           = 0.07,
               isolationConeSize        = 0.4,
               ptLeadingTrackMin        = 6,
               ptOtherTracksMin         = 1;
        string metric = "DR"; // can be DR,angle,area
        unsigned int isolationAnnulus_Tracksmaxn = 0;

	Handle<CaloTauCollection> theCaloTauHandle;
	iEvent.getByLabel(CaloTaus,theCaloTauHandle);
	Handle<CaloTauDiscriminator> theCaloTauDiscriminatorHandle;
        iEvent.getByLabel(InputTag("caloRecoTau"+Discriminator.label(),"",CaloTaus.process()),
                          theCaloTauDiscriminatorHandle);

        Handle<CaloTauCollection> theTCTauHandle;
        iEvent.getByLabel(TCTaus,theTCTauHandle);
	Handle<CaloTauDiscriminator> theTCTauDiscriminatorHandle;
        iEvent.getByLabel(InputTag("caloRecoTau"+Discriminator.label()),
                          theTCTauDiscriminatorHandle);

	string pfTau = PFTaus.label();
	pfTau = pfTau.substr(0,pfTau.find("Producer"));
        Handle<PFTauCollection> thePFTauHandle;
        iEvent.getByLabel(PFTaus,thePFTauHandle);
	Handle<PFTauDiscriminator> thePFTauDiscriminatorHandle;
	iEvent.getByLabel(pfTau+Discriminator.label(),thePFTauDiscriminatorHandle);


// MCTaus
	edm::Handle<std::vector<math::XYZTLorentzVectorD> > mcTaus;
	iEvent.getByLabel(MCTaus, mcTaus);
	if(mcTaus.isValid()){

	  std::vector<math::XYZTLorentzVectorD>::const_iterator i;
	  for(i = mcTaus->begin(); i!= mcTaus->end(); ++i){

		resetNtuple();

		if(i->pt() < tauEtCut || fabs(i->eta()) > tauEtaCut) continue;

		nMCTaus++;

		MCTau_pt = i->Pt();
		MCTau_eta = i->Eta();
		MCTau_phi = i->Phi();
		cout << "MCTau pt = " << MCTau_pt << endl;

// CaloTaus
		if(theCaloTauHandle.isValid()){

                  for(unsigned int iTau = 0; iTau < theCaloTauHandle->size(); ++iTau){
                        CaloTauRef theTau(theCaloTauHandle,iTau);
			//cout << "CaloTauRef id " << theTau.id() << endl;
			double DR = ROOT::Math::VectorUtil::DeltaR(*i, theTau->p4());

			if(DR > 0.5) continue;

	                if(!theTau->leadTrack()) continue;

	                CaloTau theCaloTau = *theTau;
	                CaloTauElementsOperators op(theCaloTau);
	                double d_trackIsolation = op.discriminatorByIsolTracksN(
                                metric,
                                matchingConeSize,
                                ptLeadingTrackMin,
                                ptOtherTracksMin,
                                metric,
                                signalConeSize,
                                metric,
                                isolationConeSize,
                                isolationAnnulus_Tracksmaxn);
	                if(d_trackIsolation == 0) continue;

	                const TrackRef leadingTrack =op.leadTk(metric,matchingConeSize,ptLeadingTrackMin);
	                if(leadingTrack.isNull()) continue;

			double theDiscriminator = (*theCaloTauDiscriminatorHandle)[theTau];

	                const TrackRefVector signalTracks = op.tracksInCone(leadingTrack->momentum(),metric,signalConeSize,ptOtherTracksMin);

			cout << "CaloTau Et = " << theCaloTau.pt() <<endl;

			nCaloTaus++;

			CaloTau_pt       = theCaloTau.pt();
			CaloTau_eta      = theCaloTau.eta();
			CaloTau_phi      = theCaloTau.phi();
			CaloTau_nProngs  = signalTracks.size();
			CaloTau_ltrackPt = leadingTrack->pt();
			CaloTau_d_isol   = theDiscriminator;
			CaloTau_d_1	 = d_trackIsolation;
		  }
		}

// TCTaus
                if(theTCTauHandle.isValid()){

                  for(unsigned int iTau = 0; iTau < theTCTauHandle->size(); ++iTau){
                        CaloTauRef theTau(theTCTauHandle,iTau);

                        double DR = ROOT::Math::VectorUtil::DeltaR(*i, theTau->p4());

                        if(DR > 0.5) continue;

                        if(!theTau->leadTrack()) continue;

                        CaloTau jptTCTauCorrected = *theTau;
                        CaloTauElementsOperators op(jptTCTauCorrected);
                        double d_trackIsolation = op.discriminatorByIsolTracksN(
                                metric,
                                matchingConeSize,
                                ptLeadingTrackMin,
                                ptOtherTracksMin,
                                metric,
                                signalConeSize,
                                metric,
                                isolationConeSize,
                                isolationAnnulus_Tracksmaxn);
                        if(d_trackIsolation == 0) continue;

                        const TrackRef leadingTrack =op.leadTk(metric,matchingConeSize,ptLeadingTrackMin);
                        if(leadingTrack.isNull()) continue;

                        double theDiscriminator = (*theTCTauDiscriminatorHandle)[theTau];

                        const TrackRefVector signalTracks = op.tracksInCone(leadingTrack->momentum(),metric,signalConeSize,ptOtherTracksMin);

	                cout << "CaloTau+JPT+TCTau Et = " << jptTCTauCorrected.pt() <<endl;

			nTCTaus++;

                        TCTau_pt       = jptTCTauCorrected.pt();
                        TCTau_eta      = jptTCTauCorrected.eta();
                        TCTau_phi      = jptTCTauCorrected.phi();
                        TCTau_nProngs  = signalTracks.size();
                        TCTau_ltrackPt = leadingTrack->pt();
			TCTau_d_isol   = theDiscriminator;
			TCTau_d_1      = d_trackIsolation;
		  }
		}

// PFTaus
		if(thePFTauHandle.isValid()){

		  for(unsigned int iTau = 0; iTau < thePFTauHandle->size(); ++iTau){
			PFTauRef thePFTau(thePFTauHandle,iTau);

                        double DR = ROOT::Math::VectorUtil::DeltaR(*i,thePFTau->p4());
                        if(DR > 0.5) continue;

                        if(!thePFTau->leadTrack()) continue;

			double theDiscriminator = (*thePFTauDiscriminatorHandle)[thePFTau];

	                PFTau theCaloTau = *thePFTau;
	                PFTauElementsOperators op(theCaloTau);
	                double d_trackIsolation = op.discriminatorByIsolTracksN(
	                                metric,
	                                matchingConeSize,
	                                ptLeadingTrackMin,
	                                ptOtherTracksMin,
	                                metric,
	                                signalConeSize,
	                                metric,
	                                isolationConeSize,
	                                isolationAnnulus_Tracksmaxn);
	                if(d_trackIsolation == 0) continue;

                        const TrackRef leadingTrack =op.leadTk(metric,matchingConeSize,ptLeadingTrackMin);
                        if(leadingTrack.isNull()) continue;

                        const TrackRefVector signalTracks = op.tracksInCone(leadingTrack->momentum(),metric,signalConeSize,ptOtherTracksMin);

			cout << "PFTau Et = " << theCaloTau.pt() <<endl;

			nPFTaus++;

                        PFTau_pt       = theCaloTau.pt();
                        PFTau_eta      = theCaloTau.eta();
                        PFTau_phi      = theCaloTau.phi();
                        PFTau_nProngs  = signalTracks.size();
                        PFTau_ltrackPt = leadingTrack->pt();
			PFTau_d_isol   = theDiscriminator;
			PFTau_d_1      = d_trackIsolation;
		  }
		}

		tauTree->Fill();
        }
	}
}

void TCTauAnalysis::endJob(){}

DEFINE_FWK_MODULE(TCTauAnalysis);
