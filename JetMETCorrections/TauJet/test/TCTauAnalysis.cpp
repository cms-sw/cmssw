// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>

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

using namespace std;
using namespace reco;

class TCTauAnalysis : public edm::EDAnalyzer {
  public:
  	explicit TCTauAnalysis(const edm::ParameterSet&);
  	~TCTauAnalysis();

  	virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  	virtual void beginJob();
  	virtual void endJob();

  private:
	void resetNtuple();

        void fillCaloTau(CaloTauRef);
        void fillTCTau(CaloTauRef);
        void fillPFTau(PFTauRef);

	TFile* outFile;
	TTree* tauTree;

	float MCTau_pt,MCTau_eta,MCTau_phi;
	float PFTau_pt,PFTau_eta,PFTau_phi,PFTau_nProngs,PFTau_ltrackPt,PFTau_d_isol,PFTau_d_1,PFTau_d_2;
	float CaloTau_pt,CaloTau_eta,CaloTau_phi,CaloTau_nProngs,CaloTau_ltrackPt,CaloTau_d_isol,CaloTau_d_1,CaloTau_d_2;
	float JPTTau_pt,JPTTau_eta,JPTTau_phi;
	float TCTau_pt,TCTau_eta,TCTau_phi,TCTau_nProngs,TCTau_ltrackPt,TCTau_d_isol,TCTau_d_1,TCTau_d_2,TCTau_algo;
        float TCTau_pt_raw,TCTau_eta_raw,TCTau_phi_raw;

	int nMCTaus,
            nCaloTaus,
            nTCTaus,
	    nPFTaus;

	double tauEtCut,tauEtaCut;
	bool useMCInfo;

	edm::InputTag CaloTaus;
	edm::InputTag TCTaus;
	edm::InputTag PFTaus;
	edm::InputTag MCTaus;
	edm::InputTag Discriminator;

        double matchingConeSize,
               signalConeSize,
               isolationConeSize,
               ptLeadingTrackMin,
	       ptOtherTracksMin;
        std::string metric;
        unsigned int isolationAnnulus_Tracksmaxn;

        edm::Handle<CaloTauCollection>    theCaloTauHandle;
        edm::Handle<CaloTauDiscriminator> theCaloTauDiscriminatorHandle;

        edm::Handle<CaloTauCollection>    theTCTauHandle;
        edm::Handle<CaloTauDiscriminator> theTCTauDiscriminatorHandle;
	edm::Handle<CaloTauDiscriminator> theTCTauAlgoHandle;

        edm::Handle<PFTauCollection>      thePFTauHandle;
        edm::Handle<PFTauDiscriminator>   thePFTauDiscriminatorHandle;
};

TCTauAnalysis::TCTauAnalysis(const edm::ParameterSet& iConfig){

        tauEtCut        = iConfig.getParameter<double>("TauJetEt");
        tauEtaCut       = iConfig.getParameter<double>("TauJetEta");
        CaloTaus        = iConfig.getParameter<edm::InputTag>("CaloTauCollection");
        TCTaus          = iConfig.getParameter<edm::InputTag>("TCTauCollection");
        PFTaus          = iConfig.getParameter<edm::InputTag>("PFTauCollection");
        MCTaus          = iConfig.getParameter<edm::InputTag>("MCTauCollection");
        Discriminator   = iConfig.getParameter<edm::InputTag>("Discriminator");
        useMCInfo       = iConfig.getParameter<bool>("UseMCInfo");

        matchingConeSize            = 0.1,
        signalConeSize              = 0.07,
        isolationConeSize           = 0.4,
        ptLeadingTrackMin           = 6,
        ptOtherTracksMin            = 1;
        metric                      = "DR"; // can be DR,angle,area
        isolationAnnulus_Tracksmaxn = 0;


	outFile = TFile::Open("tctau.root", "RECREATE");
        tauTree = new TTree("tauTree","tauTree");

	nMCTaus   = 0;
	nCaloTaus = 0;
	nTCTaus   = 0;
	nPFTaus   = 0;

	if(useMCInfo){
        MCTau_pt  = 0; tauTree->Branch("MCTau_pt",  &MCTau_pt,  "MCTau_pt/F");
        MCTau_eta = 0; tauTree->Branch("MCTau_eta", &MCTau_eta, "MCTau_eta/F");
        MCTau_phi = 0; tauTree->Branch("MCTau_phi", &MCTau_phi, "MCTau_phi/F");
	}

        PFTau_pt  = 0;      tauTree->Branch("PFTau_pt",  &PFTau_pt,  "PFTau_pt/F");
        PFTau_eta = 0;      tauTree->Branch("PFTau_eta", &PFTau_eta, "PFTau_eta/F");
        PFTau_phi = 0;      tauTree->Branch("PFTau_phi", &PFTau_phi, "PFTau_phi/F");
	PFTau_nProngs = 0;  tauTree->Branch("PFTau_nProngs",  &PFTau_nProngs,  "PFTau_nProngs/F");
	PFTau_ltrackPt = 0; tauTree->Branch("PFTau_ltrackPt",  &PFTau_ltrackPt,  "PFTau_ltrackPt/F");
	PFTau_d_isol = 0;   tauTree->Branch("PFTau_d_isol",  &PFTau_d_isol,  "PFTau_d_isol/F"); //DiscriminationByIsolation
	PFTau_d_1 = 0;      tauTree->Branch("PFTau_d_1",  &PFTau_d_1,  "PFTau_d_1/F");
	PFTau_d_2 = 0;      tauTree->Branch("PFTau_d_2",  &PFTau_d_2,  "PFTau_d_2/F");

        CaloTau_pt  = 0;      tauTree->Branch("CaloTau_pt",  &CaloTau_pt,  "CaloTau_pt/F");
        CaloTau_eta = 0;      tauTree->Branch("CaloTau_eta", &CaloTau_eta, "CaloTau_eta/F");
        CaloTau_phi = 0;      tauTree->Branch("CaloTau_phi", &CaloTau_phi, "CaloTau_phi/F");
        CaloTau_nProngs = 0;  tauTree->Branch("CaloTau_nProngs",  &CaloTau_nProngs,  "CaloTau_nProngs/F");
        CaloTau_ltrackPt = 0; tauTree->Branch("CaloTau_ltrackPt",  &CaloTau_ltrackPt,  "CaloTau_ltrackPt/F");
	CaloTau_d_isol = 0;   tauTree->Branch("CaloTau_d_isol",  &CaloTau_d_isol,  "CaloTau_d_isol/F");
        CaloTau_d_1 = 0;      tauTree->Branch("CaloTau_d_1",  &CaloTau_d_1,  "CaloTau_d_1/F");
        CaloTau_d_2 = 0;      tauTree->Branch("CaloTau_d_2",  &CaloTau_d_2,  "CaloTau_d_2/F");

        JPTTau_pt  = 0;      tauTree->Branch("JPTTau_pt",  &JPTTau_pt,  "JPTTau_pt/F");
        JPTTau_eta = 0;      tauTree->Branch("JPTTau_eta", &JPTTau_eta, "JPTTau_eta/F");
        JPTTau_phi = 0;      tauTree->Branch("JPTTau_phi", &JPTTau_phi, "JPTTau_phi/F");

        TCTau_pt  = 0;      tauTree->Branch("TCTau_pt",  &TCTau_pt,  "TCTau_pt/F");
        TCTau_eta = 0;      tauTree->Branch("TCTau_eta", &TCTau_eta, "TCTau_eta/F");
        TCTau_phi = 0;      tauTree->Branch("TCTau_phi", &TCTau_phi, "TCTau_phi/F");
        TCTau_nProngs = 0;  tauTree->Branch("TCTau_nProngs",  &TCTau_nProngs,  "TCTau_nProngs/F");
        TCTau_ltrackPt = 0; tauTree->Branch("TCTau_ltrackPt",  &TCTau_ltrackPt,  "TCTau_ltrackPt/F");
        TCTau_d_isol = 0;   tauTree->Branch("TCTau_d_isol",  &TCTau_d_isol,  "TCTau_d_isol/F");
        TCTau_d_1 = 0;      tauTree->Branch("TCTau_d_1",  &TCTau_d_1,  "TCTau_d_1/F");
        TCTau_d_2 = 0;      tauTree->Branch("TCTau_d_2",  &TCTau_d_2,  "TCTau_d_2/F");
	TCTau_algo = 0;     tauTree->Branch("TCTau_algo",  &TCTau_algo,  "TCTau_algo/F");
        TCTau_pt_raw  = 0;  tauTree->Branch("TCTau_pt_raw",  &TCTau_pt_raw,  "TCTau_pt_raw/F");
        TCTau_eta_raw = 0;  tauTree->Branch("TCTau_eta_raw", &TCTau_eta_raw, "TCTau_eta_raw/F");
        TCTau_phi_raw = 0;  tauTree->Branch("TCTau_phi_raw", &TCTau_phi_raw, "TCTau_phi_raw/F");
}

TCTauAnalysis::~TCTauAnalysis(){

        cout << endl << endl;
	if(useMCInfo)
        cout << "MCTaus     " << nMCTaus << endl;
        cout << "CaloTaus   " << nCaloTaus << endl;
        cout << "TCTaus     " << nTCTaus << endl;
        cout << "PFTaus     " << nPFTaus << endl;

	outFile->cd();
        tauTree->Write();
        outFile->Close();

}


void TCTauAnalysis::resetNtuple(){
        MCTau_pt       = 0;
        MCTau_eta      = 0;
        MCTau_phi      = 0;

        PFTau_pt       = 0;
        PFTau_eta      = 0;
        PFTau_phi      = 0;
        PFTau_nProngs  = 0;
        PFTau_ltrackPt = 0;
        PFTau_d_isol   = 0;
        PFTau_d_1      = 0;
        PFTau_d_2      = 0;

        CaloTau_pt       = 0;
        CaloTau_eta      = 0;
        CaloTau_phi      = 0;
        CaloTau_nProngs  = 0;
        CaloTau_ltrackPt = 0;
        CaloTau_d_isol   = 0;
        CaloTau_d_1      = 0;
        CaloTau_d_2      = 0;

        TCTau_pt       = 0;
        TCTau_eta      = 0;
        TCTau_phi      = 0;
        TCTau_nProngs  = 0;
        TCTau_ltrackPt = 0;
	TCTau_d_isol   = 0;
        TCTau_d_1      = 0;
        TCTau_d_2      = 0;
	TCTau_algo     = 0;
        TCTau_pt_raw   = 0;
        TCTau_eta_raw  = 0;
        TCTau_phi_raw  = 0;
}

void TCTauAnalysis::beginJob(){}

void TCTauAnalysis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

	iEvent.getByLabel(CaloTaus,theCaloTauHandle);
        iEvent.getByLabel(edm::InputTag("caloRecoTau"+Discriminator.label(),"",CaloTaus.process()),
                          theCaloTauDiscriminatorHandle);

        iEvent.getByLabel(TCTaus,theTCTauHandle);
        iEvent.getByLabel(edm::InputTag("caloRecoTau"+Discriminator.label()),
                          theTCTauDiscriminatorHandle);
	iEvent.getByLabel("tcRecoTauDiscriminationAlgoComponent",theTCTauAlgoHandle);

	std::string pfTau = PFTaus.label();
	pfTau = pfTau.substr(0,pfTau.find("Producer"));
        iEvent.getByLabel(PFTaus,thePFTauHandle);
	iEvent.getByLabel(pfTau+Discriminator.label(),thePFTauDiscriminatorHandle);


      if(useMCInfo){
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

			double DR = ROOT::Math::VectorUtil::DeltaR(*i, theTau->p4());

			if(DR > 0.5) continue;

			fillCaloTau(theTau);
		  }
		}

// TCTaus
                if(theTCTauHandle.isValid()){

                  for(unsigned int iTau = 0; iTau < theTCTauHandle->size(); ++iTau){
                        CaloTauRef theTau(theTCTauHandle,iTau);

                        double DR = ROOT::Math::VectorUtil::DeltaR(*i, theTau->p4());

                        if(DR > 0.5) continue;

			fillTCTau(theTau);
		  }
		}

// PFTaus
		if(thePFTauHandle.isValid()){

		  for(unsigned int iTau = 0; iTau < thePFTauHandle->size(); ++iTau){
			PFTauRef theTau(thePFTauHandle,iTau);

                        double DR = ROOT::Math::VectorUtil::DeltaR(*i,theTau->p4());

                        if(DR > 0.5) continue;

			fillPFTau(theTau);
		  }
		}

		tauTree->Fill();
	  }
	}

      }else{

          if(theCaloTauHandle.isValid()){
            for(unsigned int iCaloTau = 0; iCaloTau < theCaloTauHandle->size(); ++iCaloTau){
                CaloTauRef caloTau(theCaloTauHandle,iCaloTau);

		if(caloTau->pt() < tauEtCut || fabs(caloTau->eta()) > tauEtaCut) continue;

                fillCaloTau(caloTau);

                if(theTCTauHandle.isValid()){
                  for(unsigned int iTCTau = 0; iTCTau < theTCTauHandle->size(); ++iTCTau){
                        CaloTauRef tcTau(theTCTauHandle,iTCTau);

                        double DR = ROOT::Math::VectorUtil::DeltaR(tcTau->p4(), caloTau->p4());

                        if(DR > 0.5) continue;

                        fillTCTau(tcTau);
                  }
                }
                if(thePFTauHandle.isValid()){
                  for(unsigned int iPFTau = 0; iPFTau < thePFTauHandle->size(); ++iPFTau){
                        PFTauRef pfTau(thePFTauHandle,iPFTau);

                        double DR = ROOT::Math::VectorUtil::DeltaR(pfTau->p4(),caloTau->p4());

                        if(DR > 0.5) continue;

                        fillPFTau(pfTau);
                  }
                }

                tauTree->Fill();
            }
          }
      }
}

void TCTauAnalysis::fillCaloTau(CaloTauRef theTau){

	if(theTau->pt() == 0) return;

        CaloTau theCaloTau = *theTau;
/*
cout << "check TCTauAnalysis::fillCaloTau 1.1"<< endl;
        CaloTauElementsOperators op(theCaloTau);
cout << "check TCTauAnalysis::fillCaloTau 1.2"<< endl;

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

cout << "check TCTauAnalysis::fillCaloTau 2"<< endl;
        const TrackRef leadingTrack =op.leadTk(metric,matchingConeSize,ptLeadingTrackMin);
*/
	double theDiscriminator = (*theCaloTauDiscriminatorHandle)[theTau];

	cout << "CaloTau Et = " << theCaloTau.pt() <<endl;

	nCaloTaus++;

	CaloTau_pt       = theCaloTau.pt();
	CaloTau_eta      = theCaloTau.eta();
	CaloTau_phi      = theCaloTau.phi();
	CaloTau_d_isol   = theDiscriminator;
////	CaloTau_d_1	 = d_trackIsolation;

/*
	if(leadingTrack.isNonnull()) {
		const TrackRefVector signalTracks = op.tracksInCone(leadingTrack->momentum(),metric,signalConeSize,ptOtherTracksMin);
		CaloTau_nProngs  = signalTracks.size();
		CaloTau_ltrackPt = leadingTrack->pt();
	}
*/
}

void TCTauAnalysis::fillTCTau(CaloTauRef theTau){

	if(theTau->pt() == 0) return;

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

        const TrackRef leadingTrack =op.leadTk(metric,matchingConeSize,ptLeadingTrackMin);

        double theDiscriminator = (*theTCTauDiscriminatorHandle)[theTau];

	double theAlgo          = (*theTCTauAlgoHandle)[theTau];

        cout << "CaloTau+JPT+TCTau Et = " << jptTCTauCorrected.pt() <<endl;

        nTCTaus++;

	JPTTau_pt  = jptTCTauCorrected.caloTauTagInfoRef()->jetRef()->pt();
	JPTTau_eta = jptTCTauCorrected.caloTauTagInfoRef()->jetRef()->eta();
	JPTTau_phi = jptTCTauCorrected.caloTauTagInfoRef()->jetRef()->phi();

        TCTau_pt       = jptTCTauCorrected.pt();
        TCTau_eta      = jptTCTauCorrected.eta();
        TCTau_phi      = jptTCTauCorrected.phi();
        TCTau_d_isol   = theDiscriminator;
        TCTau_d_1      = d_trackIsolation;
	TCTau_algo     = theAlgo;

        TCTau_pt_raw   = jptTCTauCorrected.rawJetRef()->et();
	cout << "TCTau Raw Et         = " << TCTau_pt_raw << endl;

        if(leadingTrack.isNonnull()) {
                const TrackRefVector signalTracks = op.tracksInCone(leadingTrack->momentum(),metric,signalConeSize,ptOtherTracksMin);
                TCTau_nProngs  = signalTracks.size();
                TCTau_ltrackPt = leadingTrack->pt();
        }
}

void TCTauAnalysis::fillPFTau(PFTauRef theTau){

	if(theTau->pt() == 0) return;

        PFTau thePFTau = *theTau;
        PFTauElementsOperators op(thePFTau);
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

        const TrackRef leadingTrack =op.leadTk(metric,matchingConeSize,ptLeadingTrackMin);

	double theDiscriminator = (*thePFTauDiscriminatorHandle)[theTau];

        cout << "PFTau Et = " << thePFTau.pt() <<endl;

        nPFTaus++;

        PFTau_pt       = thePFTau.pt();
        PFTau_eta      = thePFTau.eta();
        PFTau_phi      = thePFTau.phi();
        PFTau_d_isol   = theDiscriminator;
        PFTau_d_1      = d_trackIsolation;

        if(leadingTrack.isNonnull()) {
                const TrackRefVector signalTracks = op.tracksInCone(leadingTrack->momentum(),metric,signalConeSize,ptOtherTracksMin);
                PFTau_nProngs  = signalTracks.size();
                PFTau_ltrackPt = leadingTrack->pt();
        }
}

void TCTauAnalysis::endJob(){}

DEFINE_FWK_MODULE(TCTauAnalysis);
