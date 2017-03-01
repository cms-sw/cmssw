// -*- C++ -*-
//
// Package:    RecoTauTag/RecoTau/rerunMVAIsolationOnMiniAOD
// Class:      rerunMVAIsolationOnMiniAOD
// 
/**\class rerunMVAIsolationOnMiniAOD rerunMVAIsolationOnMiniAOD.cc RecoTauTag/RecoTau/test/rerunMVAIsolationOnMiniAOD.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Alexander Nehrkorn
//         Created:  Wed, 30 Mar 2016 13:39:51 GMT
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

#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/PATTauDiscriminator.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"

#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA6.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

typedef edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> > PFTauTIPAssociationByRef;

class rerunMVAIsolationOnMiniAOD : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit rerunMVAIsolationOnMiniAOD(const edm::ParameterSet&);
      ~rerunMVAIsolationOnMiniAOD();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------
      TFile* outfile;

      bool verbosity_;
      bool additionalCollectionsAvailable_;
      TauIdMVAAuxiliaries clusterVariables_;

      edm::EDGetTokenT<pat::TauCollection> tauToken_;
      edm::EDGetTokenT<pat::PATTauDiscriminator> mvaIsolationToken_;
      edm::EDGetTokenT<pat::PATTauDiscriminator> mvaIsolationVLooseToken_;
      edm::EDGetTokenT<pat::PATTauDiscriminator> mvaIsolationLooseToken_;
      edm::EDGetTokenT<pat::PATTauDiscriminator> mvaIsolationMediumToken_;
      edm::EDGetTokenT<pat::PATTauDiscriminator> mvaIsolationTightToken_;
      edm::EDGetTokenT<pat::PATTauDiscriminator> mvaIsolationVTightToken_;
      edm::EDGetTokenT<pat::PATTauDiscriminator> mvaIsolationVVTightToken_;
      edm::EDGetTokenT<pat::PATTauDiscriminator> mvaEleRawToken_;
      edm::EDGetTokenT<reco::PFTauCollection> pfTauToken_;
      edm::EDGetTokenT<reco::PFTauDiscriminator> dmfNewToken_;
      edm::EDGetTokenT<reco::PFTauDiscriminator> chargedIsoPtSumToken_;
      edm::EDGetTokenT<reco::PFTauDiscriminator> neutralIsoPtSumToken_;
      edm::EDGetTokenT<reco::PFTauDiscriminator> puCorrPtSumToken_;
      edm::EDGetTokenT<reco::PFTauDiscriminator> photonPtSumOutsideSignalConeToken_;
      edm::EDGetTokenT<reco::PFTauDiscriminator> footprintCorrectionToken_;
      edm::EDGetTokenT<reco::PFTauDiscriminator> rawElecMVA6Token_;
      edm::EDGetTokenT<PFTauTIPAssociationByRef> tauTIPToken_;

      TH1D* mvaValueAOD;
      TH1D* mvaValueMiniAOD;
      TH1D* mvaValueDiff;

      TH1D* differences;
      TH1D* differencesWeighted;

      TH1D* difference_dxy;
      TH1D* difference_dxySig;
      TH1D* difference_ip3d;
      TH1D* difference_ip3dSig;
      TH1D* difference_flightlengthSig;
      TH1D* difference_ptWeightedDetaStrip;
      TH1D* difference_ptWeightedDphiStrip;
      TH1D* difference_ptWeightedDrSignal;
      TH1D* difference_ptWeightedDrIso;

      TH2D* mvaValue;
      TH2D* mvaValue_vLoose;
      TH2D* mvaValue_Loose;
      TH2D* mvaValue_Medium;
      TH2D* mvaValue_Tight;
      TH2D* mvaValue_vTight;
      TH2D* mvaValue_vvTight;

      TH2D* decayMode;
      TH2D* chargedIsoPtSum;
      TH2D* neutralIsoPtSum;
      TH2D* puCorrPtSum;
      TH2D* photonPtSumOutsideSignalCone;
      TH2D* footprintCorrection;

      TH2D* decayDistMag;
      TH2D* dxy;
      TH2D* dxySig;
      TH2D* ip3d;
      TH2D* ip3dSig;
      TH2D* hasSV;
      TH2D* flightlengthSig;
      TH2D* nPhoton;
      TH2D* ptWeightedDetaStrip;
      TH2D* ptWeightedDphiStrip;
      TH2D* ptWeightedDrSignal;
      TH2D* ptWeightedDrIsolation;
      TH2D* leadTrackChi2;
      TH2D* eRatio;
      TH2D* mvaValue_antiEMVA6;
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
rerunMVAIsolationOnMiniAOD::rerunMVAIsolationOnMiniAOD(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
   usesResource("TFileService");

   outfile = new TFile("outfile_rerunMVAIsolationOnMiniAOD.root","RECREATE");

   tauToken_ = consumes<pat::TauCollection>(edm::InputTag("slimmedTaus","","PAT"));
   mvaIsolationToken_ = consumes<pat::PATTauDiscriminator>(edm::InputTag("rerunDiscriminationByIsolationMVArun2v1raw","","rerunMVAIsolationOnMiniAOD"));
   mvaIsolationVLooseToken_ = consumes<pat::PATTauDiscriminator>(edm::InputTag("rerunDiscriminationByIsolationMVArun2v1VLoose","","rerunMVAIsolationOnMiniAOD"));
   mvaIsolationLooseToken_ = consumes<pat::PATTauDiscriminator>(edm::InputTag("rerunDiscriminationByIsolationMVArun2v1Loose","","rerunMVAIsolationOnMiniAOD"));
   mvaIsolationMediumToken_ = consumes<pat::PATTauDiscriminator>(edm::InputTag("rerunDiscriminationByIsolationMVArun2v1Medium","","rerunMVAIsolationOnMiniAOD"));
   mvaIsolationTightToken_ = consumes<pat::PATTauDiscriminator>(edm::InputTag("rerunDiscriminationByIsolationMVArun2v1Tight","","rerunMVAIsolationOnMiniAOD"));
   mvaIsolationVTightToken_ = consumes<pat::PATTauDiscriminator>(edm::InputTag("rerunDiscriminationByIsolationMVArun2v1VTight","","rerunMVAIsolationOnMiniAOD"));
   mvaIsolationVVTightToken_ = consumes<pat::PATTauDiscriminator>(edm::InputTag("rerunDiscriminationByIsolationMVArun2v1VVTight","","rerunMVAIsolationOnMiniAOD"));
   mvaEleRawToken_ = consumes<pat::PATTauDiscriminator>(edm::InputTag("rerunDiscriminationAgainstElectronMVA6","","rerunMVAIsolationOnMiniAOD"));
   pfTauToken_ = consumes<reco::PFTauCollection>(edm::InputTag("hpsPFTauProducer","","PAT"));
   dmfNewToken_ = consumes<reco::PFTauDiscriminator>(edm::InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs","","PAT"));
   chargedIsoPtSumToken_ = consumes<reco::PFTauDiscriminator>(edm::InputTag("hpsPFTauChargedIsoPtSum","","PAT"));
   neutralIsoPtSumToken_ = consumes<reco::PFTauDiscriminator>(edm::InputTag("hpsPFTauNeutralIsoPtSum","","PAT"));
   puCorrPtSumToken_ = consumes<reco::PFTauDiscriminator>(edm::InputTag("hpsPFTauPUcorrPtSum","","PAT"));
   photonPtSumOutsideSignalConeToken_ = consumes<reco::PFTauDiscriminator>(edm::InputTag("hpsPFTauPhotonPtSumOutsideSignalCone","","PAT"));
   footprintCorrectionToken_ = consumes<reco::PFTauDiscriminator>(edm::InputTag("hpsPFTauFootprintCorrection","","PAT"));
   rawElecMVA6Token_    = consumes<reco::PFTauDiscriminator>(edm::InputTag("hpsPFTauDiscriminationagainstElectronMVA6Raw","","RECO"));
   tauTIPToken_ = consumes<PFTauTIPAssociationByRef>(edm::InputTag("hpsPFTauTransverseImpactParameters","","PAT"));

   verbosity_ = iConfig.getParameter<int>("verbosity");
   additionalCollectionsAvailable_ = iConfig.getParameter<bool>("additionalCollectionsAvailable");

   mvaValueAOD = new TH1D("mvaValueAOD",";MVA value;",220,-1.1,1.1);
   mvaValueMiniAOD = new TH1D("mvaValueMiniAOD",";MVA value;",220,-1.1,1.1);
   mvaValueDiff = new TH1D("mvaValueDiff",";|AOD - MiniAOD|;",2000,0,2);

   differences = new TH1D("differences","",24,-0.5,23.5);
   differencesWeighted = new TH1D("differencesWeighted","",24,-0.5,23.5);

   difference_dxy = new TH1D("difference_dxy",";|AOD - MiniAOD| (dxy);",1000,0,0.0005);
   difference_dxySig = new TH1D("difference_dxySig",";|AOD - MiniAOD| (dxySig);",1000,0,0.0005);
   difference_ip3d = new TH1D("difference_ip3d",";|AOD - MiniAOD| (ip3d);",1000,0,0.0005);
   difference_ip3dSig = new TH1D("difference_ip3dSig",";|AOD - MiniAOD| (ip3dSig);",1000,0,0.0005);
   difference_flightlengthSig = new TH1D("difference_flightlengthSig",";|AOD - MiniAOD| (flightlengthSig);",1000,0,0.0005);
   difference_ptWeightedDetaStrip = new TH1D("difference_ptWeightedDetaStrip",";|AOD - MiniAOD| (ptWeightedDetaStrip);",1000,0,0.0005);
   difference_ptWeightedDphiStrip = new TH1D("difference_ptWeightedDphiStrip",";|AOD - MiniAOD| (ptWeightedDphiStrip);",1000,0,0.0005);
   difference_ptWeightedDrSignal = new TH1D("difference_ptWeightedDrSignal",";|AOD - MiniAOD| (ptWeightedDrSignal);",1000,0,0.0005);
   difference_ptWeightedDrIso = new TH1D("difference_ptWeightedDrIso",";|AOD - MiniAOD| (ptWeightedDrIso);",1000,0,0.0005);

   mvaValue = new TH2D("mvaValue",";AOD;MiniAOD",220,-1.1,1.1,220,-1.1,1.1);
   mvaValue_vLoose = new TH2D("mvaValue_vLoose",";AOD;MiniAOD",2,-0.5,1.5,2,-0.5,1.5);
   mvaValue_Loose = new TH2D("mvaValue_Loose",";AOD;MiniAOD",2,-0.5,1.5,2,-0.5,1.5);
   mvaValue_Medium = new TH2D("mvaValue_Medium",";AOD;MiniAOD",2,-0.5,1.5,2,-0.5,1.5);
   mvaValue_Tight = new TH2D("mvaValue_Tight",";AOD;MiniAOD",2,-0.5,1.5,2,-0.5,1.5);
   mvaValue_vTight = new TH2D("mvaValue_vTight",";AOD;MiniAOD",2,-0.5,1.5,2,-0.5,1.5);
   mvaValue_vvTight = new TH2D("mvaValue_vvTight",";AOD;MiniAOD",2,-0.5,1.5,2,-0.5,1.5);

   decayMode = new TH2D("decayMode",";decay mode (AOD);decay mode (MiniAOD)",12,-0.5,11.5,12,-0.5,11.5);
   chargedIsoPtSum = new TH2D("chargedIsoPtSum",";chargedIsoPtSum (AOD);chargedIsoPtSum (MiniAOD)",500,0,50,500,0,50);
   neutralIsoPtSum = new TH2D("neutralIsoPtSum",";neutralIsoPtSum (AOD);neutralIsoPtSum (MiniAOD)",500,0,50,500,0,50);
   puCorrPtSum = new TH2D("puCorrPtSum",";puCorrPtSum (AOD);puCorrPtSum (MiniAOD)",500,0,50,500,0,50);
   photonPtSumOutsideSignalCone = new TH2D("photonPtSumOutsideSignalCone",";photonPtSumOutsideSignalCone (AOD);photonPtSumOutsideSignalCone (MiniAOD)",500,0,50,500,0,50);
   footprintCorrection = new TH2D("footprintCorrection",";footprintCorrection (AOD);footprintCorrection (MiniAOD)",500,0,50,500,0,50);

   decayDistMag = new TH2D("decayDistMag",";decayDistMag (AOD);decayDistMag (MiniAOD)",100,0,10,100,0,10);
   dxy = new TH2D("dxy",";d_{xy} (AOD);d_{xy} (MiniAOD)",100,0,0.1,100,0,0.1);
   dxySig = new TH2D("dxySig",";d_{xy} significance (AOD);d_{xy} significance (MiniAOD)",10,-0.5,9.5,10,-0.5,9.5);
   ip3d = new TH2D("ip3d",";ip3d (AOD);ip3d (MiniAOD)",100,0,10,100,0,10);
   ip3dSig = new TH2D("ip3dSig",";ip3d significance (AOD);ip3d significance (MiniAOD)",10,-0.5,9.5,10,-0.5,9.5);
   hasSV = new TH2D("hasSV",";has SV (AOD);has SV (MiniAOD)",2,-0.5,1.5,2,-0.5,1.5);
   flightlengthSig = new TH2D("flightlengthSig",";flightlength significance (AOD);flightlength significance (MiniAOD)",21,-10.5,10.5,21,-10.5,10.5);
   nPhoton = new TH2D("nPhoton",";nPhoton (AOD);nPhoton (MiniAOD)",20,-0.5,19.5,20,-0.5,19.5);
   ptWeightedDetaStrip = new TH2D("ptWeightedDetaStrip",";ptWeightedDetaStrip (AOD);ptWeightedDetaStrip (MiniAOD)",50,0,0.5,50,0,0.5);
   ptWeightedDphiStrip = new TH2D("ptWeightedDphiStrip",";ptWeightedDphiStrip (AOD);ptWeightedDphiStrip (MiniAOD)",50,0,0.5,50,0,0.5);
   ptWeightedDrSignal = new TH2D("ptWeightedDrSignal",";ptWeightedDrSignal (AOD);ptWeightedDrSignal (MiniAOD)",50,0,0.5,50,0,0.5);
   ptWeightedDrIsolation = new TH2D("ptWeightedDrIsolation",";ptWeightedDrIsolation (AOD);ptWeightedDrIsolation (MiniAOD)",50,0,0.5,50,0,0.5);
   leadTrackChi2 = new TH2D("leadTrackChi2",";leadTrackChi2 (AOD);leadTrackChi2 (MiniAOD)",1000,0,100,1000,0,100);
   eRatio = new TH2D("eRatio",";eRatio (AOD);eRatio (MiniAOD)",200,0,2,200,0,2);
   mvaValue_antiEMVA6 = new TH2D("mvaValue_antiEMVA6",";AOD;MiniAOD",220,-1.1,1.1,220,-1.1,1.1);
}


rerunMVAIsolationOnMiniAOD::~rerunMVAIsolationOnMiniAOD()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

	outfile->Close();
}


//
// member functions
//

// ------------ method called for each event  ------------
void
rerunMVAIsolationOnMiniAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	edm::Handle<pat::TauCollection> taus;
	iEvent.getByToken(tauToken_,taus);

	edm::Handle<pat::PATTauDiscriminator> mvaIsoRaw;
	iEvent.getByToken(mvaIsolationToken_,mvaIsoRaw);

	edm::Handle<pat::PATTauDiscriminator> mvaIsoVLoose;
	iEvent.getByToken(mvaIsolationVLooseToken_,mvaIsoVLoose);

	edm::Handle<pat::PATTauDiscriminator> mvaIsoLoose;
	iEvent.getByToken(mvaIsolationLooseToken_,mvaIsoLoose);

	edm::Handle<pat::PATTauDiscriminator> mvaIsoMedium;
	iEvent.getByToken(mvaIsolationMediumToken_,mvaIsoMedium);

	edm::Handle<pat::PATTauDiscriminator> mvaIsoTight;
	iEvent.getByToken(mvaIsolationTightToken_,mvaIsoTight);

	edm::Handle<pat::PATTauDiscriminator> mvaIsoVTight;
	iEvent.getByToken(mvaIsolationVTightToken_,mvaIsoVTight);

	edm::Handle<pat::PATTauDiscriminator> mvaIsoVVTight;
	iEvent.getByToken(mvaIsolationVVTightToken_,mvaIsoVVTight);

        edm::Handle<reco::PFTauDiscriminator> rawElecMVA6;
	iEvent.getByToken(rawElecMVA6Token_,rawElecMVA6);
       
        edm::Handle<pat::PATTauDiscriminator> mvaEleRaw;
        iEvent.getByToken(mvaEleRawToken_,mvaEleRaw);

	std::vector<pat::TauRef> unmatchedTaus;

	for(unsigned iTau = 0; iTau < taus->size(); iTau++)
	{
		pat::TauRef tau(taus,iTau);
		float valueAOD = tau->tauID("byIsolationMVArun2v1DBoldDMwLTraw");
		float valueMiniAOD = (*mvaIsoRaw)[tau];

		mvaValueAOD->Fill(valueAOD);
		mvaValueMiniAOD->Fill(valueMiniAOD);

		mvaValue->Fill(valueAOD,valueMiniAOD);
		mvaValue_vLoose->Fill(tau->tauID("byVLooseIsolationMVArun2v1DBoldDMwLT"),(*mvaIsoVLoose)[tau]);
		mvaValue_Loose->Fill(tau->tauID("byLooseIsolationMVArun2v1DBoldDMwLT"),(*mvaIsoLoose)[tau]);
		mvaValue_Medium->Fill(tau->tauID("byMediumIsolationMVArun2v1DBoldDMwLT"),(*mvaIsoMedium)[tau]);
		mvaValue_Tight->Fill(tau->tauID("byTightIsolationMVArun2v1DBoldDMwLT"),(*mvaIsoTight)[tau]);
		mvaValue_vTight->Fill(tau->tauID("byVTightIsolationMVArun2v1DBoldDMwLT"),(*mvaIsoVTight)[tau]);
		mvaValue_vvTight->Fill(tau->tauID("byVVTightIsolationMVArun2v1DBoldDMwLT"),(*mvaIsoVVTight)[tau]);
		mvaValueDiff->Fill(std::abs(valueAOD - valueMiniAOD));
                mvaValue_antiEMVA6->Fill((*mvaEleRaw)[tau] , taus->at(iTau).tauID("againstElectronMVA6Raw"));

		if(valueAOD != valueMiniAOD)
			unmatchedTaus.push_back(tau);
	}

	// for the following code, four additional collections are needed:
	// - PFTaus
	// - PFTauTIPAssociationByRef (impact parameter info)
	// - PFCandidates
	// - Tracks

	if(additionalCollectionsAvailable_)
	{
		edm::Handle<reco::PFTauCollection> pfTaus;
		iEvent.getByToken(pfTauToken_,pfTaus);

		edm::Handle<reco::PFTauDiscriminator> dmfNew;
		iEvent.getByToken(dmfNewToken_,dmfNew);

		edm::Handle<reco::PFTauDiscriminator> chargedIso;
		iEvent.getByToken(chargedIsoPtSumToken_,chargedIso);

		edm::Handle<reco::PFTauDiscriminator> neutralIso;
		iEvent.getByToken(neutralIsoPtSumToken_,neutralIso);

		edm::Handle<reco::PFTauDiscriminator> puCorr;
		iEvent.getByToken(puCorrPtSumToken_,puCorr);

		edm::Handle<reco::PFTauDiscriminator> photonSumOutsideSignalCone;
		iEvent.getByToken(photonPtSumOutsideSignalConeToken_,photonSumOutsideSignalCone);

		edm::Handle<reco::PFTauDiscriminator> footPrint;
		iEvent.getByToken(footprintCorrectionToken_,footPrint);

		edm::Handle<PFTauTIPAssociationByRef> tauLifetimeInfos;
		iEvent.getByToken(tauTIPToken_,tauLifetimeInfos);

		for(unsigned iPFTau = 0; iPFTau < pfTaus->size(); iPFTau++)
		{
			reco::PFTauRef pfTau(pfTaus,iPFTau);

			if((*dmfNew)[pfTau] < 0.5) continue;

			if((float)pfTau->pt() < 18 || std::abs((float)pfTau->eta()) > 2.3) continue;

			for(unsigned iTau = 0; iTau < unmatchedTaus.size(); iTau++)
			{
				if((float)pfTau->pt() != (float)unmatchedTaus.at(iTau)->pt()) continue;
				if((float)pfTau->eta() != (float)unmatchedTaus.at(iTau)->eta()) continue;
				if((float)pfTau->phi() != (float)unmatchedTaus.at(iTau)->phi()) continue;
				if((float)pfTau->energy() != (float)unmatchedTaus.at(iTau)->energy()) continue;

				decayMode->Fill(pfTau->decayMode(),unmatchedTaus.at(iTau)->decayMode());
				chargedIsoPtSum->Fill((*chargedIso)[pfTau],unmatchedTaus.at(iTau)->tauID("chargedIsoPtSum"));
				neutralIsoPtSum->Fill((*neutralIso)[pfTau],unmatchedTaus.at(iTau)->tauID("neutralIsoPtSum"));
				puCorrPtSum->Fill((*puCorr)[pfTau],unmatchedTaus.at(iTau)->tauID("puCorrPtSum"));
				photonPtSumOutsideSignalCone->Fill((*photonSumOutsideSignalCone)[pfTau],unmatchedTaus.at(iTau)->tauID("photonPtSumOutsideSignalCone"));
				footprintCorrection->Fill((*footPrint)[pfTau],unmatchedTaus.at(iTau)->tauID("footprintCorrection"));

				const reco::PFTauTransverseImpactParameter& tauLifetimeInfo = *(*tauLifetimeInfos)[pfTau];

				float decayDistXAOD = tauLifetimeInfo.flightLength().x();
				float decayDistYAOD = tauLifetimeInfo.flightLength().y();
				float decayDistZAOD = tauLifetimeInfo.flightLength().z();
				float decayDistMagAOD = std::sqrt(decayDistXAOD*decayDistXAOD + decayDistYAOD*decayDistYAOD + decayDistZAOD*decayDistZAOD);

				float decayDistXMiniAOD = unmatchedTaus.at(iTau)->flightLength().x();
				float decayDistYMiniAOD = unmatchedTaus.at(iTau)->flightLength().y();
				float decayDistZMiniAOD = unmatchedTaus.at(iTau)->flightLength().z();
				float decayDistMagMiniAOD = std::sqrt(decayDistXMiniAOD*decayDistXMiniAOD + decayDistYMiniAOD*decayDistYMiniAOD + decayDistZMiniAOD*decayDistZMiniAOD);

				decayDistMag->Fill(decayDistMagAOD,decayDistMagMiniAOD);
				dxy->Fill(tauLifetimeInfo.dxy(),unmatchedTaus.at(iTau)->dxy());
				dxySig->Fill(tauLifetimeInfo.dxy_Sig(),unmatchedTaus.at(iTau)->dxy_Sig());
				ip3d->Fill(tauLifetimeInfo.ip3d(),unmatchedTaus.at(iTau)->ip3d());
				ip3dSig->Fill(tauLifetimeInfo.ip3d_Sig(),unmatchedTaus.at(iTau)->ip3d_Sig());
				hasSV->Fill(tauLifetimeInfo.hasSecondaryVertex(),unmatchedTaus.at(iTau)->hasSecondaryVertex());
				flightlengthSig->Fill(tauLifetimeInfo.flightLengthSig(),unmatchedTaus.at(iTau)->flightLengthSig());
				nPhoton->Fill((float)clusterVariables_.tau_n_photons_total(*pfTau),(float)clusterVariables_.tau_n_photons_total(*unmatchedTaus.at(iTau)));
				ptWeightedDetaStrip->Fill(clusterVariables_.tau_pt_weighted_deta_strip(*pfTau, pfTau->decayMode()),clusterVariables_.tau_pt_weighted_deta_strip(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode()));
				ptWeightedDphiStrip->Fill(clusterVariables_.tau_pt_weighted_dphi_strip(*pfTau, pfTau->decayMode()),clusterVariables_.tau_pt_weighted_dphi_strip(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode()));
				ptWeightedDrSignal->Fill(clusterVariables_.tau_pt_weighted_dr_signal(*pfTau, pfTau->decayMode()),clusterVariables_.tau_pt_weighted_dr_signal(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode()));
				ptWeightedDrIsolation->Fill(clusterVariables_.tau_pt_weighted_dr_iso(*pfTau, pfTau->decayMode()),clusterVariables_.tau_pt_weighted_dr_iso(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode()));
				leadTrackChi2->Fill(clusterVariables_.tau_leadTrackChi2(*pfTau),unmatchedTaus.at(iTau)->leadingTrackNormChi2());
				eRatio->Fill(clusterVariables_.tau_Eratio(*pfTau),clusterVariables_.tau_Eratio(*unmatchedTaus.at(iTau)));

				if(verbosity_) std::cout << "=============================================================" << std::endl;
				if(pfTau->pt() != unmatchedTaus.at(iTau)->pt()){
					if(verbosity_) std::cout << "pt: PF = " << pfTau->pt() << ", pat = " << unmatchedTaus.at(iTau)->pt() << std::endl;
					differences->Fill(0);
					differencesWeighted->Fill(0.,std::abs(pfTau->pt()-unmatchedTaus.at(iTau)->pt()));
				}
				if(pfTau->eta() != unmatchedTaus.at(iTau)->eta()){
					if(verbosity_) std::cout << "eta: PF = " << pfTau->eta() << ", pat = " << unmatchedTaus.at(iTau)->eta() << std::endl;
					differences->Fill(1);
					differencesWeighted->Fill(1,std::abs(pfTau->eta()-unmatchedTaus.at(iTau)->eta()));
				}
				if(pfTau->phi() != unmatchedTaus.at(iTau)->phi()){
					if(verbosity_) std::cout << "phi: PF = " << pfTau->phi() << ", pat = " << unmatchedTaus.at(iTau)->phi() << std::endl;
					differences->Fill(2);
					differencesWeighted->Fill(2,std::abs(pfTau->phi()-unmatchedTaus.at(iTau)->phi()));
				}
				if(pfTau->energy() != unmatchedTaus.at(iTau)->energy()){
					if(verbosity_) std::cout << "energy PF = " << pfTau->energy() << ", pat = " << unmatchedTaus.at(iTau)->energy() << std::endl;
					differences->Fill(3);
					differencesWeighted->Fill(3,std::abs(pfTau->energy()-unmatchedTaus.at(iTau)->energy()));
				}
				if(pfTau->decayMode() != unmatchedTaus.at(iTau)->decayMode()){
					if(verbosity_) std::cout << "decayMode: PF = " << pfTau->decayMode() << ", pat = " << unmatchedTaus.at(iTau)->decayMode() << std::endl;
					differences->Fill(4);
					differencesWeighted->Fill(4,std::abs(pfTau->decayMode()-unmatchedTaus.at(iTau)->decayMode()));
				}
				if((*chargedIso)[pfTau] != unmatchedTaus.at(iTau)->tauID("chargedIsoPtSum")){
					if(verbosity_) std::cout << "chargedIso: PF = " << (*chargedIso)[pfTau] << ", pat = " << unmatchedTaus.at(iTau)->tauID("chargedIsoPtSum") << std::endl;
					differences->Fill(5);
					differencesWeighted->Fill(5,std::abs((*chargedIso)[pfTau]-unmatchedTaus.at(iTau)->tauID("chargedIsoPtSum")));
				}
				if((*neutralIso)[pfTau] != unmatchedTaus.at(iTau)->tauID("neutralIsoPtSum")){
					if(verbosity_) std::cout << "neutralIso: PF = " << (*neutralIso)[pfTau] << ", pat = " << unmatchedTaus.at(iTau)->tauID("neutralIsoPtSum") << std::endl;
					differences->Fill(6);
					differencesWeighted->Fill(6,std::abs((*neutralIso)[pfTau]-unmatchedTaus.at(iTau)->tauID("neutralIsoPtSum")));
				}
				if((*puCorr)[pfTau] != unmatchedTaus.at(iTau)->tauID("puCorrPtSum")){
					if(verbosity_) std::cout << "puCorr: PF = " << (*puCorr)[pfTau] << ", pat = " << unmatchedTaus.at(iTau)->tauID("puCorrPtSum") << std::endl;
					differences->Fill(7);
					differencesWeighted->Fill(7,std::abs((*puCorr)[pfTau]-unmatchedTaus.at(iTau)->tauID("puCorrPtSum")));
				}
				if((*photonSumOutsideSignalCone)[pfTau] != unmatchedTaus.at(iTau)->tauID("photonPtSumOutsideSignalCone")){
					if(verbosity_) std::cout << "photonSumOutsideSignalCone: PF = " << (*photonSumOutsideSignalCone)[pfTau] << ", pat = " << unmatchedTaus.at(iTau)->tauID("photonPtSumOutsideSignalCone") << std::endl;
					differences->Fill(8);
					differencesWeighted->Fill(8,std::abs((*photonSumOutsideSignalCone)[pfTau]-unmatchedTaus.at(iTau)->tauID("photonPtSumOutsideSignalCone")));
				}
				if((*footPrint)[pfTau] != unmatchedTaus.at(iTau)->tauID("footprintCorrection")){
					if(verbosity_) std::cout << "footPrint: PF = " << (*footPrint)[pfTau] << ", pat = " << unmatchedTaus.at(iTau)->tauID("footprintCorrection") << std::endl;
					differences->Fill(9);
					differencesWeighted->Fill(9,std::abs((*footPrint)[pfTau]-unmatchedTaus.at(iTau)->tauID("footprintCorrection")));
				}
				if(decayDistMagAOD != decayDistMagMiniAOD){
					if(verbosity_) std::cout << "decayDistMag: PF = " << decayDistMagAOD << ", pat = " << decayDistMagMiniAOD << std::endl;
					differences->Fill(10);
					differencesWeighted->Fill(10,std::abs(decayDistMagAOD-decayDistMagMiniAOD));
				}
				if(tauLifetimeInfo.dxy() != unmatchedTaus.at(iTau)->dxy()){
					if(verbosity_) std::cout << "dxy: PF = " << tauLifetimeInfo.dxy() << ", pat = " << unmatchedTaus.at(iTau)->dxy() << std::endl;
					differences->Fill(11);
					differencesWeighted->Fill(11,std::abs((float)tauLifetimeInfo.dxy()-unmatchedTaus.at(iTau)->dxy()));
					difference_dxy->Fill(std::abs(tauLifetimeInfo.dxy()-unmatchedTaus.at(iTau)->dxy()));
				}
				if(tauLifetimeInfo.dxy_Sig() != unmatchedTaus.at(iTau)->dxy_Sig()){
					if(verbosity_) std::cout << "dxy_Sig: PF = " << tauLifetimeInfo.dxy_Sig() << ", pat = " << unmatchedTaus.at(iTau)->dxy_Sig() << std::endl;
					differences->Fill(12);
					differencesWeighted->Fill(12,std::abs((float)tauLifetimeInfo.dxy_Sig()-unmatchedTaus.at(iTau)->dxy_Sig()));
					difference_dxySig->Fill(std::abs(tauLifetimeInfo.dxy_Sig()-unmatchedTaus.at(iTau)->dxy_Sig()));
				}
				if(tauLifetimeInfo.ip3d() != unmatchedTaus.at(iTau)->ip3d()){
					if(verbosity_) std::cout << "ip3d PF: = " << tauLifetimeInfo.ip3d() << ", pat = " << unmatchedTaus.at(iTau)->ip3d() << std::endl;
					differences->Fill(13);
					differencesWeighted->Fill(13,std::abs((float)tauLifetimeInfo.ip3d()-unmatchedTaus.at(iTau)->ip3d()));
					difference_ip3d->Fill(std::abs(tauLifetimeInfo.ip3d()-unmatchedTaus.at(iTau)->ip3d()));
				}
				if(tauLifetimeInfo.ip3d_Sig() != unmatchedTaus.at(iTau)->ip3d_Sig()){
					if(verbosity_) std::cout << "ip3d_Sig: PF = " << tauLifetimeInfo.ip3d_Sig() << ", pat = " << unmatchedTaus.at(iTau)->ip3d_Sig() << std::endl;
					differences->Fill(14);
					differencesWeighted->Fill(14,std::abs((float)tauLifetimeInfo.ip3d_Sig()-unmatchedTaus.at(iTau)->ip3d_Sig()));
					difference_ip3dSig->Fill(std::abs(tauLifetimeInfo.ip3d_Sig()-unmatchedTaus.at(iTau)->ip3d_Sig()));
				}
				if(tauLifetimeInfo.hasSecondaryVertex() != unmatchedTaus.at(iTau)->hasSecondaryVertex()){
					if(verbosity_) std::cout << "hasSV: PF = " << tauLifetimeInfo.hasSecondaryVertex() << ", pat = " << unmatchedTaus.at(iTau)->hasSecondaryVertex() << std::endl;
					differences->Fill(15);
					differencesWeighted->Fill(15,std::abs((float)tauLifetimeInfo.hasSecondaryVertex()-unmatchedTaus.at(iTau)->hasSecondaryVertex()));
				}
				if(tauLifetimeInfo.flightLengthSig() != unmatchedTaus.at(iTau)->flightLengthSig()){
					if(verbosity_) std::cout << "flightlengthSig: PF = " << tauLifetimeInfo.flightLengthSig() << ", pat = " << unmatchedTaus.at(iTau)->flightLengthSig() << std::endl;
					differences->Fill(16);
					differencesWeighted->Fill(16,std::abs((float)tauLifetimeInfo.flightLengthSig()-unmatchedTaus.at(iTau)->flightLengthSig()));
					difference_flightlengthSig->Fill(std::abs(tauLifetimeInfo.flightLengthSig()-unmatchedTaus.at(iTau)->flightLengthSig()));
				}
				if((float)clusterVariables_.tau_n_photons_total(*pfTau) != (float)clusterVariables_.tau_n_photons_total(*unmatchedTaus.at(iTau))){
					if(verbosity_) std::cout << "nPhoton PF: = " << (float)clusterVariables_.tau_n_photons_total(*pfTau) << ", pat = " << (float)clusterVariables_.tau_n_photons_total(*unmatchedTaus.at(iTau)) << std::endl;
					differences->Fill(17);
					differencesWeighted->Fill(17,std::abs((float)clusterVariables_.tau_n_photons_total(*pfTau)-(float)clusterVariables_.tau_n_photons_total(*unmatchedTaus.at(iTau))));
				}
				if(clusterVariables_.tau_pt_weighted_deta_strip(*pfTau, pfTau->decayMode()) != clusterVariables_.tau_pt_weighted_deta_strip(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode())){
					if(verbosity_) std::cout << "ptWeightedDetaStrip: PF = " << clusterVariables_.tau_pt_weighted_deta_strip(*pfTau, pfTau->decayMode()) << ", pat = " << clusterVariables_.tau_pt_weighted_deta_strip(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode()) << std::endl;
					differences->Fill(18);
					differencesWeighted->Fill(18,std::abs(clusterVariables_.tau_pt_weighted_deta_strip(*pfTau, pfTau->decayMode())-clusterVariables_.tau_pt_weighted_deta_strip(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode())));
					difference_ptWeightedDetaStrip->Fill(std::abs(clusterVariables_.tau_pt_weighted_deta_strip(*pfTau, pfTau->decayMode())-clusterVariables_.tau_pt_weighted_deta_strip(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode())));
				}
				if(clusterVariables_.tau_pt_weighted_dphi_strip(*pfTau, pfTau->decayMode()) != clusterVariables_.tau_pt_weighted_dphi_strip(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode())){
					if(verbosity_) std::cout << "ptWeightedDphiStrip: PF = " << clusterVariables_.tau_pt_weighted_dphi_strip(*pfTau, pfTau->decayMode()) << ", pat = " << clusterVariables_.tau_pt_weighted_dphi_strip(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode()) << std::endl;
					differences->Fill(19);
					differencesWeighted->Fill(19,std::abs(clusterVariables_.tau_pt_weighted_dphi_strip(*pfTau, pfTau->decayMode())-clusterVariables_.tau_pt_weighted_dphi_strip(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode())));
					difference_ptWeightedDphiStrip->Fill(std::abs(clusterVariables_.tau_pt_weighted_dphi_strip(*pfTau, pfTau->decayMode())-clusterVariables_.tau_pt_weighted_dphi_strip(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode())));
				}
				if(clusterVariables_.tau_pt_weighted_dr_signal(*pfTau, pfTau->decayMode()) != clusterVariables_.tau_pt_weighted_dr_signal(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode())){
					if(verbosity_) std::cout << "ptWeightedDrSignal: PF = " << clusterVariables_.tau_pt_weighted_dr_signal(*pfTau, pfTau->decayMode()) << ", pat = " << clusterVariables_.tau_pt_weighted_dr_signal(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode()) << std::endl;
					differences->Fill(20);
					differencesWeighted->Fill(20,std::abs(clusterVariables_.tau_pt_weighted_dr_signal(*pfTau, pfTau->decayMode())-clusterVariables_.tau_pt_weighted_dr_signal(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode())));
					difference_ptWeightedDrSignal->Fill(std::abs(clusterVariables_.tau_pt_weighted_dr_signal(*pfTau, pfTau->decayMode())-clusterVariables_.tau_pt_weighted_dr_signal(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode())));
				}
				if(clusterVariables_.tau_pt_weighted_dr_iso(*pfTau, pfTau->decayMode()) != clusterVariables_.tau_pt_weighted_dr_iso(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode())){
					if(verbosity_) std::cout << "ptWeightedDrIso: PF = " << clusterVariables_.tau_pt_weighted_dr_iso(*pfTau, pfTau->decayMode()) << ", pat = " << clusterVariables_.tau_pt_weighted_dr_iso(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode()) << std::endl;
					differences->Fill(21);
					differencesWeighted->Fill(21,std::abs(clusterVariables_.tau_pt_weighted_dr_iso(*pfTau, pfTau->decayMode())-clusterVariables_.tau_pt_weighted_dr_iso(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode())));
					difference_ptWeightedDrIso->Fill(std::abs(clusterVariables_.tau_pt_weighted_dr_iso(*pfTau, pfTau->decayMode())-clusterVariables_.tau_pt_weighted_dr_iso(*unmatchedTaus.at(iTau),unmatchedTaus.at(iTau)->decayMode())));
				}
				if(clusterVariables_.tau_leadTrackChi2(*pfTau) != unmatchedTaus.at(iTau)->leadingTrackNormChi2()){
					if(verbosity_) std::cout << "leadTrackChi2: PF = " << clusterVariables_.tau_leadTrackChi2(*pfTau) << ", pat = " << unmatchedTaus.at(iTau)->leadingTrackNormChi2() << std::endl;
					differences->Fill(22);
					differencesWeighted->Fill(22,std::abs(clusterVariables_.tau_leadTrackChi2(*pfTau)-unmatchedTaus.at(iTau)->leadingTrackNormChi2()));
				}
				if(clusterVariables_.tau_Eratio(*pfTau) != clusterVariables_.tau_Eratio(*unmatchedTaus.at(iTau))){
					if(verbosity_) std::cout << "eRatio: PF = " << clusterVariables_.tau_Eratio(*pfTau) << ", pat = " << clusterVariables_.tau_Eratio(*unmatchedTaus.at(iTau)) << std::endl;
					differences->Fill(23);
					differencesWeighted->Fill(23,std::abs(clusterVariables_.tau_Eratio(*pfTau)-clusterVariables_.tau_Eratio(*unmatchedTaus.at(iTau))));
				}
				if(verbosity_) std::cout << "=============================================================" << std::endl;
			}
		}
	}
}


// ------------ method called once each job just before starting event loop  ------------
void 
rerunMVAIsolationOnMiniAOD::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
rerunMVAIsolationOnMiniAOD::endJob()
{
	mvaValueAOD->Write();
	mvaValueMiniAOD->Write();
	mvaValueDiff->Write();

	differences->Write();
	differencesWeighted->Write();

	difference_dxy->Write();
	difference_dxySig->Write();
	difference_ip3d->Write();
	difference_ip3dSig->Write();
	difference_flightlengthSig->Write();
	difference_ptWeightedDetaStrip->Write();
	difference_ptWeightedDphiStrip->Write();
	difference_ptWeightedDrSignal->Write();
	difference_ptWeightedDrIso->Write();

	mvaValue->Write();
	mvaValue_vLoose->Write();
	mvaValue_Loose->Write();
	mvaValue_Medium->Write();
	mvaValue_Tight->Write();
	mvaValue_vTight->Write();
	mvaValue_vvTight->Write();
	mvaValue_antiEMVA6->Write();

	decayMode->Write();
	chargedIsoPtSum->Write();
	neutralIsoPtSum->Write();
	puCorrPtSum->Write();
	photonPtSumOutsideSignalCone->Write();
	footprintCorrection->Write();

	decayDistMag->Write();
	dxy->Write();
	dxySig->Write();
	ip3d->Write();
	ip3dSig->Write();
	hasSV->Write();
	flightlengthSig->Write();
	nPhoton->Write();
	ptWeightedDetaStrip->Write();
	ptWeightedDphiStrip->Write();
	ptWeightedDrSignal->Write();
	ptWeightedDrIsolation->Write();
	leadTrackChi2->Write();
	eRatio->Write();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
rerunMVAIsolationOnMiniAOD::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(rerunMVAIsolationOnMiniAOD);
