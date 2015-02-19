// -*- C++ -*-
//
// Package:    EGammaCutBasedEleIdAnalyzer
// Class:      EGammaCutBasedEleIdAnalyzer
//
/**\class EGammaCutBasedEleIdAnalyzer EGammaCutBasedEleIdAnalyzer.cc EGamma/EGammaCutBasedEleIdAnalyzer/src/EGammaCutBasedEleIdAnalyzer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
 */
//
// Original Author:  Dave Evans,510 1-015,+41227679496,
//         Created:  Tue Apr 10 11:17:29 CEST 2012
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "EgammaAnalysis/ElectronTools/interface/EGammaCutBasedEleId.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEffectiveArea.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <TFile.h>
#include <TH1F.h>

//
// class declaration
//

class EGammaCutBasedEleIdAnalyzer : public edm::EDAnalyzer {
    public:

        typedef std::vector< edm::Handle< edm::ValueMap<reco::IsoDeposit> > >   IsoDepositMaps;
        typedef std::vector< edm::Handle< edm::ValueMap<double> > >             IsoDepositVals;


        explicit EGammaCutBasedEleIdAnalyzer(const edm::ParameterSet&);
        ~EGammaCutBasedEleIdAnalyzer();

        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
	ElectronEffectiveArea::ElectronEffectiveAreaTarget EAtarget;


    private:
        virtual void beginJob() ;
        virtual void analyze(const edm::Event&, const edm::EventSetup&);
        virtual void endJob() ;

        virtual void beginRun(edm::Run const&, edm::EventSetup const&);
        virtual void endRun(edm::Run const&, edm::EventSetup const&);
        virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
        virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

        // ----------member data ---------------------------

        // input tags
        edm::EDGetTokenT<reco::GsfElectronCollection>               electronsToken_;
        edm::EDGetTokenT<reco::ConversionCollection>               conversionsToken_;
        edm::EDGetTokenT<reco::BeamSpot>               beamSpotToken_;
        edm::EDGetTokenT<double>               rhoIsoToken_;
        edm::EDGetTokenT<reco::VertexCollection>               primaryVertexToken_;
        std::vector<edm::EDGetTokenT<edm::ValueMap<double> > >  isoValTokens_;
	std::string EAtargetToken_;

        // debug
        bool printDebug_;

        // histograms
        TH1F *h1_pt_;
        TH1F *h1_pt_veto_;
        TH1F *h1_pt_loose_;
        TH1F *h1_pt_medium_;
        TH1F *h1_pt_tight_;
        TH1F *h1_pt_trig_;
        TH1F *h1_pt_fbremeopin_;

};

//
// static data member definitions
//

//
// constructors and destructor
//
EGammaCutBasedEleIdAnalyzer::EGammaCutBasedEleIdAnalyzer(const edm::ParameterSet& iConfig)
{

    // get input parameters
    electronsToken_      = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electronsInputTag"));
    conversionsToken_    = consumes<reco::ConversionCollection>(iConfig.getParameter<edm::InputTag>("conversionsInputTag"));
    beamSpotToken_       = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotInputTag"));
    rhoIsoToken_         = consumes<double>(iConfig.getParameter<edm::InputTag>("rhoIsoInputTag"));
    primaryVertexToken_  = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertexInputTag"));
    isoValTokens_        = edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >("isoValInputTags"), [this](edm::InputTag const & tag){return consumes<edm::ValueMap<double> >(tag);});
    EAtargetToken_	= iConfig.getParameter<std::string>("EAtarget");//EleEANoCorr, EleEAData2011, EleEASummer11MC,EleEAFall11MC, EleEAData2012/ 

    // debug
    printDebug_             = iConfig.getParameter<bool>("printDebug");

    // output histograms
    edm::Service<TFileService> fs;

    h1_pt_               = fs->make<TH1F>("h1_pt",               "pt",              100, 0.0, 100.0);
    h1_pt_veto_          = fs->make<TH1F>("h1_pt_veto",          "pt (veto)",       100, 0.0, 100.0);
    h1_pt_loose_         = fs->make<TH1F>("h1_pt_loose",         "pt (loose)",      100, 0.0, 100.0);
    h1_pt_medium_        = fs->make<TH1F>("h1_pt_medium",        "pt (medium)",     100, 0.0, 100.0);
    h1_pt_tight_         = fs->make<TH1F>("h1_pt_tight",         "pt (tight)",      100, 0.0, 100.0);
    h1_pt_trig_          = fs->make<TH1F>("h1_pt_trig",          "pt (trig)",       100, 0.0, 100.0);
    h1_pt_fbremeopin_    = fs->make<TH1F>("h1_pt_fbremeopin",    "pt (fbremeopin)", 100, 0.0, 100.0);

}


EGammaCutBasedEleIdAnalyzer::~EGammaCutBasedEleIdAnalyzer()
{

    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
    void
EGammaCutBasedEleIdAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // electrons
    edm::Handle<reco::GsfElectronCollection> els_h;
    iEvent.getByToken(electronsToken_, els_h);

    // conversions
    edm::Handle<reco::ConversionCollection> conversions_h;
    iEvent.getByToken(conversionsToken_, conversions_h);

    // iso deposits
    IsoDepositVals isoVals(isoValTokens_.size());
    for (size_t j = 0; j < isoValTokens_.size(); ++j) {
        iEvent.getByToken(isoValTokens_[j], isoVals[j]);
    }

    // beam spot
    edm::Handle<reco::BeamSpot> beamspot_h;
    iEvent.getByToken(beamSpotToken_, beamspot_h);
    const reco::BeamSpot &beamSpot = *(beamspot_h.product());

    // vertices
    edm::Handle<reco::VertexCollection> vtx_h;
    iEvent.getByToken(primaryVertexToken_, vtx_h);

    // rho for isolation
    edm::Handle<double> rhoIso_h;
    iEvent.getByToken(rhoIsoToken_, rhoIso_h);
    double rhoIso = *(rhoIso_h.product());

    // loop on electrons
    unsigned int n = els_h->size();
    for(unsigned int i = 0; i < n; ++i) {

        // get reference to electron
        reco::GsfElectronRef ele(els_h, i);

        //
        // get particle flow isolation
        //

        double iso_ch =  (*(isoVals)[0])[ele];
        double iso_em = (*(isoVals)[1])[ele];
        double iso_nh = (*(isoVals)[2])[ele];

        //
        // test ID
        //

        // working points
        bool veto       = EgammaCutBasedEleId::PassWP(EgammaCutBasedEleId::VETO, ele, conversions_h, beamSpot, vtx_h, iso_ch, iso_em, iso_nh, rhoIso, EAtarget);
        bool loose      = EgammaCutBasedEleId::PassWP(EgammaCutBasedEleId::LOOSE, ele, conversions_h, beamSpot, vtx_h, iso_ch, iso_em, iso_nh, rhoIso, EAtarget);
        bool medium     = EgammaCutBasedEleId::PassWP(EgammaCutBasedEleId::MEDIUM, ele, conversions_h, beamSpot, vtx_h, iso_ch, iso_em, iso_nh, rhoIso, EAtarget);
        bool tight      = EgammaCutBasedEleId::PassWP(EgammaCutBasedEleId::TIGHT, ele, conversions_h, beamSpot, vtx_h, iso_ch, iso_em, iso_nh, rhoIso, EAtarget);

        // eop/fbrem cuts for extra tight ID
        bool fbremeopin = EgammaCutBasedEleId::PassEoverPCuts(ele);

        // cuts to match tight trigger requirements
        bool trigtight = EgammaCutBasedEleId::PassTriggerCuts(EgammaCutBasedEleId::TRIGGERTIGHT, ele);

        // for 2011 WP70 trigger
        bool trigwp70 = EgammaCutBasedEleId::PassTriggerCuts(EgammaCutBasedEleId::TRIGGERWP70, ele);

        //
        // fill histograms
        //

        h1_pt_->Fill(ele->pt());
        if (veto)       h1_pt_veto_         ->Fill(ele->pt());
        if (loose)      h1_pt_loose_        ->Fill(ele->pt());
        if (medium)     h1_pt_medium_       ->Fill(ele->pt());
        if (tight)      h1_pt_tight_        ->Fill(ele->pt());
        if (trigtight)  h1_pt_trig_         ->Fill(ele->pt());
        if (fbremeopin) h1_pt_fbremeopin_   ->Fill(ele->pt());

        //
        // print decisions
        //

        if (printDebug_) {
            printf("%u %u %llu : ",       iEvent.id().run(), iEvent.luminosityBlock(), iEvent.id().event());
            printf("veto(%i), ",        veto);
            printf("loose(%i), ",       loose);
            printf("medium(%i), ",      medium);
            printf("tight(%i), ",       tight);
            printf("trigtight(%i), ",   trigtight);
            printf("trigwp70(%i), ",    trigwp70);
            printf("fbremeopin(%i)\n",  fbremeopin);
        }

    }

}


// ------------ method called once each job just before starting event loop  ------------
    void
EGammaCutBasedEleIdAnalyzer::beginJob()
{
  if( EAtargetToken_ == "EleEANoCorr")
    EAtarget  =ElectronEffectiveArea::kEleEANoCorr;
  else if( EAtargetToken_ == "EleEAData2011")
    EAtarget  =ElectronEffectiveArea::kEleEAData2011;
  else if( EAtargetToken_ == "EleEASummer11MC")
    EAtarget  =ElectronEffectiveArea::kEleEASummer11MC;
  else if( EAtargetToken_ == "EleEAFall11MC")
    EAtarget  =ElectronEffectiveArea::kEleEAFall11MC;
  else if( EAtargetToken_ == "EleEAData2012")
    EAtarget  =ElectronEffectiveArea::kEleEAData2012;
  else
    EAtarget  =ElectronEffectiveArea::kEleEAData2012;

}

// ------------ method called once each job just after ending the event loop  ------------
    void
EGammaCutBasedEleIdAnalyzer::endJob()
{
}

// ------------ method called when starting to processes a run  ------------
    void
EGammaCutBasedEleIdAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
    void
EGammaCutBasedEleIdAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
    void
EGammaCutBasedEleIdAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
    void
EGammaCutBasedEleIdAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
EGammaCutBasedEleIdAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    edm::ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EGammaCutBasedEleIdAnalyzer);
