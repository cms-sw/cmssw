/** \class WMuNuValidator
 *  Simple Validator to make some W->MuNu plots
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "TH1D.h"
//#include "TH2D.h"
#include <map>
#include <string>

class WMuNuValidator : public edm::EDAnalyzer {
public:
      WMuNuValidator(const edm::ParameterSet& pset);
      virtual ~WMuNuValidator();
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
private:
      edm::InputTag trigTag_;
      edm::InputTag muonTag_;
      edm::InputTag metTag_;
      bool metIncludesMuons_;
      edm::InputTag jetTag_;
      const std::string muonTrig_;
      bool isCombinedIso_;
      double eJetMin_;

// Histograms
      std::map<std::string,TH1D*> h1_;
      //std::map<std::string,TH2D*> h2_;
  
      unsigned int numberOfEvents;
      unsigned int numberOfMuons;
};

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/View.h"

using namespace std;
using namespace edm;
using namespace reco;

/// Constructor
WMuNuValidator::WMuNuValidator(const ParameterSet& cfg) :
      trigTag_(cfg.getUntrackedParameter<edm::InputTag> ("TrigTag", edm::InputTag("TriggerResults::HLT"))),
      muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met"))),
      metIncludesMuons_(cfg.getUntrackedParameter<bool> ("METIncludesMuons", false)),
      jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("sisCone5CaloJets"))),
      muonTrig_(cfg.getUntrackedParameter<std::string> ("MuonTrig", "HLT_Mu9")),

      isCombinedIso_(cfg.getUntrackedParameter<bool>("IsCombinedIso", false)),
      eJetMin_(cfg.getUntrackedParameter<double>("EJetMin", 30.))
{
  LogDebug("WMuNuValidator")<<" WMuNuValidator constructor called";
}

/// Destructor
WMuNuValidator::~WMuNuValidator(){
}

void WMuNuValidator::beginJob(const EventSetup& eventSetup){
      edm::Service<TFileService> fs;

      numberOfEvents = 0;
      numberOfMuons = 0;

      h1_["TRIG"]  = fs->make<TH1D>("TRIG","Trigger flag",2,-0.5,1.5);
      h1_["NMU"]  = fs->make<TH1D>("NMU","Nb. muons in the event",10,-0.5,9.5);
      h1_["PTMU"] = fs->make<TH1D>("PTMU","Pt global muon (GeV)",100,0.,100.);
      h1_["PTTK"] = fs->make<TH1D>("PTTK","Pt inner track (GeV)",100,0.,100.);
      h1_["PTSA"] = fs->make<TH1D>("PTSA","Pt stand-alone muon (GeV)",100,0.,100.);
      h1_["ETAMU"] = fs->make<TH1D>("ETAMU","Eta mu",50,-2.5,2.5);
      h1_["DXY"] = fs->make<TH1D>("DXY","Transverse distance to beam spot (cm)",100,-1.,1.);
      h1_["CHI2"] = fs->make<TH1D>("CHI2","Chi2/ndof, inner track fit",100,0.,100.);
      h1_["NHITS"] = fs->make<TH1D>("NHITS","Number of hits in inner track",35,-0.5,34.5);
      h1_["MET"] = fs->make<TH1D>("MET","Missing Transverse Energy (GeV)", 100,0.,200.);
      h1_["TMASS"] = fs->make<TH1D>("TMASS","Rec. Transverse Mass (GeV)",150,0.,300.);
      h1_["ACOP"] = fs->make<TH1D>("ACOP","Mu-MET acoplanarity",50,0.,M_PI);
      h1_["NJETS"] = fs->make<TH1D>("NJETS","Number of jets above threshold",25,-0.5,24.5);
      h1_["ISOABS"] = fs->make<TH1D>("ISOABS","Transverse energy/momentum in isolation cone (GeV)", 100, 0., 50.);
      h1_["ISOREL"] = fs->make<TH1D>("ISOREL","Relative isolation variable", 100, 0., 1.);

}

void WMuNuValidator::endJob(){
      //LogVerbatim("") << "\n>>>>>> W VALIDATION SUMMARY BEGIN >>>>>>>>>>>>>>>";
      //LogVerbatim("") << "Number of analyzed events: " << numberOfEvents;
      //LogVerbatim("") << "Number of analyzed muons: " << numberOfMuons;
      //LogVerbatim("") << ">>>>>> W VALIDATION SUMMARY END   >>>>>>>>>>>>>>>\n";
}
 

void WMuNuValidator::analyze(const Event & event, const EventSetup& eventSetup){
  
      numberOfEvents++;

      // Get the Muon Track collection from the event
      Handle<View<Muon> > muonCollection;
      if (event.getByLabel(muonTag_, muonCollection)) {
            LogTrace("Validator")<<"Reconstructed Muon tracks: " << muonCollection->size() << endl;
      } else {
            LogTrace("") << ">>> Muon collection does not exist !!!";
            return;
      }
      unsigned int muonCollectionSize = muonCollection->size();
      numberOfMuons += muonCollectionSize;

      // Beam spot
      Handle<BeamSpot> beamSpotHandle;
      if (!event.getByLabel(InputTag("offlineBeamSpot"), beamSpotHandle)) {
            LogTrace("") << ">>> No beam spot found !!!";
            return;
      }
  
      // Get the MET collection from the event
      Handle<View<MET> > metCollection;
      if (event.getByLabel(metTag_, metCollection)) {
            LogTrace("Validator")<<"CaloMET collection found" << endl;
      } else {
            LogTrace("") << ">>> CaloMET collection does not exist !!!";
            return;
      }

      const MET& met = metCollection->at(0);
      double met_px = met.px();
      double met_py = met.py();
      if (!metIncludesMuons_) {
            for (unsigned int i=0; i<muonCollectionSize; i++) {
                  const Muon& mu = muonCollection->at(i);
                  if (!mu.isGlobalMuon()) continue;
                  met_px -= mu.px();
                  met_py -= mu.py();
            }
      }
      double met_et = sqrt(met_px*met_px+met_py*met_py);
      LogTrace("") << ">>> MET, MET_px, MET_py= " << met_et << ", " << met_px << ", " << met_py;
      h1_["MET"]->Fill(met_et);

      // Get the Jet collection from the event
      Handle<View<Jet> > jetCollection;
      if (event.getByLabel(jetTag_, jetCollection)) {
            LogTrace("Validator")<<"Reconstructed calojets: " << jetCollection->size() << endl;
      } else {
            LogTrace("") << ">>> CALOJET collection does not exist !!!";
            return;
      }
      unsigned int jetCollectionSize = jetCollection->size();
      int njets = 0;
      for (unsigned int i=0; i<jetCollectionSize; i++) {
            const Jet& jet = jetCollection->at(i);
            if (jet.et()>eJetMin_) njets++;
      }
      LogTrace("") << ">>> Total number of jets= " << jetCollectionSize;
      LogTrace("") << ">>> Number of jets above " << eJetMin_ << " GeV: " << njets;
      h1_["NJETS"]->Fill(njets);

      // Trigger
      Handle<TriggerResults> triggerResults;
      TriggerNames trigNames;
      if (!event.getByLabel(trigTag_, triggerResults)) {
                  LogError("") << ">>> TRIGGER collection does not exist !!!";
                  return;
      }
   
      bool trigger_sel = true;
      trigNames.init(*triggerResults);
      int itrig1 = trigNames.triggerIndex(muonTrig_);
      if (!triggerResults->accept(itrig1)) trigger_sel = false;
      h1_["TRIG"]->Fill((double)trigger_sel);

      unsigned int muons_in_this_event = 0;
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            if (mu.globalTrack().isNull()) continue;
            if (mu.innerTrack().isNull()) continue;
            if (mu.outerTrack().isNull()) continue;

            muons_in_this_event++;
            LogTrace("") << "> Processing (global) muon number " << i << "...";
            TrackRef gm = mu.globalTrack();
            TrackRef tk = mu.innerTrack();
            TrackRef sa = mu.outerTrack();

            // pt
            h1_["PTMU"]->Fill(gm->pt());
            h1_["PTTK"]->Fill(tk->pt());
            h1_["PTSA"]->Fill(sa->pt());
            LogTrace("") << "\t... pt_gm (GeV)= " << gm->pt();
            LogTrace("") << "\t... pt_tk (GeV)= " << tk->pt();
            LogTrace("") << "\t... pt_sa (GeV)= " << sa->pt();

            // eta
            double eta = tk->eta();
            h1_["ETAMU"]->Fill(eta);
            LogTrace("") << "\t... eta= " << eta;

            // d0, chi2, nhits
            double dxy = tk->dxy(beamSpotHandle->position());
            double normalizedChi2 = gm->normalizedChi2();
            double trackerHits = tk->numberOfValidHits();
            h1_["DXY"]->Fill(dxy);
            h1_["CHI2"]->Fill(normalizedChi2);
            h1_["NHITS"]->Fill(trackerHits);
            LogTrace("") << "\t... dxy, normalizedChi2, trackerHits: " << dxy << " [cm], " << normalizedChi2 << ", " << trackerHits;

            // acoplanarity
            Geom::Phi<double> deltaphi(tk->phi()-atan2(met_py,met_px));
            double acop = deltaphi.value();
            if (acop<0) acop = - acop;
            acop = M_PI - acop;
            h1_["ACOP"]->Fill(acop);
            LogTrace("") << "\t... acop= " << acop;

            // transverse mass
            double w_et = tk->pt() + met_et;
            double w_px = tk->px() + met_px;
            double w_py = tk->py() + met_py;
            double massT = w_et*w_et - w_px*w_px - w_py*w_py;
            massT = (massT>0) ? sqrt(massT) : 0;
            h1_["TMASS"]->Fill(massT);
            LogTrace("") << "\t... W_et, W_px, W_py= " << w_et << ", " << w_px << ", " << w_py << " GeV";
            LogTrace("") << "\t... Invariant transverse mass= " << massT << " GeV";
      
            // Isolation
            double etsum = mu.isolationR03().sumPt;
            if (isCombinedIso_) {
                  etsum += mu.isolationR03().emEt;
                  etsum += mu.isolationR03().hadEt;
            }
            h1_["ISOABS"]->Fill(etsum);
            h1_["ISOREL"]->Fill(etsum/mu.pt());
            LogTrace("") << "\t... Isol, Muon pt= " << mu.pt() << " GeV, " << " etsum = " << etsum;

      }

      h1_["NMU"]->Fill(muons_in_this_event);
      numberOfMuons += muons_in_this_event;

      return;
  
}

DEFINE_FWK_MODULE(WMuNuValidator);
