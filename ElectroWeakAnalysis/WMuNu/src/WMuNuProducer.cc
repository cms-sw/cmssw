#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TH1D.h"
#include <map>
// system include files
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/Common/interface/View.h"

#include "DataFormats/Candidate/src/CompositeCandidate.cc"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "ElectroWeakAnalysis/WMuNu/interface/WMuNuCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"

class WMuNuProducer : public edm::EDProducer {
public:
  WMuNuProducer(const edm::ParameterSet&);
  ~WMuNuProducer();
  
private:

  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();

  edm::InputTag trigTag_;
  edm::InputTag muonTag_;
  edm::InputTag metTag_;
  bool metIncludesMuons_;
  edm::InputTag jetTag_;

  bool applyPreselection_;
  const std::string muonTrig_;
  bool useTrackerPt_;
  double ptThrForZ1_;
  double ptThrForZ2_;
  double eJetMin_;
  int nJetMax_;

  const std::string WMuNuCollectionTag_;


  unsigned int nall;
  unsigned int npresel;


};
  
using namespace edm;
using namespace std;
using namespace reco;

WMuNuProducer::WMuNuProducer( const ParameterSet & cfg ) :
      // Input collections
      trigTag_(cfg.getUntrackedParameter<edm::InputTag> ("TrigTag", edm::InputTag("TriggerResults::HLT"))),
      muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met"))),
      metIncludesMuons_(cfg.getUntrackedParameter<bool> ("METIncludesMuons", false)),
      jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("sisCone5CaloJets"))),

      // Preselection cuts 
      applyPreselection_(cfg.getUntrackedParameter<bool>("ApplyPreselection",true)),
      muonTrig_(cfg.getUntrackedParameter<std::string>("MuonTrig", "HLT_Mu9")),
      useTrackerPt_(cfg.getUntrackedParameter<bool>("UseTrackerPt", true)),
      ptThrForZ1_(cfg.getUntrackedParameter<double>("PtThrForZ1", 20.)),
      ptThrForZ2_(cfg.getUntrackedParameter<double>("PtThrForZ2", 10.)),
      eJetMin_(cfg.getUntrackedParameter<double>("EJetMin", 999999.)),
      nJetMax_(cfg.getUntrackedParameter<int>("NJetMax", 999999))
{
  produces< WMuNuCandidateCollection >("WMuNuCandidates");
}

void WMuNuProducer::beginJob(const EventSetup &) {
      nall = 0;
      npresel = 0;
}

void WMuNuProducer::endJob() {
    double all = nall;
      double esel = npresel/all;
      LogVerbatim("") << "\n>>>>>> W SELECTION SUMMARY BEGIN >>>>>>>>>>>>>>>";
      LogVerbatim("") << "Total numer of events analyzed: " << nall << " [events]";
      LogVerbatim("") << "Total numer of events selected: " << npresel << " [events]";
      LogVerbatim("") << "Overall efficiency:             " << "(" << setprecision(4) << esel*100. <<" +/- "<< setprecision(2) << sqrt(esel*(1-esel)/all)*100. << ")%";
      LogVerbatim("") << "\n>>>>>> W SELECTION SUMMARY END >>>>>>>>>>>>>>>";
}


WMuNuProducer::~WMuNuProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


void WMuNuProducer::produce (Event & ev, const EventSetup &) {

      // Muon collection
      Handle<View<Muon> > muonCollection;
      if (!ev.getByLabel(muonTag_, muonCollection)) {
            LogError("") << ">>> Muon collection does not exist !!!";
            return;
      }
      unsigned int muonCollectionSize = muonCollection->size();

      // MET
      Handle<View<MET> > metCollection;
      if (!ev.getByLabel(metTag_, metCollection)) {
            LogError("") << ">>> MET collection does not exist !!!";
            return;
      }
      const MET& Met = metCollection->at(0);

      // Trigger
      Handle<TriggerResults> triggerResults;
      TriggerNames trigNames;
      if (!ev.getByLabel(trigTag_, triggerResults)) {
            LogError("") << ">>> TRIGGER collection does not exist !!!";
            return;
      }
      trigNames.init(*triggerResults);
      bool trigger_fired = false;
      int itrig1 = trigNames.triggerIndex(muonTrig_);
      if (triggerResults->accept(itrig1)) trigger_fired = true;
      LogTrace("") << ">>> Trigger bit: " << trigger_fired << " (" << muonTrig_ << ")";

      // Loop to reject/control Z->mumu is done separately
      unsigned int nmuonsForZ1 = 0;
      unsigned int nmuonsForZ2 = 0;
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            double pt = mu.pt();
            if (useTrackerPt_) {
                  reco::TrackRef tk = mu.innerTrack();
                  if (mu.innerTrack().isNull()) continue;
                  pt = tk->pt();
            }
            if (pt>ptThrForZ1_) nmuonsForZ1++;
            if (pt>ptThrForZ2_) nmuonsForZ2++;
      }
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ1_ << " [GeV]: " << nmuonsForZ1;
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ2_ << " [GeV]: " << nmuonsForZ2;
      
      // Jet collection
      Handle<View<Jet> > jetCollection;
      if (!ev.getByLabel(jetTag_, jetCollection)) {
            LogError("") << ">>> JET collection does not exist !!!";
            return;
      }
      unsigned int jetCollectionSize = jetCollection->size();
      int njets = 0;
      for (unsigned int i=0; i<jetCollectionSize; i++) {
            const Jet& jet = jetCollection->at(i);
            if (jet.et()>eJetMin_) njets++;
      }
      LogTrace("") << ">>> Total number of jets: " << jetCollectionSize;
      LogTrace("") << ">>> Number of jets above " << eJetMin_ << " [GeV]: " << njets;

      
      // Ensure there is at least 1 muon candidate to build the W...
      nall++;

      if (muonCollectionSize<1) return;

      // Preselection of events: ensure there is 1 and only 1 muon candidate to build the W... This is only for the Inclusive Analysis
      if(applyPreselection_){ 
      if (!trigger_fired) return;
      if (nmuonsForZ1>=0 && nmuonsForZ2>=2) return;
      if (njets>nJetMax_) return;
      } 
      npresel ++;   
      

      double MaxPt[5]={0,0,0,0,0};
      int Maxi[5]={0,0,0,0,0};
     
      // Now order remaining Muons by Pt  - there should be only one if preselection has been applied
      
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            if (mu.globalTrack().isNull()) continue;
            if (mu.innerTrack().isNull()) continue;

            reco::TrackRef gm = mu.globalTrack();
            reco::TrackRef tk = mu.innerTrack();

            // Pt,eta cuts
            double pt = mu.pt();
            if (useTrackerPt_) pt = tk->pt();
            bool foundPosition=false; int j=0;
            do{   if (pt > MaxPt[j]) { MaxPt[j]=pt; Maxi[j]=i; foundPosition=true;}
                  j++;
            }while(!foundPosition);
               
     }
     auto_ptr< WMuNuCandidateCollection > WMuNuCandidates(new WMuNuCandidateCollection );

     for (int indx=0; indx<5; indx++){ 
     int MuonIndx=Maxi[indx]; 
     const Muon& Muon = muonCollection->at(MuonIndx);
     edm::Ptr<reco::Muon> muon(muonCollection,MuonIndx);
   
      // Build WMuNuCandidate
      LogTrace("")<<"Building WMuNu Candidate!"; 
      WMuNuCandidate* WCand = new WMuNuCandidate(Muon,Met); WCand->setMuon(muonCollection,MuonIndx); WCand->setNeutrino(metCollection,0);
      LogTrace("") << "\t... W mass, W_et, W_px, W_py: "<<WCand->massT(useTrackerPt_)<<", "<<WCand->eT(useTrackerPt_)<<"[GeV]";
      LogTrace("") << "\t... W_px, W_py: "<<WCand->px()<<", "<< WCand->py() <<"[GeV]";
      LogTrace("") << "\t... acop  " << WCand->acop();
      LogTrace("") << "\t... Muon pt, px, py, pz: "<<WCand->getMuon().pt()<<", "<<WCand->getMuon().px()<<", "<<WCand->getMuon().py()<<", "<< WCand->getMuon().pz()<<" [GeV]";
      LogTrace("") << "\t... Met  met_et, met_px, met_py : "<<WCand->getNeutrino().pt()<<", "<<WCand->getNeutrino().px()<<", "<<WCand->getNeutrino().py()<<" [GeV]";
  	WMuNuCandidates->push_back(*WCand);
       } 

      ev.put(WMuNuCandidates,"WMuNuCandidates");

}

DEFINE_FWK_MODULE(WMuNuProducer);
