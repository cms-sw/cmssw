//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                      //
//                                    WMuNuTriggerValidation                                                            //
//          Intended for a prompt validation of the muon triggers used in EWK samples.                                  // 
//                                                                                                                      //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "TH1D.h"
#include <map>
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"


class WMuNuTriggerValidation : public edm::EDFilter {
public:
  WMuNuTriggerValidation (const edm::ParameterSet &);
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void beginJob();
  virtual void endJob();
  void init_histograms();
  double HLTMatch(edm::Handle<trigger::TriggerEvent> triggerObj,const reco::Muon&, std::string);
  void FillTurnOnCurve(std::string TriggerTag);
  void FillEfficiencyPlots(std::string TriggerTag, std::string Type);

private:
  bool fastOption_;
  edm::InputTag trigTag_;
  edm::InputTag trigEv_;

  edm::InputTag muonTag_;
  edm::InputTag metTag_;
  bool metIncludesMuons_;
  edm::InputTag jetTag_;

  const std::string muonTrig_;
  double ptCut_;
  double etaMinCut_;
  double etaMaxCut_;

  double dxyCut_;
  double normalizedChi2Cut_;
  int trackerHitsCut_;
  int muonHitsCut_;
  bool isAlsoTrackerMuon_;

  unsigned int nall;

  std::map<std::string,TH1D*> h1_;


};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DataFormats/Common/interface/View.h"
  
using namespace edm;
using namespace std;
using namespace reco;

const int numberOfHLTTriggers=6;
std::string HLTTriggers[numberOfHLTTriggers]={"HLT_L1DoubleMuOpen","HLT_L1Mu", "HLT_L1Mu20","HLT_L1MuOpen","HLT_L2Mu11","HLT_L2Mu9"};

std::string hltpath[numberOfHLTTriggers]={"hltDoubleMuLevel1PathL1OpenFiltered::HLT","hltL1MuL1Filtered0::HLT", "hltL1Mu20L1Filtered20::HLT","hltL1MuOpenL1Filtered0::HLT","hltL2Mu11L2Filtered11::HLT","hltL2Mu9Filtered9::HLT"};

WMuNuTriggerValidation::WMuNuTriggerValidation( const ParameterSet & cfg ) :
      fastOption_(cfg.getUntrackedParameter<bool> ("FastOption", false)),

      // Input collections
      trigTag_(cfg.getUntrackedParameter<edm::InputTag> ("TrigTag", edm::InputTag("TriggerResults::HLT"))),
      trigEv_(cfg.getUntrackedParameter<edm::InputTag> ("triggerEvent",edm::InputTag("hltTriggerSummaryAOD::HLT"))),
      muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met"))),
      metIncludesMuons_(cfg.getUntrackedParameter<bool> ("METIncludesMuons", false)),
      jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("sisCone5CaloJets"))),

      // Main cuts 
      ptCut_(cfg.getUntrackedParameter<double>("PtCut", 0.)),
      etaMinCut_(cfg.getUntrackedParameter<double>("EtaMinCut", -2.6)),
      etaMaxCut_(cfg.getUntrackedParameter<double>("EtaMaxCut", 2.6)),

      // Muon quality cuts
      dxyCut_(cfg.getUntrackedParameter<double>("DxyCut", 0.2)),
      normalizedChi2Cut_(cfg.getUntrackedParameter<double>("NormalizedChi2Cut", 10.)),
      trackerHitsCut_(cfg.getUntrackedParameter<int>("TrackerHitsCut", 11)),
      muonHitsCut_(cfg.getUntrackedParameter<int>("MuonHitsCut", 0)),
      isAlsoTrackerMuon_(cfg.getUntrackedParameter<bool>("IsAlsoTrackerMuon", true))
{
}

void WMuNuTriggerValidation::beginJob() {
      nall = 0;
      init_histograms();
}

void WMuNuTriggerValidation::init_histograms() {
      edm::Service<TFileService> fs;
 
      TFileDirectory subDir1 = fs->mkdir("Histos");
      TFileDirectory subDir2 = fs->mkdir("Efficiencies");
      TFileDirectory subDir3 = fs->mkdir("TurnOnCurves");
     
      TFileDirectory* subDir[3]; subDir[0]=&subDir1; subDir[1]=&subDir2; subDir[2]=&subDir3;

      char chname[256] = "";
      char chtitle[256] = "";

      for (int i=0; i<numberOfHLTTriggers; i++) {

            // Filling standard histograms (before and after triggers)
            snprintf(chname, 255, "PT_%s", HLTTriggers[i].data());
            snprintf(chtitle, 255, "PT of Muons passing %s",HLTTriggers[i].data());
            h1_[chname] = subDir[0]->make<TH1D>(chname,chtitle,100,0.,100.);

            snprintf(chname, 255, "ETA_%s", HLTTriggers[i].data());
            snprintf(chtitle, 255, "ETA of Muons passing %s",HLTTriggers[i].data());
            h1_[chname] = subDir[0]->make<TH1D>(chname,chtitle,100,-2.5,2.5);

            snprintf(chname, 255, "PHI_%s", HLTTriggers[i].data());
            snprintf(chtitle, 255, "PHI of Muons passing %s",HLTTriggers[i].data());
            h1_[chname] = subDir[0]->make<TH1D>(chname,chtitle,100,-M_PI,M_PI);


            // Efficiency plots (so #triggered / #all for each bin) 
            snprintf(chname, 255, "Effi_PT_%s", HLTTriggers[i].data());
            snprintf(chtitle, 255, "Efficiency vs PT of Muons passing %s",HLTTriggers[i].data());
            h1_[chname] = subDir[1]->make<TH1D>(chname,chtitle,100,0.,100.);

            snprintf(chname, 255, "Effi_ETA_%s", HLTTriggers[i].data());
            snprintf(chtitle, 255, "Efficiency vs ETA of Muons passing %s",HLTTriggers[i].data());
            h1_[chname] = subDir[1]->make<TH1D>(chname,chtitle,100,-2.5,2.5);

            snprintf(chname, 255, "Effi_PHI_%s", HLTTriggers[i].data());
            snprintf(chtitle, 255, "Efficiency vs PHI of Muons passing %s",HLTTriggers[i].data());
            h1_[chname] = subDir[1]->make<TH1D>(chname,chtitle,100,-M_PI,M_PI);

            // Turn-On curves (so #triggered / # all but cumulative for pt>PtBin)

            snprintf(chname, 255, "TurnOn_%s", HLTTriggers[i].data());
            snprintf(chtitle, 255, "Turn-On Curve for %s",HLTTriggers[i].data());
            h1_[chname] = subDir[2]->make<TH1D>(chname,chtitle,100,0.,100.);


            // Turn-On curves (so #triggered / # all but cumulative for pt>PtBin)
            snprintf(chname, 255, "dR_%s", HLTTriggers[i].data());
            snprintf(chtitle, 255, "dR between muon and triggerobject %s",HLTTriggers[i].data());
            h1_[chname] = subDir[0]->make<TH1D>(chname,chtitle,200,0,2.);



 
      }


      // Plots without cuts (for normalization)
      h1_["PT_AllMuons"] = subDir[0]->make<TH1D>("PT_AllMuons","Muon Pt",100,0.,100.);
      h1_["ETA_AllMuons"] = subDir[0]->make<TH1D>("ETA_AllMuons","Muon Eta",100,-2.5,2.5);
      h1_["PHI_AllMuons"] = subDir[0]->make<TH1D>("PHI_AllMuons","Muon Phi",100,-M_PI,M_PI);


}

void WMuNuTriggerValidation::FillEfficiencyPlots(std::string TriggerTag, std::string Type){

      // Divide histograms before and after applying the trigger (for now consider errors binomial)
      char chname[256] = "";
      for (int i=0; i<101; i++){
         snprintf(chname, 255, "%s_%s",Type.data(), TriggerTag.data());
         double numerator= h1_[chname] -> GetBinContent(i);
         snprintf(chname, 255, "%s_AllMuons",Type.data());
         double denominator= h1_[chname] -> GetBinContent(i);
         double effi=0, effiError=0;     

         if (denominator!=0) {
            effi=numerator/denominator;
            effiError=sqrt(effi*(1-effi)/denominator);   
            snprintf(chname, 255, "Effi_%s_%s", Type.data(),TriggerTag.data());

            h1_[chname] -> SetBinContent(i,effi); 
            h1_[chname] -> SetBinError(i,effiError);
         }    
      } 

}

void WMuNuTriggerValidation::FillTurnOnCurve(std::string TriggerTag){

      char chname[256] = "";
      for (int i=0; i<101; i++){
         snprintf(chname, 255, "PT_%s", TriggerTag.data());
         double numerator= h1_[chname] -> Integral(i,-1);
         double denominator= h1_["PT_AllMuons"] -> Integral(i,-1);
         double effi=0, effiError=0;

         if (denominator!=0) {
            effi=numerator/denominator;
            effiError=sqrt(effi*(1-effi)/denominator);
            snprintf(chname, 255, "TurnOn_%s", TriggerTag.data());
            h1_[chname] -> SetBinContent(i,effi);
            h1_[chname] -> SetBinError(i,effiError);
         }
      }

}


void WMuNuTriggerValidation::endJob() {
    LogTrace("")<<"Analized "<<nall<<" events"<<endl;
    if (nall>0){
    for (int i=0; i<numberOfHLTTriggers; i++) {
            FillTurnOnCurve( HLTTriggers[i].data());
            FillEfficiencyPlots(HLTTriggers[i].data(), "PT");
            FillEfficiencyPlots(HLTTriggers[i].data(), "ETA");
            FillEfficiencyPlots(HLTTriggers[i].data(), "PHI");
    }  
    }
}

double WMuNuTriggerValidation::HLTMatch(edm::Handle<trigger::TriggerEvent> triggerObj, const Muon& mu, std::string hltFilterTag_){
       double minDR=100000.;

       const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());
       //LogDebug("")<<"TriggerObject Size: "<<triggerObj->sizeFilters();  
       for ( size_t ia = 0; ia < triggerObj->sizeFilters(); ++ ia) {
            //LogDebug("")<<"Tags: "<< triggerObj->filterTag(ia);
            if( triggerObj->filterTag(ia)  == hltFilterTag_) {
                  const trigger::Keys & k = triggerObj->filterKeys(ia);
                  for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
                        double pttrig= toc[*ki].pt();
                        double etatrig=toc[*ki].eta();
                        double phitrig=toc[*ki].phi();
                        Geom::Phi<double> deltaphi(phitrig-mu.phi());
                        double dRtrig=sqrt(pow(etatrig-mu.eta(),2)+pow(deltaphi.value(),2));
                        if(dRtrig < minDR ) {minDR=dRtrig;}
                  }
            }
       }

      return minDR;
}


bool WMuNuTriggerValidation::filter (Event & ev, const EventSetup &) {

      nall++;
      // Muon collection
      Handle<View<Muon> > muonCollection;
      if (!ev.getByLabel(muonTag_, muonCollection)) {
            LogError("") << ">>> Muon collection does not exist !!!";
            return false;
      }
      unsigned int muonCollectionSize = muonCollection->size();

      // Beam spot
      Handle<reco::BeamSpot> beamSpotHandle;
      if (!ev.getByLabel(InputTag("offlineBeamSpot"), beamSpotHandle)) {
            LogTrace("") << ">>> No beam spot found !!!";
            return false;
      }
  
      // Trigger
      Handle<TriggerResults> triggerResultsHandle;
      if (!ev.getByLabel(trigTag_, triggerResultsHandle)) {
            LogError("") << ">>> TRIGGER collection does not exist !!!";
            return false;
      }

      const edm::TriggerResults& triggerResults = (*triggerResultsHandle);
      edm::TriggerNames triggerNames;
      triggerNames.init(triggerResults);
      LogTrace("") << ">>>>> HLT bits fired in the event:\n";
      for (unsigned int bit=0; bit<triggerResults.size(); ++bit) {
            if (triggerResults.accept(bit)) {
                  LogTrace("") << "\t Bit "<< bit <<" : "<<triggerNames.triggerName(bit).data();
            }
      }

      //bool MuonL1TriggersFired[numberOfL1Triggers];
      bool MuonHLTTriggersFired[10];
      char histoname[256] = "";
      for (int j=0; j<numberOfHLTTriggers; j++){
            MuonHLTTriggersFired[j]=false;
                  for (unsigned int i=0; i<triggerResults.size(); i++) {
                        std::string trigName = triggerNames.triggerName(i);
                        if ( trigName == HLTTriggers[j] && triggerResults.accept(i)) {
                              MuonHLTTriggersFired[j] = true;
                        }
                  }   
      }

      edm::Handle<trigger::TriggerEvent> triggerObj;
      if(!ev.getByLabel(trigEv_,triggerObj)){
              LogTrace("") << "Summary HLT objects not found, skipping event";
               return false;
       } 


      bool muon_sel=true;

      for (unsigned int i=0; i<muonCollectionSize; i++) {

            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            if (mu.globalTrack().isNull()) continue;
            if (mu.innerTrack().isNull()) continue;
            reco::TrackRef gm = mu.globalTrack();

            double pt = mu.pt();
            double eta = mu.eta();
            double phi = mu.phi();

            // d0, chi2, nhits quality cuts
            double dxy = gm->dxy(beamSpotHandle->position());
            double normalizedChi2 = gm->normalizedChi2();
            double validmuonhits=gm->hitPattern().numberOfValidMuonHits();
            double trackerHits = gm->hitPattern().numberOfValidTrackerHits(); 
            
            if (fabs(dxy)>=dxyCut_) muon_sel = false; 
            if (validmuonhits<=muonHitsCut_) muon_sel = false;
            if (normalizedChi2>normalizedChi2Cut_) muon_sel = false;                       
            if (trackerHits<=trackerHitsCut_) muon_sel = false; 
            if (!mu.isTrackerMuon()) muon_sel = false; 

            // Other cuts?
            if (pt<=ptCut_) muon_sel = false; // This is just here for possible further studies of trigger eficciencies by 
            if (eta<=etaMinCut_ || eta>=etaMaxCut_) muon_sel = false;  // parts of the detector, not for the turn on curves
                        

            if(muon_sel){
            // Plots for Trigger (turn on curves will be filled in endjob)
            h1_["PT_AllMuons"]->Fill(pt);
            h1_["ETA_AllMuons"]->Fill(eta);
            h1_["PHI_AllMuons"]->Fill(phi);


            for (int j=0; j<numberOfHLTTriggers; j++){
                  if(MuonHLTTriggersFired[j]){
                        double dR=HLTMatch(triggerObj,mu,hltpath[j]);
                        snprintf(histoname, 255, "dR_%s", HLTTriggers[j].data());
                        h1_[histoname]->Fill(dR);
                                    if(dR<0.5){
                                    snprintf(histoname, 255, "PT_%s", HLTTriggers[j].data());
                                    h1_[histoname]->Fill(pt);
                                    snprintf(histoname, 255, "ETA_%s", HLTTriggers[j].data());
                                    h1_[histoname]->Fill(eta);
                                    snprintf(histoname, 255, "PHI_%s", HLTTriggers[j].data());
                                    h1_[histoname]->Fill(phi);
                                    }
                  }
            }

            }

      }


      return true;  // this is really not a filter... :-)

}

#include "FWCore/Framework/interface/MakerMacros.h"

      DEFINE_FWK_MODULE( WMuNuTriggerValidation );
