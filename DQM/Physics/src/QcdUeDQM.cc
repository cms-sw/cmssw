
/*
    This is the DQM code for UE physics plots
    11/12/2009 Sunil Bansal
*/
#include "DQM/Physics/src/QcdUeDQM.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "CommonTools/RecoAlgos/src/TrackToRefCandidate.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h" 
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include <TString.h>
#include <TMath.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TProfile.h>
using namespace std;
using namespace edm;

#define CP(level) \
  if (level>=verbose_)

struct deleter {
  void operator()(TH3F *&h) { delete h; h=0;}
};


//--------------------------------------------------------------------------------------------------
QcdUeDQM::QcdUeDQM(const ParameterSet &parameters) :
  hltResName_(parameters.getUntrackedParameter<string>("hltTrgResults")),
  verbose_(parameters.getUntrackedParameter<int>("verbose",3)),
  tgeo_(0),
  theDbe_(0),
  repSumMap_(0),
  repSummary_(0),
  h2TrigCorr_(0),
  caloJetLabel_(parameters.getUntrackedParameter<edm::InputTag>("caloJetTag")),
  chargedJetLabel_(parameters.getUntrackedParameter<edm::InputTag>("chargedJetTag")),
  trackLabel_(parameters.getUntrackedParameter<edm::InputTag>("trackTag")),
  vtxLabel_(parameters.getUntrackedParameter<edm::InputTag>("vtxTag"))
{
  // Constructor.

  if (parameters.exists("hltTrgNames"))
    hltTrgNames_ = parameters.getUntrackedParameter<vector<string> >("hltTrgNames");

  if (parameters.exists("hltProcNames"))
     hltProcNames_ = parameters.getUntrackedParameter<vector<string> >("hltProcNames");
  else {
     hltProcNames_.push_back("FU");
     hltProcNames_.push_back("HLT");
  }

 
}

//--------------------------------------------------------------------------------------------------
QcdUeDQM::~QcdUeDQM()
{
  // Destructor.


}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::analyze(const Event &iEvent, const EventSetup &iSetup) 
{

  if( ! isHltConfigSuccessful_ ) return;


  // Analyze the given event.

   edm::Handle<reco::TrackCollection>tracks ;
   bool ValidTrack_ = iEvent.getByLabel(trackLabel_,tracks);
   if(!ValidTrack_)return;

   edm::Handle<reco::CandidateView> trkJets;
   bool ValidTrackJet_ = iEvent.getByLabel (chargedJetLabel_,trkJets);
   if(!ValidTrackJet_)return;
   
   edm::Handle<reco::CaloJetCollection> calJets;
   bool ValidCaloJet_ = iEvent.getByLabel (caloJetLabel_,calJets);
   if(!ValidCaloJet_)return;
 
   edm::Handle< reco::VertexCollection > vertexColl;
   bool ValidVtxColl_ = iEvent.getByLabel (vtxLabel_, vertexColl);
   if(!ValidVtxColl_)return;

   reco::TrackCollection tracks_sort = *tracks;
   std::sort(tracks_sort.begin(), tracks_sort.end(), PtSorter()); 
   

  // get tracker geometry
  ESHandle<TrackerGeometry> trackerHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  tgeo_ = trackerHandle.product();
  if (!tgeo_)return;

  
  fillHltBits(iEvent);
  bool validVtx = fillVtxPlots(iEvent, vertexColl);
  if(validVtx){
      fillpTMaxRelated(iEvent, tracks, vertexColl);
      fillChargedJetSpectra(iEvent, trkJets);  
      fillCaloJetSpectra(iEvent, calJets);
      fillUE_with_MaxpTtrack(iEvent, tracks_sort, vertexColl);
      fillUE_with_ChargedJets(iEvent, tracks_sort,trkJets, vertexColl); 
      fillUE_with_CaloJets(iEvent, tracks_sort,calJets, vertexColl);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::beginJob() 
{
  // Begin job and setup the DQM store.

  theDbe_ = Service<DQMStore>().operator->();
  if (!theDbe_)return;
  
  //  theDbe_->setCurrentFolder("Physics/QcdUe");
  
}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::beginLuminosityBlock(const LuminosityBlock &l, 
                                       const EventSetup &iSetup)
{
  // At the moment, nothing needed to be done.
}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::beginRun(const Run &run, const EventSetup &iSetup)
{

  isHltConfigSuccessful_ = false; // init

  // Begin run, get or create needed structures.  TODO: can this be called several times in DQM???

  //--- htlConfig_
  bool changed(true);
  if (hltConfig.init(run,iSetup,"HLT",changed)) {
    if (changed) {
      LogInfo("QcdUeDQM")  << "QcdUeDQM:analyze: The number of valid triggers has changed since beginning of job." << std::endl;
    }
  }

 if ( hltConfig.size() <= 0 ) return;
  


  bool isinit = false;
  string teststr;
  for(size_t i=0; i<hltProcNames_.size(); ++i) {
    if (i>0) 
      teststr += ", ";
    teststr += hltProcNames_.at(i);
    if (hltConfig.init(hltProcNames_.at(i))) {
      isinit = true;
      hltUsedResName_ = hltResName_;
      if (hltResName_.find(':')==string::npos)
        hltUsedResName_ += "::";
      else 
        hltUsedResName_ += ":";
      hltUsedResName_ += hltProcNames_.at(i);
      break;
    }
  }

  if (!isinit)return;

  // setup "Any" bit
  hltTrgBits_.clear();
  hltTrgBits_.push_back(-1);
  hltTrgDeci_.clear();
  hltTrgDeci_.push_back(true);
  hltTrgUsedNames_.clear();
  hltTrgUsedNames_.push_back("Any");

  // figure out relation of trigger name to trigger bit and store used trigger names/bits
  for(size_t i=0;i<hltTrgNames_.size();++i) {
    const string &n1(hltTrgNames_.at(i));
    bool found = 0;
    for(size_t j=0;j<hltConfig.size();++j) {
      const string &n2(hltConfig.triggerName(j));
      if (n2==n1) {
        hltTrgBits_.push_back(j);
        hltTrgUsedNames_.push_back(n1);
        hltTrgDeci_.push_back(false);
        found = 1;
        break;
      }
    }      
    if (!found) {
      CP(2) cout<<"Could not find trigger bit"<<endl ;
    }
  }
 
  // book monitoring histograms
  createHistos();
  isHltConfigSuccessful_ = true;

}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::book1D(std::vector<MonitorElement*> &mes, 
                         const std::string &name, const std::string &title, 
                         int nx, double x1, double x2, bool sumw2, bool sbox)
{
  // Book 1D histos.

  for(size_t i=0;i<hltTrgUsedNames_.size();++i) {
    MonitorElement *e = theDbe_->book1D(Form("%s_%s",name.c_str(),hltTrgUsedNames_.at(i).c_str()),
                                        Form("%s: %s",hltTrgUsedNames_.at(i).c_str(), title.c_str()), 
                                        nx, x1, x2);
    TH1 *h1 = e->getTH1();
    if (sumw2)
      h1->Sumw2();
    h1->SetStats(sbox);
    mes.push_back(e);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::book2D(std::vector<MonitorElement*> &mes, 
                         const std::string &name, const std::string &title, 
                         int nx, double x1, double x2, int ny, double y1, double y2, 
                         bool sumw2, bool sbox)
{
  // Book 2D histos.

  for(size_t i=0;i<hltTrgUsedNames_.size();++i) {
    MonitorElement *e = theDbe_->book2D(Form("%s_%s",name.c_str(),hltTrgUsedNames_.at(i).c_str()),
                                        Form("%s: %s",hltTrgUsedNames_.at(i).c_str(), title.c_str()), 
                                        nx, x1, x2, ny, y1, y2);
    TH1 *h1 = e->getTH1();
    if (sumw2)
      h1->Sumw2();
    h1->SetStats(sbox);
    mes.push_back(e);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::bookProfile(std::vector<MonitorElement*> &mes, 
                         const std::string &name, const std::string &title, 
                         int nx, double x1, double x2,  double y1, double y2, 
                         bool sumw2, bool sbox)
{
  // Book Profile histos.

  for(size_t i=0;i<hltTrgUsedNames_.size();++i) {
    MonitorElement *e = theDbe_->bookProfile(Form("%s_%s",name.c_str(),hltTrgUsedNames_.at(i).c_str()),
                                        Form("%s: %s",hltTrgUsedNames_.at(i).c_str(), title.c_str()), 
                                        nx, x1, x2, y1, y2," ");
    mes.push_back(e);
  }
}
//--------------------------------------------------------------------------------------------------
void QcdUeDQM::createHistos()
{
  // Book histograms if needed.


  if (1) {
    theDbe_->setCurrentFolder("Physics/EventInfo/");
    repSumMap_  = theDbe_->book2D("reportSummaryMap","reportSummaryMap",1,0,1,1,0,1);
    repSummary_ = theDbe_->bookFloat("reportSummary");
  }

   theDbe_->setCurrentFolder("Physics/QcdUe");

  if (1) {
    const int Nx = hltTrgUsedNames_.size();
    const double x1 = -0.5;
    const double x2 = Nx-0.5;
    h2TrigCorr_ = theDbe_->book2D("h2TriCorr","Trigger bit x vs y;y&&!x;x&&y",Nx,x1,x2,Nx,x1,x2);
    for(size_t i=1;i<=hltTrgUsedNames_.size();++i) {
      h2TrigCorr_->setBinLabel(i,hltTrgUsedNames_.at(i-1),1);
      h2TrigCorr_->setBinLabel(i,hltTrgUsedNames_.at(i-1),2);
    }
    TH1 *h = h2TrigCorr_->getTH1();
    if (h)
      h->SetStats(0);
  }

  if(1) {
    const int Nx = 5;
    const double x1 = 0;
    const double x2 = 5;
    book1D(hEvtSel_pTMax_,"hEvtSel_pTMax","Number of events at each stage of selection",Nx,x1,x2);
    book1D(hEvtSel_ChargedJet_,"hEvtSel_ChargedJet","Number of events at each stage of selection",Nx,x1,x2);   
    book1D(hEvtSel_CaloJet_,"hEvtSel_CaloJet","Number of events at each stage of selection",Nx,x1,x2);
    setLabel1D(hEvtSel_pTMax_);
    setLabel1D(hEvtSel_ChargedJet_);
    setLabel1D(hEvtSel_CaloJet_);
  }
 
  if (1) {
    const int Nx = 50;
    const double x1 = 0.0;
    const double x2 = 50.0;
    book1D(hNTrack500_,"hNTrack500","number of track with pT > 500 MeV;multiplicity",Nx,x1,x2);
 
  }

 if(1){
   const int Nz = 100;
   const double z1 = -20.0;
   const double z2 = 20.0;
   book1D(hVertex_z_,"hVertex_z","z position of vertex; z[cm]",Nz,z1,z2);
}

  if(1){
    const int Nx = 5;
    const double x1 = 0;
    const double x2 = 5;
    book1D(hNvertices_,"hNvertices","number of vertices",Nx,x1,x2);

  }


  if (1) {
    const int Nx = 25;
    const double x1 = 0.0;
    const double x2 = 50.0;
    book1D(hTrack_pTSpectrum_,"hTrack_pTSpectrum","pT spectrum of leading track;pT(GeV/c)",Nx,x1,x2);
    book1D(hCaloJet_pTSpectrum_,"hCalo_pTSpectrum","pT spectrum of leading calo jet;pT(GeV/c)",Nx,x1,x2);
    book1D(hChargedJet_pTSpectrum_,"hChargedJet_pTSpectrum","pT spectrum of leading track jet;pT(GeV/c)",Nx,x1,x2);
    
  }
  
  if (1) {
    const int Nx = 25;
    const double x1 = -3.2;
    const double x2 =  3.2;
    book1D(hTrack_phiSpectrum_,"hTrack_phiSpectrum","#phi spectrum of leading track;#phi",Nx,x1,x2);
    book1D(hCaloJet_phiSpectrum_,"hCaloJet_phiSpectrum","#phi spectrum of leading calo jet;#phi",Nx,x1,x2);
    book1D(hChargedJet_phiSpectrum_,"hChargedJet_phiSpectrum","#phi spectrum of leading track jet;#phi",Nx,x1,x2);

  }
  
  if (1) {
    const int Nx = 25;
    const double x1 = -5.;
    const double x2 =  5.;
    book1D(hTrack_etaSpectrum_,"hTrack_etaSpectrum","#eta spectrum of leading track;#eta",Nx,x1,x2);
    book1D(hCaloJet_etaSpectrum_,"hCaloJet_etaSpectrum","#eta spectrum of leading calo jet;#eta",Nx,x1,x2);
    book1D(hChargedJet_etaSpectrum_,"hChargedJet_etaSpectrum","#eta spectrum of leading track jet;#eta",Nx,x1,x2);

  }


if (1) {
    const int Nx = 20;
    const double x1 = 0.0;
    const double x2 = 20.0;
    const double y1 = 0.;
    const double y2 = 50.;
    bookProfile(hdNdEtadPhi_pTMax_Toward500_,"hdNdEtadPhi_pTMax_Toward500", 
                 "Average number of tracks (pT > 500 MeV) in toward region vs leading track pT;pT(GeV/c);dN/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
    bookProfile(hdNdEtadPhi_pTMax_Transverse500_,"hdNdEtadPhi_pTMax_Transverse500", 
                 "Average number of tracks (pT > 500 MeV) in transverse region vs leading track pT;pT(GeV/c);dN/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
    bookProfile(hdNdEtadPhi_pTMax_Away500_,"hdNdEtadPhi_pTMax_Away500", 
                 "Average number of tracks (pT > 500 MeV) in away region vs leading track pT;pT(GeV/c);dN/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
 
    bookProfile(hdNdEtadPhi_caloJet_Toward500_,"hdNdEtadPhi_caloJet_Toward500", 
                 "Average number of tracks (pT > 500 MeV) in toward region vs leading calo jet pT;pT(GeV/c);dN/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
    bookProfile(hdNdEtadPhi_caloJet_Transverse500_,"hdNdEtadPhi_caloJet_Transverse500", 
                 "Average number of tracks (pT > 500 MeV) in transverse region vs leading calo jet pT;pT(GeV/c);dN/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
    bookProfile(hdNdEtadPhi_caloJet_Away500_,"hdNdEtadPhi_caloJet_Away500", 
                 "Average number of tracks (pT > 500 MeV) in away region vs leading calo jet pT;pT(GeV/c);dN/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);   
  
    bookProfile(hdNdEtadPhi_trackJet_Toward500_,"hdNdEtadPhi_trackJet_Toward500", 
                 "Average number of tracks (pT > 500 MeV) in toward region vs leading track jet pT;pT(GeV/c);dN/d#eta d#phi",Nx,x1,x2,y1,y2);
    bookProfile(hdNdEtadPhi_trackJet_Transverse500_,"hdNdEtadPhi_trackJet_Transverse500", 
                 "Average number of tracks (pT > 500 MeV) in transverse region vs leading track jet pT;pT(GeV/c);dN/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
    bookProfile(hdNdEtadPhi_trackJet_Away500_,"hdNdEtadPhi_trackJet_Away500", 
                 "Average number of tracks (pT > 500 MeV) in away region vs leading track jet pT;pT(GeV/c);dN/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);


 
    bookProfile(hpTSumdEtadPhi_pTMax_Toward500_,"hpTSumdEtadPhi_pTMax_Toward500", 
                 "Average number of tracks (pT > 500 MeV) in toward region vs leading track pT;pT(GeV/c);dpTSum/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
    bookProfile(hpTSumdEtadPhi_pTMax_Transverse500_,"hpTSumdEtadPhi_pTMax_Transverse500", 
                 "Average number of tracks (pT > 500 MeV) in transverse region vs leading track pT;pT(GeV/c);dpTSum/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
    bookProfile(hpTSumdEtadPhi_pTMax_Away500_,"hpTSumdEtadPhi_pTMax_Away500", 
                 "Average number of tracks (pT > 500 MeV) in away region vs leading track pT;pT(GeV/c);dpTSum/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
 
    bookProfile(hpTSumdEtadPhi_caloJet_Toward500_,"hpTSumdEtadPhi_caloJet_Toward500", 
                 "Average number of tracks (pT > 500 MeV) in toward region vs leading calo jet pT;pT(GeV/c);dpTSum/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
    bookProfile(hpTSumdEtadPhi_caloJet_Transverse500_,"hpTSumdEtadPhi_caloJet_Transverse500", 
                 "Average number of tracks (pT > 500 MeV) in transverse region vs leading calo jet pT;pT(GeV/c);dpTSum/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
    bookProfile(hpTSumdEtadPhi_caloJet_Away500_,"hpTSumdEtadPhi_caloJet_Away500", 
                 "Average number of tracks (pT > 500 MeV) in away region vs leading calo jet pT;pT(GeV/c);dpTSum/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);   
  
    bookProfile(hpTSumdEtadPhi_trackJet_Toward500_,"hpTSumdEtadPhi_trackJet_Toward500", 
                 "Average number of tracks (pT > 500 MeV) in toward region vs leading track jet pT;pT(GeV/c);dpTSum/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
    bookProfile(hpTSumdEtadPhi_trackJet_Transverse500_,"hpTSumdEtadPhi_trackJet_Transverse500", 
                 "Average number of tracks (pT > 500 MeV) in transverse region vs leading track jet pT;pT(GeV/c);dpTSum/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
    bookProfile(hpTSumdEtadPhi_trackJet_Away500_,"hpTSumdEtadPhi_trackJet_Away500", 
                 "Average number of tracks (pT > 500 MeV) in away region vs leading track jet pT;pT(GeV/c);dpTSum/d#eta d#phi",Nx,x1,x2,y1,y2,0,0);
 
   
  }

if (1) {
    const int Nx = 20;
    const double x1 = 0.0;
    const double x2 = 20.0;

        book1D(hChargedJetMulti_,"hChargedJetMulti","Charged jet multiplicity;multiplicities",Nx,x1,x2);
        book1D(hChargedJetConstituent_,"hChargedJetConstituent","Charged Jet Constituent;number of constituents",Nx,x1,x2);
        book1D(hCaloJetMulti_,"hCaloJetMulti","Calo jet multiplicity;multiplicities",Nx,x1,x2);
        book1D(hCaloJetConstituent_,"hCaloJetConstituent","Calo Jet Constituent;number of constituents",Nx,x1,x2);

  }


if (1) {
    const int Nx = 60;
    const double x1 = -180.0;
    const double x2 = 180.0;

        book1D(hdPhi_maxpTTrack_tracks_,"hdPhi_maxpTTrack_tracks","delta phi between leading tracks and other tracks;#Delta#phi(leading track-track)",Nx,x1,x2);
        book1D(hdPhi_caloJet_tracks_,"hdPhi_caloJet_tracks","delta phi between leading calo jet  and tracks;#Delta#phi(leading calo jet-track)",Nx,x1,x2);
        book1D(hdPhi_chargedJet_tracks_,"hdPhi_chargedJet_tracks","delta phi between leading charged jet  and tracks;#Delta#phi(leading charged jet-track)",Nx,x1,x2);

}
            

}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::endJob(void) 
{
}

//--------------------------------------------------------------------------------------------------

void QcdUeDQM::endLuminosityBlock(const LuminosityBlock &l, 
                                     const EventSetup &iSetup)
{
  // Update various histograms.

  repSummary_->Fill(1.);
  repSumMap_->Fill(0.5,0.5,1.);

  
}

//--------------------------------------------------------------------------------------------------

void QcdUeDQM::endRun(const Run &, const EventSetup &)
{
  // End run, cleanup. TODO: can this be called several times in DQM???

}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::fill1D(std::vector<TH1F*> &hs, double val, double w)
{
  // Loop over histograms and fill if trigger has fired.

  for(size_t i=0;i<hs.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    hs.at(i)->Fill(val,w);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::fill1D(std::vector<MonitorElement*> &mes, double val, double w)
{
  // Loop over histograms and fill if trigger has fired.

  for(size_t i=0;i<mes.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    mes.at(i)->Fill(val,w);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::setLabel1D(std::vector<MonitorElement*> &mes)
{
  // Loop over histograms and fill if trigger has fired.
  string cut[5] = {"Nevt","vtx!=bmspt","Zvtx<10cm","pT>1GeV","trackFromVtx"};
  for(size_t i=0;i<mes.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    for(size_t j = 1;j < 6;j++)mes.at(i)->setBinLabel(j,cut[j-1],1);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::fill2D(std::vector<TH2F*> &hs, double valx, double valy, double w)
{
  // Loop over histograms and fill if trigger has fired.

  for(size_t i=0;i<hs.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    hs.at(i)->Fill(valx, valy ,w);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::fill2D(std::vector<MonitorElement*> &mes, double valx, double valy, double w)
{
  // Loop over histograms and fill if trigger has fired.

  for(size_t i=0;i<mes.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    mes.at(i)->Fill(valx, valy ,w);
  }
}
//--------------------------------------------------------------------------------------------------
void QcdUeDQM::fillProfile(std::vector<TProfile*> &hs, double valx, double valy, double w)
{
  // Loop over histograms and fill if trigger has fired.

  for(size_t i=0;i<hs.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
    hs.at(i)->Fill(valx, valy ,w);
  }
}

//--------------------------------------------------------------------------------------------------
void QcdUeDQM::fillProfile(std::vector<MonitorElement*> &mes, double valx, double valy, double w)
{
  // Loop over histograms and fill if trigger has fired.

  for(size_t i=0;i<mes.size();++i) {
    if (!hltTrgDeci_.at(i))
      continue;
   const double y = valy*w; 
    mes.at(i)->Fill(valx, y);
  }
}

//--------------------------------------------------------------------------------------------------
bool QcdUeDQM::fillVtxPlots(const edm::Event &iEvent, const edm::Handle< reco::VertexCollection > vtxColl)
{
  bool vtxFound = false;
  const reco::VertexCollection theVertices = *(vtxColl.product());
  fill1D(hNvertices_,theVertices.size()); 
  //  if (theVertices.size() > 0){
    for (reco::VertexCollection::const_iterator vertexIt = theVertices.begin(); vertexIt != theVertices.end(); ++vertexIt) 
      {
	fill1D(hVertex_z_,vertexIt->z());
        if(fabs(vertexIt->z()) < 10)vtxFound = true; 
      
      } // Loop over vertcies
    //  }//At least one vertex
  if(vtxFound)return true;
  return false;
 
}
//--------------------------------------------------------------------------------------------------
void QcdUeDQM::fillpTMaxRelated(const edm::Event &iEvent, const edm::Handle<reco::TrackCollection> &track, const edm::Handle< reco::VertexCollection > vtxColl)
       {
const double beamspot=0.110321;
int Ntrack500 = 0;

 const reco::VertexCollection theVertices = *(vtxColl.product());
   for (reco::VertexCollection::const_iterator vertexIt = theVertices.begin(); vertexIt != theVertices.end(); ++vertexIt) 
	   {
	    
	     if(vertexIt->z()!=beamspot)
	       {   // Vertex is not beam spot
              
		 if(fabs(vertexIt->z()) < 10.)
		   { // vertex z position cut
		    
		     for(reco::TrackCollection::const_iterator trk = track->begin(); trk != track->end(); ++trk)
		       {
                        if(trk->pt() > 0.5)Ntrack500++;                  
			 if(trk == track->begin() && fabs(vertexIt->z() - trk->vz()) < 1.)
			   {
			     fill1D(hTrack_pTSpectrum_,trk->pt());
			     fill1D(hTrack_phiSpectrum_,trk->phi());
			     fill1D(hTrack_etaSpectrum_,trk->eta()); 
			   }	     
		       }
		     
		     
		   }// vertex z position cut
		 
	       }// vertex is not beam spot
	     break;  // Look for only one vertex of interest
	   }// Loop over vertices
fill1D(hNTrack500_,Ntrack500);
 }
void QcdUeDQM::fillChargedJetSpectra(const edm::Event &iEvent, const edm::Handle<reco::CandidateView> trackJets)
{
  fill1D(hChargedJetMulti_,trackJets->size());
  for( reco::CandidateView::const_iterator f  = trackJets->begin();  f != trackJets->end(); f++) 
    {
      if(f != trackJets->begin())continue;
      fill1D(hChargedJet_pTSpectrum_,f->pt());
      fill1D(hChargedJet_etaSpectrum_,f->eta());
      fill1D(hChargedJet_phiSpectrum_,f->phi());
      int nConst = 0;
      for( reco::Candidate::const_iterator c  = f->begin(); c != f->end(); c ++)
	{  
	  nConst++;  
	}
      fill1D(hChargedJetConstituent_,nConst);
    } 
	
}

void QcdUeDQM::fillCaloJetSpectra(const edm::Event &iEvent, const edm::Handle<reco::CaloJetCollection> caloJets)
{
  fill1D(hCaloJetMulti_,caloJets->size());
   for( reco::CaloJetCollection::const_iterator f  = caloJets->begin();  f != caloJets->end(); f++)
     {
       if(f != caloJets->begin())continue;
       fill1D(hCaloJet_pTSpectrum_,f->pt()); 
       fill1D(hCaloJetConstituent_,f->nConstituents());
       fill1D(hCaloJet_etaSpectrum_,f->eta());
       fill1D(hCaloJet_phiSpectrum_,f->phi());
     }
   
}

/*
 weight for transverse/toward/away region = 0.12
 

*/

void QcdUeDQM::fillUE_with_MaxpTtrack(const edm::Event &iEvent, const reco::TrackCollection &track, const edm::Handle< reco::VertexCollection > vtxColl)
{
double w = 0.119;          
//double w = 1.;
const double beamspot=0.110321;
double nTrk500_TransReg = 0;
double nTrk500_AwayReg = 0;
double nTrk500_TowardReg = 0;
 
double pTSum500_TransReg = 0;
double pTSum500_AwayReg = 0;
double pTSum500_TowardReg = 0;
double nevt = 0.0; 	 
fill1D(hEvtSel_pTMax_,nevt);
 const reco::VertexCollection theVertices = *(vtxColl.product());
   for (reco::VertexCollection::const_iterator vertexIt = theVertices.begin(); vertexIt != theVertices.end(); ++vertexIt) 
     {
       nevt = nevt + 1.0;
       fill1D(hEvtSel_pTMax_,nevt);
       if(vertexIt->z()!=beamspot)
	 {   // Vertex is not beam spot
          nevt = nevt + 1.0;
          fill1D(hEvtSel_pTMax_,nevt);
	   if(fabs(vertexIt->z()) < 10.)
	     { // vertex z posiion cut
              nevt = nevt + 1.0;
              fill1D(hEvtSel_pTMax_,nevt);
              if(track[0].pt() > 1.)
               {
               nevt = nevt + 1.0;
               fill1D(hEvtSel_pTMax_,nevt);
	       for(size_t i = 1; i < track.size();i++)
		 {
		   if(fabs(vertexIt->z() - track[i].vz()) < 1.)
		     { 
		       double dphi = (180./PI)*(deltaPhi(track[0].phi(),track[i].phi()));
		       fill1D(hdPhi_maxpTTrack_tracks_,dphi);
                        
                       
		       if(fabs(dphi)>60. && fabs(dphi)<120.)
			 {
			   if(track[i].pt() > 0.5 && fabs(track[i].eta()) < 2.)
			     {
			       pTSum500_TransReg =  pTSum500_TransReg + track[i].pt();     
			       nTrk500_TransReg++;
			     }  
			 }            
			
		       if(fabs(dphi)>120. && fabs(dphi)<180.)
			 {
			   if(track[i].pt() > 0.5 && fabs(track[i].eta()) < 2.)
			     {
			       pTSum500_AwayReg =  pTSum500_AwayReg + track[i].pt();   
			       nTrk500_AwayReg++;
			     }
			 } 
		       
		       if(fabs(dphi)<60.)
			 {
			   if(track[i].pt() > 0.5 && fabs(track[i].eta()) < 2.)
			     {
			       pTSum500_TowardReg =  pTSum500_TowardReg + track[i].pt();
			       nTrk500_TowardReg++;
			     }
			 }           
		       
		     } // track from vertex
		 }// Loop over tracks
               fillProfile(hdNdEtadPhi_pTMax_Toward500_, track[0].pt(),nTrk500_TowardReg,w);
               fillProfile(hdNdEtadPhi_pTMax_Transverse500_, track[0].pt(),nTrk500_TransReg,w);
               fillProfile(hdNdEtadPhi_pTMax_Away500_, track[0].pt(),nTrk500_AwayReg,w);

               fillProfile(hpTSumdEtadPhi_pTMax_Toward500_,track[0].pt() ,pTSum500_TowardReg,w);
               fillProfile(hpTSumdEtadPhi_pTMax_Transverse500_,track[0].pt(),pTSum500_TransReg,w);
               fillProfile(hpTSumdEtadPhi_pTMax_Away500_, track[0].pt(),pTSum500_AwayReg,w);
//	        break;  // Look for only one vertex of interest 
	      }// pT cut on leading track
	     }// vertex z position cut
	 }// vertex is not beam spot
       break;  // Look for only one vertex of interest
     }// Loop over vertices
}

void QcdUeDQM::fillUE_with_ChargedJets(const edm::Event &iEvent, const reco::TrackCollection &track, const edm::Handle<reco::CandidateView> &trackJets, const edm::Handle< reco::VertexCollection > vtxColl)
{
double w = 0.119;
//double w = 1.;
const double beamspot=0.110321; 
double nTrk500_TransReg = 0;
double nTrk500_AwayReg = 0;
double nTrk500_TowardReg = 0;
  
double pTSum500_TransReg = 0;
double pTSum500_AwayReg = 0;
double pTSum500_TowardReg = 0;
double nevt = 0.0; 
fill1D(hEvtSel_ChargedJet_,nevt);
   const reco::VertexCollection theVertices = *(vtxColl.product());
     for (reco::VertexCollection::const_iterator vertexIt = theVertices.begin(); vertexIt != theVertices.end(); ++vertexIt) 
       {
        nevt = nevt + 1.0;
       fill1D(hEvtSel_ChargedJet_,nevt);
	 if(vertexIt->z()!=beamspot)
	   {   // Vertex is not beam spot
             nevt = nevt + 1.0;
             fill1D(hEvtSel_ChargedJet_,nevt);  
	     if(fabs(vertexIt->z()) < 10)
	       { // vertex z posiion cut
                nevt = nevt + 1.0;
                fill1D(hEvtSel_ChargedJet_,nevt);   
                if(!(trackJets->empty()) && (trackJets->begin())->pt() > 1.)
                 {
                 double jetPhi = (trackJets->begin())->phi();
                 nevt = nevt + 1.0;
                 fill1D(hEvtSel_ChargedJet_,nevt);
		 for(size_t i = 0; i < track.size();i++)
		   {
		     if(fabs(vertexIt->z() - track[i].vz()) < 1.)
		       {
			 double dphi = (180./PI)*(deltaPhi(jetPhi,track[i].phi()));
			 fill1D(hdPhi_chargedJet_tracks_,dphi);
			 if(fabs(dphi)>60. && fabs(dphi)<120.)
			   {
			     if(track[i].pt() > 0.5 && fabs(track[i].eta()) < 2.)
			       {
				 pTSum500_TransReg =  pTSum500_TransReg + track[i].pt();
				 nTrk500_TransReg++;
			       }
			    
			   }
			
			 if(fabs(dphi)>120. && fabs(dphi)<180.)
			   {
			     if(track[i].pt() > 0.5 && fabs(track[i].eta()) < 2.)
			       {
				 pTSum500_AwayReg =  pTSum500_AwayReg + track[i].pt();
				 nTrk500_AwayReg++;
			       }
			   }
			 if(fabs(dphi)<60.)
			   {
			     if(track[i].pt() > 0.5 && fabs(track[i].eta()) < 2.)
			       {
				 pTSum500_TowardReg =  pTSum500_TowardReg + track[i].pt();
				 nTrk500_TowardReg++;
			       }
			   }
		       }// tracks from vertex

		   }// Track Loop
		 
		 fillProfile(hdNdEtadPhi_trackJet_Toward500_, (trackJets->begin())->pt(),nTrk500_TowardReg,w);
		 fillProfile(hdNdEtadPhi_trackJet_Transverse500_, (trackJets->begin())->pt(),nTrk500_TransReg,w);
		 fillProfile(hdNdEtadPhi_trackJet_Away500_, (trackJets->begin())->pt(),nTrk500_AwayReg,w);
		 
		 fillProfile(hpTSumdEtadPhi_trackJet_Toward500_, (trackJets->begin())->pt(),pTSum500_TowardReg,w);
		 fillProfile(hpTSumdEtadPhi_trackJet_Transverse500_, (trackJets->begin())->pt(),pTSum500_TransReg,w);
		 fillProfile(hpTSumdEtadPhi_trackJet_Away500_, (trackJets->begin())->pt(),pTSum500_AwayReg,w);
//                 break;  // Look for only one vertex of interest
		 }//pT cut on leading charged jet
	       }// vertex z position cut
	   }// vertex is not beam spot
	 break;  // Look for only one vertex of interest    
       }// Loop over vertices
     
		
 
}                                                                                       

void QcdUeDQM:: fillUE_with_CaloJets(const edm::Event &iEvent, const reco::TrackCollection &track, const edm::Handle<reco::CaloJetCollection> &caloJets, const edm::Handle< reco::VertexCollection > vtxColl)
{
const double beamspot=0.110321;
double w = 0.119;
//double w = 1.;
double nTrk500_TransReg = 0;
double nTrk500_AwayReg = 0;
double nTrk500_TowardReg = 0;
                 
double pTSum500_TransReg = 0;
double pTSum500_AwayReg = 0;
double pTSum500_TowardReg = 0;
double nevt = 0.0;
fill1D(hEvtSel_CaloJet_,nevt);
    
  

     const reco::VertexCollection theVertices = *(vtxColl.product());
     for (reco::VertexCollection::const_iterator vertexIt = theVertices.begin(); vertexIt != theVertices.end(); ++vertexIt) 
       {
          nevt = nevt + 1.0;
          fill1D(hEvtSel_CaloJet_,nevt);
	 if(vertexIt->z()!=beamspot)
	   {   // Vertex is not beam spot
            nevt = nevt + 1.0;
            fill1D(hEvtSel_CaloJet_,nevt);
	     if(fabs(vertexIt->z()) < 10)
	       { // vertex z posiion cut
                nevt = nevt + 1.0;
                fill1D(hEvtSel_CaloJet_,nevt);
                if(!(caloJets->empty()) && (caloJets->begin())->pt() > 1.)
                 {
                 double jetPhi = (caloJets->begin())->phi();
                 nevt = nevt + 1.0;
                 fill1D(hEvtSel_CaloJet_,nevt);
		 for(size_t i = 0; i < track.size();i++)
		   {
		     if(fabs(vertexIt->z() - track[i].vz()) < 1.)
		       {
			 double dphi = (180./PI)*(deltaPhi(jetPhi,track[i].phi()));
			 fill1D(hdPhi_caloJet_tracks_,dphi);
			 if(fabs(dphi)>60. && fabs(dphi)<120.)
			   {
			     
			     if(track[i].pt() > 0.5 && fabs(track[i].eta()) < 2.)
			       {
				 pTSum500_TransReg =  pTSum500_TransReg + track[i].pt();
				 nTrk500_TransReg++;
			       }
			   }
			 if(fabs(dphi)>120. && fabs(dphi)<180.)
			   {
			     if(track[i].pt() > 0.5 && fabs(track[i].eta()) < 2.)
			       {
				 pTSum500_AwayReg =  pTSum500_AwayReg + track[i].pt();
				 nTrk500_AwayReg++;
			       }
			   }
			 if(fabs(dphi)<60.)
			   {
			     if(track[i].pt() > 0.5 && fabs(track[i].eta()) < 2.)
			       {
				 pTSum500_TowardReg =  pTSum500_TowardReg + track[i].pt();
				 nTrk500_TowardReg++;
			       }
			   
			   }
		       }// tracks from vertex

		   }// Loop over tracks
		 fillProfile(hdNdEtadPhi_caloJet_Toward500_, (caloJets->begin())->pt(),nTrk500_TowardReg,w);
		 fillProfile(hdNdEtadPhi_caloJet_Transverse500_, (caloJets->begin())->pt(),nTrk500_TransReg,w);
		 fillProfile(hdNdEtadPhi_caloJet_Away500_, (caloJets->begin())->pt(),nTrk500_AwayReg,w);
		   
		 fillProfile(hpTSumdEtadPhi_caloJet_Toward500_, (caloJets->begin())->pt(),pTSum500_TowardReg,w);
		 fillProfile(hpTSumdEtadPhi_caloJet_Transverse500_, (caloJets->begin())->pt(),pTSum500_TransReg,w);
		 fillProfile(hpTSumdEtadPhi_caloJet_Away500_, (caloJets->begin())->pt(),pTSum500_AwayReg,w);
//                 break;  // Look for only one vertex of interest
                } //pT cut on leading calo jet
	       }// vertex z position cut
	   }// vertex is not beam spot
         break;  // Look for only one vertex of interest	    
       }// Loop over vertices
       

}

void QcdUeDQM::fillHltBits(const Event &iEvent)
{
  // Fill HLT trigger bits.

  Handle<TriggerResults> triggerResultsHLT;
  getProduct(hltUsedResName_, triggerResultsHLT, iEvent);

  for(size_t i=0;i<hltTrgBits_.size();++i) {
    if (hltTrgBits_.at(i)<0) 
      continue; //ignore unknown trigger 
    size_t tbit = hltTrgBits_.at(i);
    if (tbit<triggerResultsHLT->size()) {
      hltTrgDeci_[i] = triggerResultsHLT->accept(tbit);
  }
 }
  // fill correlation histogram
  for(size_t i=0;i<hltTrgBits_.size();++i) {
    if (hltTrgDeci_.at(i))
      h2TrigCorr_->Fill(i,i);
    for(size_t j=i+1;j<hltTrgBits_.size();++j) {
      if (hltTrgDeci_.at(i) && hltTrgDeci_.at(j))
        h2TrigCorr_->Fill(i,j);
      if (hltTrgDeci_.at(i) && !hltTrgDeci_.at(j))
        h2TrigCorr_->Fill(j,i);
    }
  }
}



//--------------------------------------------------------------------------------------------------

