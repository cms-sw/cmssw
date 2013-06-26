#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/ParticleFlowReco/interface/PreIdFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "DataFormats/Common/interface/ValueMap.h"

class PreIdAnalyzer : public edm::EDAnalyzer {

 public:
  explicit PreIdAnalyzer(const edm::ParameterSet &);
  ~PreIdAnalyzer();

  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  virtual void beginRun(edm::Run const&, edm::EventSetup const& );
  //  virtual void beginJobAnalyze(const edm::EventSetup & c);
  virtual void endRun();
 private:
  edm::InputTag PreIdMapLabel_;
  edm::InputTag TrackLabel_;

  DQMStore * dbe;
  MonitorElement * TracksPt;
  MonitorElement * TracksEta; 
  MonitorElement * TracksPtEcalMatch;
  MonitorElement * TracksEtaEcalMatch; 
  MonitorElement * geomMatchChi2; 
  MonitorElement * geomMatchEop; 

  MonitorElement * TracksChi2;
  MonitorElement * TracksNhits;
  MonitorElement * TracksPtFiltered;
  MonitorElement * TracksEtaFiltered; 
  MonitorElement * TracksPtNotFiltered;
  MonitorElement * TracksEtaNotFiltered; 

  MonitorElement * TracksPtPreIded;
  MonitorElement * TracksEtaPreIded; 
  MonitorElement * trackdpt;
  MonitorElement * gsfChi2;  
  MonitorElement * chi2Ratio;  
  MonitorElement * mva;  

};

PreIdAnalyzer::PreIdAnalyzer(const edm::ParameterSet &pset) {
  PreIdMapLabel_ = pset.getParameter<edm::InputTag>("PreIdMap");
  TrackLabel_ = pset.getParameter<edm::InputTag>("TrackCollection");
}

PreIdAnalyzer::~PreIdAnalyzer() {
  dbe->save("PreId.root"); 
  ;}

void PreIdAnalyzer::beginRun(edm::Run const & run, edm::EventSetup const& es) {
  dbe = edm::Service<DQMStore>().operator->();
  //}

  //void  PreIdAnalyzer::beginJobAnalyze(const edm::EventSetup & c){
  TracksPt = dbe->book1D("TracksPt","pT",1000,0,100.);
  TracksEta = dbe->book1D("TracksEta","eta",50,-2.5,2.5);
  TracksPtEcalMatch = dbe->book1D("TracksPtEcalMatch","pT",1000,0,100.);
  TracksEtaEcalMatch = dbe->book1D("TracksEtaEcalMatch","eta",50,-2.5,2.5);
  TracksPtFiltered = dbe->book1D("TracksPtFiltered","pT",1000,0,100.);
  TracksEtaFiltered = dbe->book1D("TracksEtaFiltered","eta",50,-2.5,2.5);
  TracksPtNotFiltered = dbe->book1D("TracksPtNotFiltered","pT",1000,0,100.);
  TracksEtaNotFiltered = dbe->book1D("TracksEtaNotFiltered","eta",50,-2.5,2.5);
  TracksPtPreIded = dbe->book1D("TracksPtPreIded","pT",1000,0,100.);
  TracksEtaPreIded = dbe->book1D("TracksEtaPreIded","eta",50,-2.5,2.5);
  TracksChi2 = dbe->book1D("TracksChi2","chi2",100,0,10.);
  TracksNhits = dbe->book1D("TracksNhits","Nhits",30,-0.5,29.5);

  geomMatchChi2 = dbe->book1D("geomMatchChi2","Geom Chi2",100,0.,50.);
  geomMatchEop = dbe->book1D("geomMatchEop","E/p",100,0.,5.);
  trackdpt = dbe->book1D("trackdpt","dpt/pt",100,0.,5.);
  gsfChi2 = dbe->book1D("gsfChi2","GSF chi2",100,0.,10.);
  chi2Ratio = dbe->book1D("chi2Ratio","Chi2 ratio",100,0.,10.);
  mva = dbe->book1D("mva","mva",100,-1.,1.);
}

void PreIdAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<reco::TrackCollection> trackh;
  iEvent.getByLabel(TrackLabel_,trackh);
  edm::Handle<edm::ValueMap<reco::PreIdRef> > vmaph;
  iEvent.getByLabel(PreIdMapLabel_,vmaph);
  
  const reco::TrackCollection&  tracks=*(trackh.product());
  const edm::ValueMap<reco::PreIdRef> & preidMap = *(vmaph.product());

  unsigned ntracks=tracks.size();
  for(unsigned itrack=0;itrack<ntracks;++itrack)
    {
      reco::TrackRef theTrackRef(trackh,itrack);
      TracksPt->Fill(theTrackRef->pt());
      TracksEta->Fill(theTrackRef->eta());
      
      if(preidMap[theTrackRef].isNull())
	continue;
      
      const reco::PreId & myPreId(*(preidMap[theTrackRef]));
      geomMatchChi2->Fill(myPreId.geomMatching()[4]);
      geomMatchEop->Fill(myPreId.eopMatch());
      
      if(myPreId.ecalMatching())
	{
	  TracksPtEcalMatch->Fill(theTrackRef->pt());
	  TracksEtaEcalMatch->Fill(theTrackRef->eta());
	}
      else
	{
	  TracksChi2->Fill(myPreId.kfChi2());
	  TracksNhits->Fill(myPreId.kfNHits());
	  if(myPreId.trackFiltered())
	    {
	      TracksPtFiltered->Fill(theTrackRef->pt());
	      TracksEtaFiltered->Fill(theTrackRef->eta());
	      trackdpt->Fill(myPreId.dpt());
	      gsfChi2->Fill(myPreId.gsfChi2());
	      chi2Ratio->Fill(myPreId.chi2Ratio());
	      mva->Fill(myPreId.mva());
	    }
	  else
	    {
	      TracksPtNotFiltered->Fill(theTrackRef->pt());
	      TracksEtaNotFiltered->Fill(theTrackRef->eta());
	    }
	}
      if(myPreId.preIded())
	{
	  TracksPtPreIded->Fill(theTrackRef->pt());
	  TracksEtaPreIded->Fill(theTrackRef->eta());
	}
    }
}

void PreIdAnalyzer::endRun() {;}


DEFINE_FWK_MODULE(PreIdAnalyzer);


