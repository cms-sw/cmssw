// -*- C++ -*-
//
// Package:    TrackCount
// Class:      TrackCount
// 
/**\class TrackCount TrackCount.cc trackCount/TrackCount/src/TrackCount.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Thu Sep 25 16:32:56 CEST 2008
// $Id: TrackCount.cc,v 1.15 2011/11/15 10:09:24 venturia Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// my includes

#include <iostream>
#include <map>
#include <string>
#include <numeric>

#include "TH1F.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TProfile2D.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"

#include "DataFormats/Luminosity/interface/LumiDetails.h"
//
// class decleration
//

class TrackCount : public edm::EDAnalyzer {
public:
  explicit TrackCount(const edm::ParameterSet&);
  ~TrackCount();
  
  
private:
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  
      // ----------member data ---------------------------

  TH1F* m_ntrk;
  TProfile* m_ntrkvslumi;
  TH2D* m_ntrkvslumi2D;
  TH1F* m_nhptrk;
  TH1F* m_hhpfrac;
  TH1F* m_hsqsumptsq;
  TH1F* m_hphi;
  TH1F* m_heta;
  TH1F* m_hcos;
  TH1F* m_hpt;
  TH1F* m_chisq;
  TH1F* m_chisqnorm;
  TH1F* m_ndof;
  TH2F* m_chisqvseta;
  TH2F* m_chisqnormvseta;
  TH2F* m_ndofvseta;
  TProfile2D* m_hptphieta;
  TH1D* m_hnlosthits;
  TH1D* m_hnrhits;
  TH1D* m_hnpixelrhits;
  TH1D* m_hnstriprhits;
  TH1D* m_hnlostlayers;
  TH1D* m_hnlayers;
  TH1D* m_hnpixellayers;
  TH1D* m_hnstriplayers;
  TH1D* m_halgo;
  TH2D* m_hphieta;
  TProfile2D* m_hnhitphieta;
  TProfile2D* m_hnlayerphieta;

  TProfile** m_ntrkvsorbrun;


  //  std::map<int,int> m_multiplicity;
  RunHistogramManager m_rhm;
  const unsigned int m_maxLS;
  const unsigned int m_LSfrac;
  const bool m_2dhistos;
  const bool m_runHisto;
  const bool m_dump;
  edm::EDGetTokenT<reco::TrackCollection> m_trkcollToken;
  edm::EDGetTokenT<LumiDetails> m_lumiProducerToken;
  const unsigned int m_nptbin;
  const double m_ptmin;
  const double m_ptmax;
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
TrackCount::TrackCount(const edm::ParameterSet& iConfig):
  m_rhm(consumesCollector()),
  m_maxLS(100),m_LSfrac(16),
  m_2dhistos(iConfig.getUntrackedParameter<bool>("wanted2DHistos",false)),
  m_runHisto(iConfig.getUntrackedParameter<bool>("runHisto",false)),
  m_dump(iConfig.getUntrackedParameter<bool>("dumpTracks",false)),
  m_trkcollToken(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trackCollection"))),
  m_lumiProducerToken(consumes<LumiDetails,edm::InLumi>(edm::InputTag("lumiProducer"))),
  m_nptbin(iConfig.getUntrackedParameter<unsigned int>("numberPtBins",200)),
  m_ptmin(iConfig.getUntrackedParameter<double>("ptMin",0.)),
  m_ptmax(iConfig.getUntrackedParameter<double>("ptMax",20.))


{
   //now do what ever initialization is needed

  // histogram parameters

  const unsigned int netabin1d = iConfig.getUntrackedParameter<unsigned int>("netabin1D",120);
  const unsigned int netabin2d = iConfig.getUntrackedParameter<unsigned int>("netabin2D",40);
  const float etamin = iConfig.getUntrackedParameter<double>("etaMin",-3.);
  const float etamax = iConfig.getUntrackedParameter<double>("etaMax",3.);
  const unsigned int nchisqbin1d = iConfig.getUntrackedParameter<unsigned int>("nchi2bin1D",50);
  const unsigned int nndofbin1d = iConfig.getUntrackedParameter<unsigned int>("nndofbin1D",50);
  const unsigned int nchisqbin2d = iConfig.getUntrackedParameter<unsigned int>("nchi2bin2D",50);
  const unsigned int nndofbin2d = iConfig.getUntrackedParameter<unsigned int>("nndofbin2D",50);

  edm::LogInfo("TrackCollection") << "Using collection " << iConfig.getParameter<edm::InputTag>("trackCollection").label().c_str() ;


  edm::Service<TFileService> tfserv;

  m_ntrk = tfserv->make<TH1F>("ntrk","Number of Tracks",2500,-0.5,2499.5);
  m_ntrk->GetXaxis()->SetTitle("Tracks");   m_ntrk->GetYaxis()->SetTitle("Events"); 
  m_ntrkvslumi = tfserv->make<TProfile>("ntrkvslumi","Number of Tracks vs Luminosity",250,0.,10.);
  m_ntrkvslumi->GetXaxis()->SetTitle("BX lumi [10^{30}cm^{-2}s^{-1}]");  m_ntrkvslumi->GetYaxis()->SetTitle("Ntracks");
  m_ntrkvslumi2D = tfserv->make<TH2D>("ntrkvslumi2D","Number of Tracks vs Luminosity",80,0.,10.,125,-0.5,2499.5);
  m_ntrkvslumi2D->GetXaxis()->SetTitle("BX lumi [10^{30}cm^{-2}s^{-1}]");  m_ntrkvslumi2D->GetYaxis()->SetTitle("Ntracks");

  m_nhptrk = tfserv->make<TH1F>("nhptrk","Number of High Purity Tracks",2500,-0.5,2499.5);
  m_nhptrk->GetXaxis()->SetTitle("Tracks");   m_nhptrk->GetYaxis()->SetTitle("Events"); 
  m_hhpfrac = tfserv->make<TH1F>("hhpfrac","Fraction of High Purity Tracks",51,0.,1.02);
  m_hhpfrac->GetXaxis()->SetTitle("hp/all");   m_hhpfrac->GetYaxis()->SetTitle("Events"); 
  m_hsqsumptsq = tfserv->make<TH1F>("hsqsumptsq","Sqrt(Sum pt**2)",1000,0.,200.);
  m_hsqsumptsq->GetXaxis()->SetTitle("#sqrt(#Sigma pt^2) (GeV)");   m_hsqsumptsq->GetYaxis()->SetTitle("Events"); 
  
  m_hphi = tfserv->make<TH1F>("phi","Track azimuth",40,-M_PI,M_PI);
  m_hphi->GetXaxis()->SetTitle("#phi (rad)");   m_hphi->GetYaxis()->SetTitle("Tracks"); 
  m_heta = tfserv->make<TH1F>("eta","Track pseudorapidity",netabin1d,etamin,etamax);
  m_heta->GetXaxis()->SetTitle("#eta");   m_heta->GetYaxis()->SetTitle("Tracks"); 
  m_hcos = tfserv->make<TH1F>("cos","Track polar angle",50,-1.,1.);
  m_hcos->GetXaxis()->SetTitle("cos(#theta)");   m_hcos->GetYaxis()->SetTitle("Tracks"); 
  m_hpt = tfserv->make<TH1F>("pt","Track pt",m_nptbin,m_ptmin,m_ptmax);
  m_hpt->GetXaxis()->SetTitle("pt (GeV)");   m_hpt->GetYaxis()->SetTitle("Tracks"); 

  m_chisq = tfserv->make<TH1F>("chisq","Track Chi2",nchisqbin1d,0.,100.);
  m_chisq->GetXaxis()->SetTitle("chi2");  m_chisq->GetYaxis()->SetTitle("Tracks");
  m_chisqnorm = tfserv->make<TH1F>("chisqnorm","Track normalized Chi2",nchisqbin1d,0.,10.);
  m_chisqnorm->GetXaxis()->SetTitle("normalized chi2");  m_chisqnorm->GetYaxis()->SetTitle("Tracks");
  m_ndof = tfserv->make<TH1F>("ndof","Track ndof",nndofbin1d,0.,100.);
  m_ndof->GetXaxis()->SetTitle("ndof");  m_ndof->GetYaxis()->SetTitle("Tracks");
  if(m_2dhistos) {
    m_chisqvseta = tfserv->make<TH2F>("chisqvseta","Track Chi2 vs #eta",netabin2d,etamin,etamax,nchisqbin2d,0.,100.);
    m_chisqvseta->GetXaxis()->SetTitle("#eta");  m_chisqvseta->GetYaxis()->SetTitle("#chi2");
    m_chisqnormvseta = tfserv->make<TH2F>("chisqnormvseta","Track normalized Chi2 vs #eta",netabin2d,etamin,etamax,nchisqbin2d,0.,10.);
    m_chisqnormvseta->GetXaxis()->SetTitle("#eta");  m_chisqnormvseta->GetYaxis()->SetTitle("normalized #chi2");
    m_ndofvseta = tfserv->make<TH2F>("ndofvseta","Track ndof vs #eta",netabin2d,etamin,etamax,nndofbin2d,0.,100.);
    m_ndofvseta->GetXaxis()->SetTitle("#eta");  m_ndofvseta->GetYaxis()->SetTitle("ndof");
  }

  m_hptphieta = tfserv->make<TProfile2D>("ptphivseta","Average pt vs #phi vs #eta",netabin2d,etamin,etamax,40,-M_PI,M_PI);
  m_hptphieta->GetXaxis()->SetTitle("#eta");   m_hptphieta->GetYaxis()->SetTitle("#phi"); 

  

  m_hnlosthits = tfserv->make<TH1D>("nlosthits","Number of Lost Hits",10,-0.5,9.5);
  m_hnlosthits->GetXaxis()->SetTitle("Nlost");   m_hnlosthits->GetYaxis()->SetTitle("Tracks"); 

  m_hnrhits = tfserv->make<TH1D>("nrhits","Number of Valid Hits",55,-0.5,54.5);
  m_hnrhits->GetXaxis()->SetTitle("Nvalid");   m_hnrhits->GetYaxis()->SetTitle("Tracks"); 
  m_hnpixelrhits = tfserv->make<TH1D>("npixelrhits","Number of Valid Pixel Hits",20,-0.5,19.5);
  m_hnpixelrhits->GetXaxis()->SetTitle("Nvalid");   m_hnpixelrhits->GetYaxis()->SetTitle("Tracks"); 
  m_hnstriprhits = tfserv->make<TH1D>("nstriprhits","Number of Valid Strip Hits",45,-0.5,44.5);
  m_hnstriprhits->GetXaxis()->SetTitle("Nvalid");   m_hnstriprhits->GetYaxis()->SetTitle("Tracks"); 

  m_hnlostlayers = tfserv->make<TH1D>("nlostlayers","Number of Layers w/o measurement",10,-0.5,9.5);
  m_hnlostlayers->GetXaxis()->SetTitle("Nlostlay");   m_hnlostlayers->GetYaxis()->SetTitle("Tracks"); 

  m_hnlayers = tfserv->make<TH1D>("nlayers","Number of Layers",20,-0.5,19.5);
  m_hnlayers->GetXaxis()->SetTitle("Nlayers");   m_hnlayers->GetYaxis()->SetTitle("Tracks"); 
  m_hnpixellayers = tfserv->make<TH1D>("npixellayers","Number of Pixel Layers",10,-0.5,9.5);
  m_hnpixellayers->GetXaxis()->SetTitle("Nlayers");   m_hnpixellayers->GetYaxis()->SetTitle("Tracks"); 
  m_hnstriplayers = tfserv->make<TH1D>("nstriplayers","Number of Strip Layers",20,-0.5,19.5);
  m_hnstriplayers->GetXaxis()->SetTitle("Nlayers");   m_hnstriplayers->GetYaxis()->SetTitle("Tracks"); 

  m_hnhitphieta = tfserv->make<TProfile2D>("nhitphivseta","N valid hits vs #phi vs #eta",netabin2d,etamin,etamax,40,-M_PI,M_PI);
  m_hnhitphieta->GetXaxis()->SetTitle("#eta");   m_hnhitphieta->GetYaxis()->SetTitle("#phi"); 
  m_hnlayerphieta = tfserv->make<TProfile2D>("nlayerphivseta","N layers vs #phi vs #eta",netabin2d,etamin,etamax,40,-M_PI,M_PI);
  m_hnlayerphieta->GetXaxis()->SetTitle("#eta");   m_hnlayerphieta->GetYaxis()->SetTitle("#phi"); 

  m_halgo = tfserv->make<TH1D>("algo","Algorithm number",reco::TrackBase::algoSize,-0.5,reco::TrackBase::algoSize-0.5);
  m_halgo->GetXaxis()->SetTitle("algorithm");   m_halgo->GetYaxis()->SetTitle("Tracks"); 
  m_hphieta = tfserv->make<TH2D>("phivseta","#phi vs #eta",netabin2d,etamin,etamax,40,-M_PI,M_PI);
  m_hphieta->GetXaxis()->SetTitle("#eta");   m_hphieta->GetYaxis()->SetTitle("#phi"); 

  if(m_runHisto) {
    m_ntrkvsorbrun = m_rhm.makeTProfile("ntrkvsorbrun","Number of tracks vs orbit number",m_maxLS*m_LSfrac,0,m_maxLS*262144);
  }

}


TrackCount::~TrackCount()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TrackCount::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   edm::Service<TFileService> tfserv;

   Handle<reco::TrackCollection> tracks;
   iEvent.getByToken(m_trkcollToken,tracks);

   //

   m_ntrk->Fill(tracks->size());

  // get luminosity

  edm::Handle<LumiDetails> ld;
  iEvent.getLuminosityBlock().getByToken(m_lumiProducerToken,ld);

  if(ld.isValid()) {
    if(ld->isValid()) {
    float lumi = ld->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing())*6.37;
    m_ntrkvslumi->Fill(lumi,tracks->size());
    m_ntrkvslumi2D->Fill(lumi,tracks->size());
    }
  }

   if(m_runHisto) {
     if(m_ntrkvsorbrun && *m_ntrkvsorbrun) (*m_ntrkvsorbrun)->Fill(iEvent.orbitNumber(),tracks->size());
   }

   unsigned int nhptrk = 0;
   double sumptsq = 0.;

   reco::TrackBase::TrackQuality quality = reco::TrackBase::qualityByName("highPurity");

   if(m_dump) edm::LogInfo("TrackDump") << " isHP algo pt eta phi chi2N chi2 ndof nlay npxl n3dl nlost ";

   for(reco::TrackCollection::const_iterator it = tracks->begin();it!=tracks->end();it++) {

     if(m_dump) {
       edm::LogVerbatim("TrackDump") << it->quality(quality) << " "
				     << it->algo() << " "
				     << it->pt() << " "
				     << it->eta() << " "
				     << it->phi() << " "
				     << it->normalizedChi2() << " "
				     << it->chi2() << " "
				     << it->ndof() << " "
				     << it->hitPattern().trackerLayersWithMeasurement() << " "
				     << it->hitPattern().pixelLayersWithMeasurement() << " "
				     << it->hitPattern().numberOfValidStripLayersWithMonoAndStereo() << " "
				     << it->hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS) << " ";

     }

     m_hnlosthits->Fill(it->hitPattern().numberOfLostTrackerHits(reco::HitPattern::TRACK_HITS));

     m_hnrhits->Fill(it->hitPattern().numberOfValidTrackerHits());
     m_hnpixelrhits->Fill(it->hitPattern().numberOfValidPixelHits());
     m_hnstriprhits->Fill(it->hitPattern().numberOfValidStripHits());
     m_hnhitphieta->Fill(it->eta(),it->phi(),it->hitPattern().numberOfValidTrackerHits());

     m_hnlostlayers->Fill(it->hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS));

     m_hnlayers->Fill(it->hitPattern().trackerLayersWithMeasurement());
     m_hnpixellayers->Fill(it->hitPattern().pixelLayersWithMeasurement());
     m_hnstriplayers->Fill(it->hitPattern().stripLayersWithMeasurement());
     m_hnlayerphieta->Fill(it->eta(),it->phi(),it->hitPattern().trackerLayersWithMeasurement());

     m_halgo->Fill(it->algo());

     m_hphi->Fill(it->phi());
     m_heta->Fill(it->eta());
     m_hphieta->Fill(it->eta(),it->phi());

     double pt = it->pt();
     m_hpt->Fill(pt);
     m_chisq->Fill(it->chi2());
     m_chisqnorm->Fill(it->normalizedChi2());
     m_ndof->Fill(it->ndof());
     if(m_2dhistos) {
       m_chisqvseta->Fill(it->eta(),it->chi2());
       m_chisqnormvseta->Fill(it->eta(),it->normalizedChi2());
       m_ndofvseta->Fill(it->eta(),it->ndof());
     }
     m_hptphieta->Fill(it->eta(),it->phi(),pt);
     sumptsq += pt*pt;
     if(it->p()) m_hcos->Fill(it->pz()/it->p());
     if(it->quality(quality)) nhptrk++;
   }

   m_nhptrk->Fill(nhptrk);

   const double hpfrac = tracks->size() > 0 ? double(nhptrk)/double(tracks->size()) : 0.;
   m_hhpfrac->Fill(hpfrac);
   m_hsqsumptsq->Fill(sqrt(sumptsq));

}

void 
TrackCount::beginRun(const edm::Run& iRun, const edm::EventSetup&)
{
  m_rhm.beginRun(iRun);

  if(m_runHisto) {
    (*m_ntrkvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");    (*m_ntrkvsorbrun)->GetYaxis()->SetTitle("Ntracks");
    (*m_ntrkvsorbrun)->SetCanExtend(TH1::kXaxis);
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(TrackCount);
