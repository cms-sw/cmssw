#include "FWCore/Framework/interface/Event.h"

#include <FWCore/PluginManager/interface/ModuleDef.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 
#include "FWCore/Utilities/interface/InputTag.h"

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <DataFormats/MuonReco/interface/EmulatedME0Segment.h>
#include <DataFormats/MuonReco/interface/EmulatedME0SegmentCollection.h>

#include <DataFormats/MuonReco/interface/ME0Muon.h>
#include <DataFormats/MuonReco/interface/ME0MuonCollection.h>

// #include "CLHEP/Matrix/SymMatrix.h"
// #include "CLHEP/Matrix/Matrix.h"
// #include "CLHEP/Vector/ThreeVector.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
//#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "TLorentzVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

//#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
//#include "TRandom3.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"

#include "DataFormats/Math/interface/deltaR.h"
//#include "DataFormats/Math/interface/deltaPhi.h"
//#include <deltaR.h>
#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"


//Associator for chi2: Including header files
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/plugins/CosmicParametersDefinerForTPESProducer.h"

#include "CommonTools/CandAlgos/interface/GenParticleCustomSelector.h"


#include "Fit/FitResult.h"
#include "TF1.h" 


#include "TMath.h"
#include "TLorentzVector.h"

#include "TH1.h" 
#include <TH2.h>
#include "TFile.h"
#include <TProfile.h>
#include "TStyle.h"
#include <TCanvas.h>
#include <TLatex.h>
//#include "CMSStyle.C"
//#include "tdrstyle.C"
//#include "lumi.C"
#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include <Geometry/GEMGeometry/interface/ME0EtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <DataFormats/MuonDetId/interface/ME0DetId.h>


#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TGraph.h"

#include <sstream>    

#include <iostream>
#include <fstream>

class StandAloneMuonAnalyzer : public edm::EDAnalyzer {
public:
  explicit StandAloneMuonAnalyzer(const edm::ParameterSet&);
  ~StandAloneMuonAnalyzer();
  FreeTrajectoryState getFTS(const GlobalVector& , const GlobalVector& , 
			     int , const AlgebraicSymMatrix66& ,
			     const MagneticField* );

  FreeTrajectoryState getFTS(const GlobalVector& , const GlobalVector& , 
			     int , const AlgebraicSymMatrix55& ,
			     const MagneticField* );
    void getFromFTS(const FreeTrajectoryState& ,
		  GlobalVector& , GlobalVector& , 
		  int& , AlgebraicSymMatrix66& );


  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  //virtual void beginJob(const edm::EventSetup&);
  void beginJob();

  //For Track Association



  //protected:
  
  private:
  //Associator for chi2: objects
  //edm::InputTag associatormap;
  bool UseAssociators;
  bool RejectEndcapMuons;
  //std::string parametersDefiner;


  TString histoFolder;
  TFile* histoFile; 
  TH2F *StandalonePtDiff_s; TProfile *StandalonePtDiff_p; TH1F *StandalonePtDiff_h; TH1F *StandaloneQOverPtDiff_h; TH1F *StandalonePtDiff_rms; TH1F *StandalonePtDiff_gaus_narrow; TH1F *StandalonePtDiff_gaus_wide;
  TH1F *StandalonePtDiff_gaus;
  TH1F *StandaloneMatchedMuon_Eta;
  TH1F *StandaloneMuonRecoEff_Eta;

  TH1F *GenMuon_Eta;

  double  FakeRatePtCut, MatchingWindowDelR;

  double Nevents;


//Removing this
};

StandAloneMuonAnalyzer::StandAloneMuonAnalyzer(const edm::ParameterSet& iConfig) 
{
  histoFile = new TFile(iConfig.getParameter<std::string>("HistoFile").c_str(), "recreate");
  histoFolder = iConfig.getParameter<std::string>("HistoFolder").c_str();
  // me0MuonSelector = iConfig.getParameter<std::string>("ME0MuonSelectionType").c_str();
  // RejectEndcapMuons = iConfig.getParameter< bool >("RejectEndcapMuons");
  // UseAssociators = iConfig.getParameter< bool >("UseAssociators");

  FakeRatePtCut   = iConfig.getParameter<double>("FakeRatePtCut");
  MatchingWindowDelR   = iConfig.getParameter<double>("MatchingWindowDelR");



}



//void StandAloneMuonAnalyzer::beginJob(const edm::EventSetup& iSetup)
void StandAloneMuonAnalyzer::beginJob()
{
  StandaloneMatchedMuon_Eta = new TH1F("StandaloneMatchedMuon_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );
  StandaloneMuonRecoEff_Eta = new TH1F("StandaloneMuonRecoEff_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );

  StandalonePtDiff_s = new TH2F("StandalonePtDiff_s" , "Relative pt difference", 4, 2.0, 2.4, 200,-1,1.0);
  StandalonePtDiff_h = new TH1F("StandalonePtDiff_h" , "pt resolution", 100,-0.5,0.5);
  StandaloneQOverPtDiff_h = new TH1F("StandaloneQOverPtDiff_h" , "q/pt resolution", 100,-0.5,0.5);
  StandalonePtDiff_p = new TProfile("StandalonePtDiff_p" , "pt resolution vs. #eta", 4, 2.0, 2.4, -1.0,1.0,"s");


  StandalonePtDiff_gaus    = new TH1F( "StandalonePtDiff_gaus",    "GAUS_WIDE", 4, 2.0, 2.4 ); 

  GenMuon_Eta = new TH1F("GenMuon_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );

  Nevents=0;


}


StandAloneMuonAnalyzer::~StandAloneMuonAnalyzer(){}

void
StandAloneMuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)

{
  using namespace edm;
  using namespace reco;
  //run_ = (int)iEvent.id().run();
  //event_ = (int)iEvent.id().event();


    //David's functionality

  Handle<GenParticleCollection> genParticles;

  iEvent.getByLabel<GenParticleCollection>("genParticles", genParticles);
  const GenParticleCollection genParticlesForChi2 = *(genParticles.product());

  unsigned int gensize=genParticles->size();




  //Standalone Muon study:
  Nevents++;
  edm::Handle<std::vector<Muon> > muons;
  iEvent.getByLabel("muons", muons);
  std::cout<<"Muon size = "<<muons->size()<<std::endl;

  for(unsigned int i=0; i<gensize; ++i) {
    const reco::GenParticle& CurrentParticle=(*genParticles)[i];
    if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) ){  

      if (CurrentParticle.pt() >FakeRatePtCut) {
	GenMuon_Eta->Fill(fabs(CurrentParticle.eta()));
      }

      double LowestDelR = 9999;
      double thisDelR = 9999;
      
      std::vector<double> ReferenceTrackPt;

      //double VertexDiff=-1,PtDiff=-1,QOverPtDiff=-1,PDiff=-1,MatchedEta=-1;
      double PtDiff=-1,QOverPtDiff=-1,MatchedEta=-1;
      for (std::vector<Muon>::const_iterator thisMuon = muons->begin();
  	   thisMuon != muons->end(); ++thisMuon){
  	if (thisMuon->isStandAloneMuon()){
	  
  	  TrackRef tkRef = thisMuon->outerTrack();
  	  thisDelR = reco::deltaR(CurrentParticle,*tkRef);
  	  if (tkRef->pt() > FakeRatePtCut ) {
  	    if (thisDelR < MatchingWindowDelR ){
  	      if (thisDelR < LowestDelR){
  		LowestDelR = thisDelR;

  		MatchedEta=fabs(CurrentParticle.eta());
  		//VertexDiff = fabs(tkRef->vz()-CurrentParticle.vz());
  		QOverPtDiff = ( (tkRef->charge() /tkRef->pt()) - (CurrentParticle.charge()/CurrentParticle.pt() ) )/  (CurrentParticle.charge()/CurrentParticle.pt() );
  		PtDiff = (tkRef->pt() - CurrentParticle.pt())/CurrentParticle.pt();
  		//PDiff = (tkRef->p() - CurrentParticle.p())/CurrentParticle.p();
  	      }
  	    }
  	  }
  	}
      }
      StandaloneMatchedMuon_Eta->Fill(MatchedEta);
      //StandaloneVertexDiff_h->Fill(VertexDiff);
      StandalonePtDiff_h->Fill(PtDiff);	
      StandaloneQOverPtDiff_h->Fill(QOverPtDiff);
      StandalonePtDiff_s->Fill(CurrentParticle.eta(),PtDiff);

    }
  }

  
}

void StandAloneMuonAnalyzer::endJob() 
{


  std::cout<<"Nevents = "<<Nevents<<std::endl;
  //TString cmsText     = "CMS Prelim.";
  //TString cmsText     = "#splitline{CMS PhaseII Simulation}{Prelim}";
  TString cmsText     = "CMS PhaseII Simulation Prelim.";

  TString lumiText = "PU 140, 14 TeV";
  float cmsTextFont   = 61;  // default is helvetic-bold

 
  //float extraTextFont = 52;  // default is helvetica-italics
  float lumiTextSize     = 0.05;

  float lumiTextOffset   = 0.2;
  float cmsTextSize      = 0.05;
  //float cmsTextOffset    = 0.1;  // only used in outOfFrame version

  // float relPosX    = 0.045;
  // float relPosY    = 0.035;
  // float relExtraDY = 1.2;

  // //ratio of "CMS" and extra text size
  // float extraOverCmsTextSize  = 0.76;


  histoFile->cd();

  TCanvas *c1 = new TCanvas("c1", "canvas" );

  std::stringstream PtCutString;
  PtCutString<<"#splitline{DY }{Reco Track p_{T} > "<<FakeRatePtCut<<" GeV}";
  const std::string& ptmp = PtCutString.str();
  const char* pcstr = ptmp.c_str();


  TLatex* txt =new TLatex;
  //txt->SetTextAlign(12);
  //txt->SetTextFont(42);
  txt->SetNDC();
  //txt->SetTextSize(0.05);
  txt->SetTextFont(132);
  txt->SetTextSize(0.05);


  float t = c1->GetTopMargin();
  float r = c1->GetRightMargin();

  TLatex* latex1 = new TLatex;
  latex1->SetNDC();
  latex1->SetTextAngle(0);
  latex1->SetTextColor(kBlack);    


  latex1->SetTextFont(42);
  latex1->SetTextAlign(31); 
  //latex1->SetTextSize(lumiTextSize*t);    
  latex1->SetTextSize(lumiTextSize);    
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  TLatex* latex = new TLatex;
  latex->SetTextFont(cmsTextFont);
  latex->SetNDC();
  latex->SetTextSize(cmsTextSize);


  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  StandaloneMatchedMuon_Eta->Write();   StandaloneMatchedMuon_Eta->Draw();  c1->Print(histoFolder+"/StandaloneMatchedMuon_Eta.png");

  GenMuon_Eta->Write();   GenMuon_Eta->Draw();  c1->Print(histoFolder+"/GenMuon_Eta.png");

  StandaloneMatchedMuon_Eta->Sumw2();  GenMuon_Eta->Sumw2();

  std::cout<<"Making eff plot"<<std::endl;

  StandaloneMuonRecoEff_Eta->Divide(StandaloneMatchedMuon_Eta, GenMuon_Eta, 1, 1, "B");
  std::cout<<"Making eff plot"<<std::endl;
  StandaloneMuonRecoEff_Eta->GetXaxis()->SetTitle("Gen Muon |#eta|");
  StandaloneMuonRecoEff_Eta->GetXaxis()->SetTitleSize(0.05);
  StandaloneMuonRecoEff_Eta->GetYaxis()->SetTitle("Standalone Muon Efficiency");
  StandaloneMuonRecoEff_Eta->GetYaxis()->SetTitleSize(0.05);
  std::cout<<"Making eff plot"<<std::endl;
  //StandaloneMuonRecoEff_Eta->SetMinimum(StandaloneMuonRecoEff_Eta->GetMinimum()-0.1);
  StandaloneMuonRecoEff_Eta->SetMinimum(0);
  //StandaloneMuonRecoEff_Eta->SetMaximum(StandaloneMuonRecoEff_Eta->GetMaximum()+0.1);
  StandaloneMuonRecoEff_Eta->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  StandaloneMuonRecoEff_Eta->Write();   StandaloneMuonRecoEff_Eta->Draw();  
  std::cout<<"Making eff plot"<<std::endl;
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestStandaloneMuonRecoEff_Eta.png");
  c1->Print(histoFolder+"/StandaloneMuonRecoEff_Eta.png");

  TH1D *test;
  test= new TH1D("test"   , "pt resolution"   , 200, -1.0, 1.0 );      
  for(Int_t i=1; i<=StandalonePtDiff_s->GetNbinsX(); ++i) {


    std::stringstream tempstore;
    tempstore<<i;
    const std::string& thistemp = tempstore.str();
    test->Draw();



    test= new TH1D("test"   , "pt resolution"   , 200, -1.0, 1.0 );  
    test->Draw();
    // Redoing for pt 40+
    StandalonePtDiff_s->ProjectionY("test",i,i,"");
    if (test->Integral() < 1.0) continue;

    TF1 *Standalonegaus = new TF1("Standalonegaus","gaus",-.2,.2);
    test->Fit(Standalonegaus,"R");

    Double_t w2 ,e_w2;
    w2  = Standalonegaus->GetParameter(2);
    e_w2  = Standalonegaus->GetParError(2);

    StandalonePtDiff_gaus->SetBinContent(i, w2); 
    StandalonePtDiff_gaus->SetBinError(i, e_w2); 

    test->Draw();
    TString  FileName = "Bin"+thistemp+"StandaloneFit.png";
    c1->Print(histoFolder+"/"+FileName);
     
    delete test;
    //test->Clear();
  }

  StandalonePtDiff_gaus->SetMarkerStyle(22); 
  StandalonePtDiff_gaus->SetMarkerSize(1.2); 
  StandalonePtDiff_gaus->SetMarkerColor(kBlue); 
  //StandalonePtDiff_gaus->SetLineColor(kRed); 
  
  //StandalonePtDiff_gaus->Draw("PL"); 

  StandalonePtDiff_gaus->GetXaxis()->SetTitle("Gen Muon |#eta|");
  StandalonePtDiff_gaus->GetYaxis()->SetTitle("Gaussian width of (pt track-ptgen)/ptgen");
  StandalonePtDiff_gaus->Write();     StandalonePtDiff_gaus->Draw("PE");  c1->Print(histoFolder+"/StandalonePtDiff_gaus.png");


  delete histoFile; histoFile = 0;
}



FreeTrajectoryState
StandAloneMuonAnalyzer::getFTS(const GlobalVector& p3, const GlobalVector& r3, 
			   int charge, const AlgebraicSymMatrix55& cov,
			   const MagneticField* field){

  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CurvilinearTrajectoryError tCov(cov);
  
  return cov.kRows == 5 ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars) ;
}

FreeTrajectoryState
StandAloneMuonAnalyzer::getFTS(const GlobalVector& p3, const GlobalVector& r3, 
			   int charge, const AlgebraicSymMatrix66& cov,
			   const MagneticField* field){

  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CartesianTrajectoryError tCov(cov);
  
  return cov.kRows == 6 ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars) ;
}

void StandAloneMuonAnalyzer::getFromFTS(const FreeTrajectoryState& fts,
				    GlobalVector& p3, GlobalVector& r3, 
				    int& charge, AlgebraicSymMatrix66& cov){
  GlobalVector p3GV = fts.momentum();
  GlobalPoint r3GP = fts.position();

  GlobalVector p3T(p3GV.x(), p3GV.y(), p3GV.z());
  GlobalVector r3T(r3GP.x(), r3GP.y(), r3GP.z());
  p3 = p3T;
  r3 = r3T;  //Yikes, was setting this to p3T instead of r3T!?!
  // p3.set(p3GV.x(), p3GV.y(), p3GV.z());
  // r3.set(r3GP.x(), r3GP.y(), r3GP.z());
  
  charge = fts.charge();
  cov = fts.hasError() ? fts.cartesianError().matrix() : AlgebraicSymMatrix66();

}

DEFINE_FWK_MODULE(StandAloneMuonAnalyzer);
