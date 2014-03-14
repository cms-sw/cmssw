#include "FWCore/Framework/interface/Event.h"

#include <FWCore/PluginManager/interface/ModuleDef.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

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

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
//#include "TRandom3.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"

#include "DataFormats/Math/interface/deltaR.h"
//#include "DataFormats/Math/interface/deltaPhi.h"
//#include <deltaR.h>


#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "TMath.h"
#include "TLorentzVector.h"

#include "TH1.h" 
#include <TH2.h>
#include "TFile.h"
#include <TProfile.h>


class TestAnalyzer_Final : public edm::EDAnalyzer {
public:
  explicit TestAnalyzer_Final(const edm::ParameterSet&);
  ~TestAnalyzer_Final();


  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  void beginJob();

  //protected:
  
  private:
  TFile* histoFile;
  TH1F *Candidate_Eta;  TH1F *Mass_h;  int NumCands; int NumSegs;
  TH1F *Segment_Eta;  TH1F *Track_Eta;  TH1F *Muon_Eta; 
  TH1F *TracksPerSegment_h;  TH2F *TracksPerSegment_s;  TProfile *TracksPerSegment_p;
  TH1F *GenMuon_Eta; TH1F *GenMuon_Pt;   TH1F *MatchedME0Muon_Eta; TH1F *MatchedME0Muon_Pt; 
  TH1F *MuonRecoEff_Eta;  TH1F *MuonRecoEff_Pt;
  int TrackCount;


//Removing this
};

TestAnalyzer_Final::TestAnalyzer_Final(const edm::ParameterSet& iConfig) 
{
  histoFile = new TFile(iConfig.getParameter<std::string>("HistoFile").c_str(), "recreate");
  NumCands = 0;
  NumSegs = 0;
  TrackCount = 0;
}



void TestAnalyzer_Final::beginJob()
{
  Candidate_Eta = new TH1F("Candidate_Eta"      , "Candidate #eta"   , 40, 2.2, 4.2 );
  Track_Eta = new TH1F("Track_Eta"      , "Track #eta"   , 40, 2.2, 4.2 );
  Segment_Eta = new TH1F("Segment_Eta"      , "Segment #eta"   , 40, 2.2, 4.2 );
  Muon_Eta = new TH1F("Muon_Eta"      , "Muon #eta"   , 40, 2.2, 4.2 );
  GenMuon_Eta = new TH1F("GenMuon_Eta"      , "Muon #eta"   , 40, 2.2, 4.2 );
  GenMuon_Pt = new TH1F("GenMuon_Pt"      , "Muon p_{T}"   , 40,0 , 40 );
  MatchedME0Muon_Eta = new TH1F("MatchedME0Muon_Eta"      , "Muon #eta"   , 40, 2.2, 4.2 );
  MatchedME0Muon_Pt = new TH1F("MatchedME0Muon_Pt"      , "Muon p_{T}"   , 40,0 , 40 );
  Mass_h = new TH1F("Mass_h"      , "Mass"   , 100, 0., 200 );
  MuonRecoEff_Eta = new TH1F("MuonRecoEff_Eta"      , "Reco Efficiency"   ,40, 2.2, 4.2  );
  MuonRecoEff_Pt = new TH1F("MuonRecoEff_Pt"      , "Reco Efficiency"   ,40, 0,40  );
  TracksPerSegment_h = new TH1F("TracksPerSegment_h", "Number of tracks", 5,0.,5.);
  TracksPerSegment_s = new TH2F("TracksPerSegment_s" , "Tracks per segment vs |#eta|, z = 560 cm", 40, 2.4, 4.0, 5,0.,5.);
  TracksPerSegment_p = new TProfile("TracksPerSegment_p" , "Tracks per segment vs |#eta|, z = 560 cm", 40, 2.4, 4.0, 0.,5.);
  
}


TestAnalyzer_Final::~TestAnalyzer_Final(){}

void
TestAnalyzer_Final::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)

{

  using namespace edm;

  //run_ = (int)iEvent.id().run();
  //event_ = (int)iEvent.id().event();


    //David's functionality
    

  using namespace reco;

  // Handle <ME0MuonCollection > OurMuons;
  // iEvent.getByLabel <ME0MuonCollection> ("me0SegmentMatcher", OurMuons);

  Handle <std::vector<RecoChargedCandidate> > OurCandidates;
  iEvent.getByLabel <std::vector<RecoChargedCandidate> > ("me0MuonConverter", OurCandidates);

  Handle<std::vector<EmulatedME0Segment> > OurSegments;
  iEvent.getByLabel<std::vector<EmulatedME0Segment> >("me0SegmentProducer", OurSegments);

  Handle<GenParticleCollection> genParticles;
  iEvent.getByLabel<GenParticleCollection>("genParticles", genParticles);

  Handle <TrackCollection > generalTracks;
  iEvent.getByLabel <TrackCollection> ("generalTracks", generalTracks);

  Handle <std::vector<ME0Muon> > OurMuons;
  iEvent.getByLabel <std::vector<ME0Muon> > ("me0SegmentMatcher", OurMuons);



  unsigned int gensize=genParticles->size();
  for(unsigned int i=0; i<gensize; ++i) {
    const reco::GenParticle& CurrentParticle=(*genParticles)[i];
    if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) && ((abs(CurrentParticle.eta())>2.4) && (abs(CurrentParticle.eta()) < 4.0))){  
      GenMuon_Eta->Fill(CurrentParticle.eta());
      GenMuon_Pt->Fill(CurrentParticle.pt());

      double LowestDelR = 9999;
      double thisDelR = 9999;

      for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
	   thisMuon != OurMuons->end(); ++thisMuon){
	TrackRef tkRef = thisMuon->innerTrack();
	
	thisDelR = reco::deltaR(CurrentParticle,*tkRef);
	if (thisDelR < 0.15){
	  if (thisDelR < LowestDelR){
	    LowestDelR = thisDelR;
	    MatchedME0Muon_Eta->Fill(CurrentParticle.eta());
	    MatchedME0Muon_Pt->Fill(CurrentParticle.pt());
	  }
	}
      }

    }
  }
  //unsigned int recosize=OurCandidates->size();
  for (std::vector<EmulatedME0Segment>::const_iterator thisSegment = OurSegments->begin();
       thisSegment != OurSegments->end();++thisSegment){
    // double theta = atan(thisSegment->localDirection().y()/ thisSegment->localDirection().x());
    // double tempeta = -log(tan (theta/2.));
    LocalVector TempVect(thisSegment->localDirection().x(),thisSegment->localDirection().y(),thisSegment->localDirection().z());
    Segment_Eta->Fill(TempVect.eta());
    NumSegs++;
  }

  
  for (std::vector<Track>::const_iterator thisTrack = generalTracks->begin();
       thisTrack != generalTracks->end();++thisTrack){
    Track_Eta->Fill(thisTrack->eta());
    if ( (thisTrack->eta() > 2.4) && (thisTrack->eta() < 4.0)) TrackCount++;
    //std::cout<<thisTrack->eta()<<std::endl;
  }

  // std::vector<int> UniqueIdList;
  // std::vector<int> Ids;
  std::vector<Double_t> SegmentEta;
  std::vector<const EmulatedME0Segment*> Ids;
  std::vector<const EmulatedME0Segment*> UniqueIdList;

  for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
       thisMuon != OurMuons->end(); ++thisMuon){
    TrackRef tkRef = thisMuon->innerTrack();
    EmulatedME0SegmentRef segRef = thisMuon->me0segment();
    //int SegId = thisMuon->segRefId();
    //int SegId = segRef.get();
    const EmulatedME0Segment* SegId = segRef.get();
    //PointersVector.push_back(segRef.get());
    bool IsNew = true;
    for (unsigned int i =0; i < Ids.size(); i++){
      if (SegId == Ids[i]) IsNew=false;
    }
    if (IsNew) {
      std::cout<<"Found: "<<SegId<<std::endl;
      UniqueIdList.push_back(SegId);
      LocalVector TempVect(segRef->localDirection().x(),segRef->localDirection().y(),segRef->localDirection().z());
      SegmentEta.push_back(TempVect.eta());
    }
    Ids.push_back(SegId);

    //LocalVector TempVect(tkRef->px(),tkRef->py(),tkRef->pz());
    Muon_Eta->Fill(tkRef->eta());
  }
  
  for (unsigned int i = 0; i < UniqueIdList.size(); i++){
    int Num=0;
    for (unsigned int j = 0; j < Ids.size(); j++){
      if (Ids[j] == UniqueIdList[i]) Num++;
    }
    std::cout<<Num<<std::endl;
    TracksPerSegment_h->Fill((double)Num);
    TracksPerSegment_s->Fill(SegmentEta[i], (double)Num);
    TracksPerSegment_p->Fill(SegmentEta[i], (double)Num);
  }
  

  //std::cout<<recosize<<std::endl;
  for (std::vector<RecoChargedCandidate>::const_iterator thisCandidate = OurCandidates->begin();
       thisCandidate != OurCandidates->end(); ++thisCandidate){
    NumCands++;
    TLorentzVector CandidateVector;
    CandidateVector.SetPtEtaPhiM(thisCandidate->pt(),thisCandidate->eta(),thisCandidate->phi(),0);
    //std::cout<<"On a muon"<<std::endl;
    //std::cout<<thisCandidate->eta()<<std::endl;
    Candidate_Eta->Fill(thisCandidate->eta());
  }

  if (OurCandidates->size() == 2){
    TLorentzVector CandidateVector1,CandidateVector2;
    CandidateVector1.SetPtEtaPhiM((*OurCandidates)[0].pt(),(*OurCandidates)[0].eta(),(*OurCandidates)[0].phi(),0);
    CandidateVector2.SetPtEtaPhiM((*OurCandidates)[1].pt(),(*OurCandidates)[1].eta(),(*OurCandidates)[1].phi(),0);
    Double_t Mass = (CandidateVector1+CandidateVector2).M();
    Mass_h->Fill(Mass);
  }
  
  
}

void TestAnalyzer_Final::endJob() 
{
  histoFile->cd();
  Candidate_Eta->Write();
  Track_Eta->Write();
  Segment_Eta->Write();

  Muon_Eta->Write();
  GenMuon_Eta->Write();
  GenMuon_Pt->Write();

  MatchedME0Muon_Eta->Write();
  MatchedME0Muon_Pt->Write();

  Mass_h->Write();
  TracksPerSegment_h->Write();  TracksPerSegment_s->Write();  TracksPerSegment_p->Write();

  GenMuon_Eta->Sumw2();  MatchedME0Muon_Eta->Sumw2();
  GenMuon_Pt->Sumw2();  MatchedME0Muon_Pt->Sumw2();

  
  MuonRecoEff_Eta->Divide(MatchedME0Muon_Eta, GenMuon_Eta, 1, 1, "B");
  MuonRecoEff_Eta->Write();

  MuonRecoEff_Pt->Divide(MatchedME0Muon_Pt, GenMuon_Pt, 1, 1, "B");
  MuonRecoEff_Pt->Write();
  //std::cout<<NumCands<<std::endl;
  //std::cout<<NumSegs<<std::endl;
  //std::cout<<TrackCount<<std::endl;
  delete histoFile; histoFile = 0;
}

DEFINE_FWK_MODULE(TestAnalyzer_Final);
