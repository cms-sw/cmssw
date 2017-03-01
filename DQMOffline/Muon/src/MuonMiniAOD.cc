#include "DQMOffline/Muon/interface/MuonMiniAOD.h"




using namespace std;
using namespace edm;



MuonMiniAOD::MuonMiniAOD(const edm::ParameterSet& pSet) {
  parameters = pSet;

  // the services:
  
  theMuonCollectionLabel_ = consumes<edm::View<pat::Muon> >  (parameters.getParameter<edm::InputTag>("MuonCollection"));
  theVertexLabel_          = consumes<reco::VertexCollection>(parameters.getParameter<edm::InputTag>("VertexLabel"));
  theBeamSpotLabel_        = mayConsume<reco::BeamSpot>      (parameters.getParameter<edm::InputTag>("BeamSpotLabel"));

}


MuonMiniAOD::~MuonMiniAOD() { }
void MuonMiniAOD::bookHistograms(DQMStore::IBooker & ibooker,
				      edm::Run const & /*iRun*/,
				      edm::EventSetup const & /* iSetup */){
    
  ibooker.cd();
  ibooker.setCurrentFolder("Muons_miniAOD/MuonMiniAOD");
  

  workingPoints.push_back(ibooker.book2D("tightMuons" ,"Tight Muons"  ,2,1,3,2,1,3));
  workingPoints.push_back(ibooker.book2D("mediumMuons","Medium Muons" ,2,1,3,2,1,3));
  workingPoints.push_back(ibooker.book2D("looseMuons" ,"Loose Muons"  ,2,1,3,2,1,3));
  workingPoints.push_back(ibooker.book2D("highPtMuons","High Pt Muons",2,1,3,2,1,3));
  workingPoints.push_back(ibooker.book2D("softMuons"  ,"Soft Muons"   ,2,1,3,2,1,3));

  for (std::vector<MonitorElement*>::iterator monitor = workingPoints.begin();
       monitor != workingPoints.end(); ++monitor){
    (*monitor)-> setBinLabel(1,"Pass",1);
    (*monitor) -> setBinLabel(2,"No Pass",1);
    (*monitor) -> setBinLabel(1,"Pass",2);
    (*monitor) -> setBinLabel(2,"No Pass",2);
  }

}

bool MuonMiniAOD::PassesCut_A(edm::View<pat::Muon>::const_iterator muon1, reco::Vertex thePrimaryVertex, TString WorkingPoint){

  if (WorkingPoint == "tightMuons")
    return muon::isTightMuon(*muon1,thePrimaryVertex);
  else if (WorkingPoint == "mediumMuons")
    return muon::isMediumMuon(*muon1);
  else if (WorkingPoint == "looseMuons")
    return muon::isLooseMuon(*muon1);
  else if (WorkingPoint == "highPtMuons")
    return muon::isHighPtMuon(*muon1,thePrimaryVertex);
  else if (WorkingPoint == "softMuons")
    return muon::isSoftMuon(*muon1,thePrimaryVertex);
  else{
    LogInfo("RecoMuonValidator") << "[MuonMiniAOD]: MuonMiniAOD. Unknown WP, returning false.\n";
    return false;
  }
  
}

bool MuonMiniAOD::PassesCut_B(edm::View<pat::Muon>::const_iterator muon1, reco::Vertex thePrimaryVertex, TString WorkingPoint){
  
  if (WorkingPoint == "tightMuons")
    return muon1 -> isTightMuon(thePrimaryVertex);
  else if (WorkingPoint == "mediumMuons")
    return muon1 -> isMediumMuon();
  else if (WorkingPoint == "looseMuons")
    return muon1 -> isLooseMuon();
  else if (WorkingPoint == "highPtMuons")
    return muon1 -> isHighPtMuon(thePrimaryVertex);
  else if (WorkingPoint == "softMuons")
    return muon1 -> isSoftMuon(thePrimaryVertex);
  else{
    LogInfo("RecoMuonValidator") << "[MuonMiniAOD]: MuonMiniAOD. Unknown WP, returning false.\n";
    return false;
  }
   
}


void MuonMiniAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  LogTrace(metname)<<"[MuonMiniAOD] Analyze the mu";
  
 
  // Take the muon container
  edm::Handle<edm::View<pat::Muon> > muons; 
  iEvent.getByToken(theMuonCollectionLabel_,muons);
 
  //Vertex information
  edm::Handle<reco::VertexCollection> vertex;
  iEvent.getByToken(theVertexLabel_, vertex);  


  if(!muons.isValid()) return;

  reco::Vertex::Point posVtx;
  reco::Vertex::Error errVtx;
  unsigned int theIndexOfThePrimaryVertex = 999.;
  if (!vertex.isValid()) {
    LogTrace(metname) << "[EfficiencyAnalyzer] Could not find vertex collection" << std::endl;
    for (unsigned int ind=0; ind<vertex->size(); ++ind) {
      if ( (*vertex)[ind].isValid() && !((*vertex)[ind].isFake()) ) {
	theIndexOfThePrimaryVertex = ind;
	break;
      }
    }
  }

  if (theIndexOfThePrimaryVertex<100) {
    posVtx = ((*vertex)[theIndexOfThePrimaryVertex]).position();
    errVtx = ((*vertex)[theIndexOfThePrimaryVertex]).error();
  }   
  else {
    LogInfo("RecoMuonValidator") << "reco::PrimaryVertex not found, use BeamSpot position instead\n";
    
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByToken(theBeamSpotLabel_,recoBeamSpotHandle);
    reco::BeamSpot bs = *recoBeamSpotHandle;
    
    posVtx = bs.position();
    errVtx(0,0) = bs.BeamWidthX();
    errVtx(1,1) = bs.BeamWidthY();
    errVtx(2,2) = bs.sigmaZ();
  }
    


  const reco::Vertex thePrimaryVertex(posVtx,errVtx);

  for (edm::View<pat::Muon>::const_iterator muon1 = muons->begin(); muon1 != muons->end(); ++muon1){

    for (std::vector<MonitorElement*>::iterator monitor = workingPoints.begin();
	 monitor != workingPoints.end(); ++monitor){
      int Pass_A = 0;
      int Pass_B = 0;
      if (PassesCut_A(muon1, thePrimaryVertex,(*monitor)->getName()))
	Pass_A = 1;
      else
	Pass_A = 2;
      if (PassesCut_B(muon1, thePrimaryVertex,(*monitor)->getName()))
	Pass_B = 1;
      else
	Pass_B = 2;

      (*monitor)->Fill(Pass_A,Pass_B);
    }

  }
}
