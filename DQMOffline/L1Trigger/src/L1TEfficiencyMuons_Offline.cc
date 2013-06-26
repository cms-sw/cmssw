 /*
  * \file L1TEfficiencyMuons_Offline.cc
  *
  * $Date: 2013/03/18 17:17:53 $
  * $Revision: 1.2 $
  * \author J. Pela, C. Battilana
  *
  */
 
 #include "DQMOffline/L1Trigger/interface/L1TEfficiencyMuons_Offline.h"
  
 #include "DQMServices/Core/interface/DQMStore.h"
 
 #include "DataFormats/Histograms/interface/MEtoEDMFormat.h"

 #include "DataFormats/MuonReco/interface/MuonSelectors.h"
 
 #include "DataFormats/GeometrySurface/interface/Cylinder.h"
 #include "DataFormats/GeometrySurface/interface/Plane.h"

 #include "TMath.h"
 
 using namespace reco;
 using namespace trigger;
 using namespace edm;
 using namespace std;
 

//__________RECO-GMT Muon Pair Helper Class____________________________

MuonGmtPair::MuonGmtPair(const MuonGmtPair& muonGmtPair) {

  m_muon    = muonGmtPair.m_muon;
  m_gmt     = muonGmtPair.m_gmt;
  m_eta     = muonGmtPair.m_eta;
  m_phi_bar = muonGmtPair.m_phi_bar;
  m_phi_end = muonGmtPair.m_phi_end;

}


double MuonGmtPair::dR() {
  
  float dEta = m_gmt ? (m_gmt->etaValue() - eta()) : 999.;
  float dPhi = m_gmt ? (m_gmt->phiValue() - phi()) : 999.;
    
  float dr = sqrt(dEta*dEta + dPhi*dPhi);

  return dr;

}


void MuonGmtPair::propagate(ESHandle<MagneticField> bField,
			    ESHandle<Propagator> propagatorAlong,
			    ESHandle<Propagator> propagatorOpposite) {

  m_BField = bField;
  m_propagatorAlong = propagatorAlong;
  m_propagatorOpposite = propagatorOpposite;

  TrackRef standaloneMuon = m_muon->outerTrack();  
    
  TrajectoryStateOnSurface trajectory;
  trajectory = cylExtrapTrkSam(standaloneMuon, 500);  // track at MB2 radius - extrapolation
  if (trajectory.isValid()) {
    m_eta     = trajectory.globalPosition().eta();
    m_phi_bar = trajectory.globalPosition().phi();
  }
  
  trajectory = surfExtrapTrkSam(standaloneMuon, 790);   // track at ME2+ plane - extrapolation
  if (trajectory.isValid()) {
    m_eta     = trajectory.globalPosition().eta();      
    m_phi_end = trajectory.globalPosition().phi();
  }
  
  trajectory = surfExtrapTrkSam(standaloneMuon, -790); // track at ME2- disk - extrapolation
  if (trajectory.isValid()) {
    m_eta     = trajectory.globalPosition().eta();      
    m_phi_end = trajectory.globalPosition().phi();
  }
    
}


TrajectoryStateOnSurface MuonGmtPair::cylExtrapTrkSam(TrackRef track, double rho)
{

  Cylinder::PositionType pos(0, 0, 0);
  Cylinder::RotationType rot;
  Cylinder::CylinderPointer myCylinder = Cylinder::build(pos, rot, rho);
  
  FreeTrajectoryState recoStart = freeTrajStateMuon(track);
  TrajectoryStateOnSurface recoProp;
  recoProp = m_propagatorAlong->propagate(recoStart, *myCylinder);
  if (!recoProp.isValid()) {
    recoProp = m_propagatorOpposite->propagate(recoStart, *myCylinder);
  }
  return recoProp;

}


TrajectoryStateOnSurface MuonGmtPair::surfExtrapTrkSam(TrackRef track, double z)
{
  
  Plane::PositionType pos(0, 0, z);
  Plane::RotationType rot;
  Plane::PlanePointer myPlane = Plane::build(pos, rot);
    
  FreeTrajectoryState recoStart = freeTrajStateMuon(track);
  TrajectoryStateOnSurface recoProp;
  recoProp = m_propagatorAlong->propagate(recoStart, *myPlane);
  if (!recoProp.isValid()) {
    recoProp = m_propagatorOpposite->propagate(recoStart, *myPlane);
  }
  return recoProp;
}


FreeTrajectoryState MuonGmtPair::freeTrajStateMuon(TrackRef track)
{
 
  GlobalPoint  innerPoint(track->innerPosition().x(), track->innerPosition().y(),  track->innerPosition().z());
  GlobalVector innerVec  (track->innerMomentum().x(),  track->innerMomentum().y(),  track->innerMomentum().z());  
    
  FreeTrajectoryState recoStart(innerPoint, innerVec, track->charge(), &*m_BField);
    
  return recoStart;

}


//__________DQM_base_class_______________________________________________
L1TEfficiencyMuons_Offline::L1TEfficiencyMuons_Offline(const ParameterSet & ps){

  if (m_verbose) {
    cout << "[L1TEfficiencyMuons_Offline:] ____________ Storage initialization ____________ " << endl;
  }
  
  // Initializing DQM Store
  dbe = Service<DQMStore>().operator->();
  dbe->setVerbose(0);
  if (m_verbose) {cout << "[L1TEfficiencyMuons_Offline:] Pointer for DQM Store: " << dbe << endl;}
  
  // Initializing config params
  m_GmtPtCuts = ps.getUntrackedParameter< vector<int> >("gmtPtCuts");
  
  m_MuonInputTag =  ps.getUntrackedParameter<InputTag>("muonInputTag");
  m_GmtInputTag  =  ps.getUntrackedParameter<InputTag>("gmtInputTag");
  
  m_VtxInputTag =  ps.getUntrackedParameter<InputTag>("vtxInputTag");
  m_BsInputTag  =  ps.getUntrackedParameter<InputTag>("bsInputTag");

  m_trigInputTag = ps.getUntrackedParameter<InputTag>("trigInputTag");
  m_trigProcess  = ps.getUntrackedParameter<string>("trigProcess");
  m_trigNames    = ps.getUntrackedParameter<vector<string> >("triggerNames");

  // CB do we need them from cfi?
  m_MaxMuonEta   = 2.4;
  m_MaxGmtMuonDR = 0.7;
  m_MaxHltMuonDR = 0.1;
  // CB ignored at present
  //m_MinMuonDR    = 1.2;
  
}

 
//_____________________________________________________________________
L1TEfficiencyMuons_Offline::~L1TEfficiencyMuons_Offline(){ }
 

//_____________________________________________________________________
void L1TEfficiencyMuons_Offline::beginJob(void){
   
  if (m_verbose) {cout << "[L1TEfficiencyMuons_Offline:] Called beginJob." << endl;}
  
  bookControlHistos();
  
  vector<int>::const_iterator gmtPtCutsIt  = m_GmtPtCuts.begin();
  vector<int>::const_iterator gmtPtCutsEnd = m_GmtPtCuts.end();
  
  for (; gmtPtCutsIt!=gmtPtCutsEnd; ++ gmtPtCutsIt) {
    bookEfficiencyHistos((*gmtPtCutsIt));
  } 
  
}


//_____________________________________________________________________
void L1TEfficiencyMuons_Offline::endJob(void){
  
  if (m_verbose) {cout << "[L1TEfficiencyMuons_Offline:] Called endJob." << endl;}
  
}

 
//_____________________________________________________________________
void L1TEfficiencyMuons_Offline::beginRun(const edm::Run& run, const edm::EventSetup& iSetup){

  if (m_verbose) {cout << "[L1TEfficiencyMuons_Offline:] Called beginRun." << endl;}
  
  bool changed = true;
  
  m_hltConfig.init(run,iSetup,m_trigProcess,changed);
  
  vector<string>::const_iterator trigNamesIt  = m_trigNames.begin();
  vector<string>::const_iterator trigNamesEnd = m_trigNames.end();

  for (; trigNamesIt!=trigNamesEnd; ++trigNamesIt) { 
    
    TString tNameTmp = TString(*trigNamesIt); // use TString as it handles regex
    TRegexp tNamePattern = TRegexp(tNameTmp,true);
    int tIndex = -1;
    
    for (unsigned ipath = 0; ipath < m_hltConfig.size(); ++ipath) {
      
      TString tmpName = TString(m_hltConfig.triggerName(ipath));
      if (tmpName.Contains(tNamePattern)) {
	tIndex = int(ipath);
	m_trigIndices.push_back(tIndex);
      }

    }
    
    if (tIndex < 0 && m_verbose) {
      cout << "[L1TEfficiencyMuons_Offline:] Warning: Could not find trigger " 
	   << (*trigNamesIt) << endl;
    }
    
  }
  
}  


//_____________________________________________________________________
void L1TEfficiencyMuons_Offline::endRun(const edm::Run& run, const edm::EventSetup& iSetup){
  
  if (m_verbose) {cout << "[L1TEfficiencyMuons_Offline:] Called endRun." << endl;}
  
}


//_____________________________________________________________________
void L1TEfficiencyMuons_Offline::beginLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {
  
  if(m_verbose){
    cout << "[L1TEfficiencyMuons_Offline:] Called beginLuminosityBlock at LS=" 
         << lumiBlock.id().luminosityBlock() << endl;
  }
  
}


//_____________________________________________________________________
void L1TEfficiencyMuons_Offline::endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {
  
  if(m_verbose){
    cout << "[L1TEfficiencyMuons_Offline:] Called endLuminosityBlock at LS=" 
         << lumiBlock.id().luminosityBlock() << endl;
  }
  
}


//_____________________________________________________________________
void L1TEfficiencyMuons_Offline::analyze(const Event & iEvent, const EventSetup & eventSetup){

  Handle<reco::MuonCollection> muons;
  iEvent.getByLabel(m_MuonInputTag, muons);

  Handle<BeamSpot> beamSpot;
  iEvent.getByLabel(m_BsInputTag, beamSpot);

  Handle<VertexCollection> vertex;
  iEvent.getByLabel(m_VtxInputTag, vertex);
  
  Handle<L1MuGMTReadoutCollection> gmtCands;
  iEvent.getByLabel(m_GmtInputTag,gmtCands);
  
  Handle<edm::TriggerResults> trigResults;
  iEvent.getByLabel(InputTag("TriggerResults","",m_trigProcess),trigResults);
  
  edm::Handle<trigger::TriggerEvent> trigEvent;
  iEvent.getByLabel(m_trigInputTag,trigEvent);

  eventSetup.get<IdealMagneticFieldRecord>().get(m_BField);

  eventSetup.get<TrackingComponentsRecord>().get("SmartPropagatorAny",m_propagatorAlong);
  eventSetup.get<TrackingComponentsRecord>().get("SmartPropagatorAnyOpposite",m_propagatorOpposite);

  const Vertex primaryVertex = getPrimaryVertex(vertex,beamSpot);

  getTightMuons(muons,primaryVertex);
  getProbeMuons(trigResults,trigEvent); // CB add flag to run on orthogonal datasets (no T&P)
  getMuonGmtPairs(gmtCands);

  cout << "[L1TEfficiencyMuons_Offline:] Computing efficiencies" << endl;

  vector<MuonGmtPair>::const_iterator muonGmtPairsIt  = m_MuonGmtPairs.begin();
  vector<MuonGmtPair>::const_iterator muonGmtPairsEnd = m_MuonGmtPairs.end();
  
  for(; muonGmtPairsIt!=muonGmtPairsEnd; ++muonGmtPairsIt) {

    float eta = muonGmtPairsIt->eta();
    float phi = muonGmtPairsIt->phi();
    float pt  = muonGmtPairsIt->pt();

    // unmatched gmt cands have gmtPt = -1.
    float gmtPt  = muonGmtPairsIt->gmtPt();

    vector<int>::const_iterator gmtPtCutsIt  = m_GmtPtCuts.begin();
    vector<int>::const_iterator gmtPtCutsEnd = m_GmtPtCuts.end();

    for (; gmtPtCutsIt!=gmtPtCutsEnd; ++ gmtPtCutsIt) {
      
      int gmtPtCut = (*gmtPtCutsIt);
      bool gmtAboveCut = (gmtPt > gmtPtCut);

      stringstream ptCutToTag; ptCutToTag << gmtPtCut;
      string ptTag = ptCutToTag.str();

      if (fabs(eta) < m_MaxMuonEta) {
	
	m_EfficiencyHistos[gmtPtCut]["EffvsPt" + ptTag + "Den"]->Fill(pt);
	if (gmtAboveCut) m_EfficiencyHistos[gmtPtCut]["EffvsPt" + ptTag + "Num"]->Fill(pt);

	if (pt > gmtPtCut + 8.) { // efficiency in eta/phi at plateau
	  
	  m_EfficiencyHistos[gmtPtCut]["EffvsPhi" + ptTag + "Den"]->Fill(phi);
	  m_EfficiencyHistos[gmtPtCut]["EffvsEta" + ptTag + "Den"]->Fill(eta);
	  
	  if (gmtAboveCut) { 
	    m_EfficiencyHistos[gmtPtCut]["EffvsPhi" + ptTag + "Num"]->Fill(phi);
	    m_EfficiencyHistos[gmtPtCut]["EffvsEta" + ptTag + "Num"]->Fill(eta);
	  }
	}
      }
    }
  }

  cout << "[L1TEfficiencyMuons_Offline:] Computation finished" << endl;

  
}


//_____________________________________________________________________
void L1TEfficiencyMuons_Offline::bookControlHistos() { 
  
  if(m_verbose){cout << "[L1TEfficiencyMuons_Offline:] Booking Control Plot Histos" << endl;}

  dbe->setCurrentFolder("L1T/Efficiency/Muons/Control");
  
  string name = "MuonGmtDeltaR";
  m_ControlHistos[name] = dbe->book1D(name.c_str(),name.c_str(),25.,0.,2.5);

  name = "NTightVsAll";
  m_ControlHistos[name] = dbe->book2D(name.c_str(),name.c_str(),5,-0.5,4.5,5,-0.5,4.5);

  name = "NProbesVsTight";
  m_ControlHistos[name] = dbe->book2D(name.c_str(),name.c_str(),5,-0.5,4.5,5,-0.5,4.5);
  
}


//_____________________________________________________________________
void L1TEfficiencyMuons_Offline::bookEfficiencyHistos(int ptCut) { 
  
  if(m_verbose){
    cout << "[L1TEfficiencyMuons_Offline:] Booking Efficiency Plot Histos for pt cut = " 
	 << ptCut << endl;
  }

  stringstream ptCutToTag; ptCutToTag << ptCut;
  string ptTag = ptCutToTag.str();
  
  dbe->setCurrentFolder("L1T/Efficiency/Muons/");

  string effTag[2] = {"Den", "Num"};
  
  for(int iEffTag=0; iEffTag<2; ++ iEffTag) {
    string name = "EffvsPt" + ptTag + effTag[iEffTag];
    m_EfficiencyHistos[ptCut][name] = dbe->book1D(name.c_str(),name.c_str(),16,0.,40.);
    
    name = "EffvsPhi" + ptTag + effTag[iEffTag];
    m_EfficiencyHistos[ptCut][name] = dbe->book1D(name.c_str(),name.c_str(),12,0.,2*TMath::Pi());
    
    name = "EffvsEta" + ptTag + effTag[iEffTag];
    m_EfficiencyHistos[ptCut][name] = dbe->book1D(name.c_str(),name.c_str(),12,-2.4,2.4);
  }
  
}


//_____________________________________________________________________
const reco::Vertex L1TEfficiencyMuons_Offline::getPrimaryVertex( Handle<VertexCollection> & vertex,
								 Handle<BeamSpot> & beamSpot ) {
  
  Vertex::Point posVtx;
  Vertex::Error errVtx;
  
  bool hasPrimaryVertex = false;

  if (vertex.isValid())
    {

      vector<Vertex>::const_iterator vertexIt  = vertex->begin();
      vector<Vertex>::const_iterator vertexEnd = vertex->end();

      for (;vertexIt!=vertexEnd;++vertexIt) 
	{
	  if (vertexIt->isValid() && 
	      !vertexIt->isFake()) 
	    {
	      posVtx = vertexIt->position();
	      errVtx = vertexIt->error();
	      hasPrimaryVertex = true;	      
	      break;
	    }
	}
    }

  if ( !hasPrimaryVertex ) {

    if(m_verbose){
      cout << "[L1TEfficiencyMuons_Offline:] PrimaryVertex not found, use BeamSpot position instead" << endl;
    }
    
    posVtx = beamSpot->position();
    errVtx(0,0) = beamSpot->BeamWidthX();
    errVtx(1,1) = beamSpot->BeamWidthY();
    errVtx(2,2) = beamSpot->sigmaZ();
    
  }

  const Vertex primaryVertex(posVtx,errVtx);
  
  return primaryVertex;

}


//_____________________________________________________________________
void L1TEfficiencyMuons_Offline::getTightMuons(edm::Handle<reco::MuonCollection> & muons, 
					       const Vertex & vertex) {

  cout << "[L1TEfficiencyMuons_Offline:] Getting tight muons" << endl;
     
  m_TightMuons.clear();
  
  MuonCollection::const_iterator muonIt  = muons->begin();
  MuonCollection::const_iterator muonEnd = muons->end();
  
  for(; muonIt!=muonEnd; ++muonIt) {
    if (muon::isTightMuon((*muonIt), vertex)) {
      m_TightMuons.push_back(&(*muonIt));
    }
  }
  
  m_ControlHistos["NTightVsAll"]->Fill(muons->size(),m_TightMuons.size());
  
}


//_____________________________________________________________________
void L1TEfficiencyMuons_Offline::getProbeMuons(Handle<edm::TriggerResults> & trigResults,
					       edm::Handle<trigger::TriggerEvent> & trigEvent) {

  cout << "[L1TEfficiencyMuons_Offline:] getting probe muons" << endl;  

  m_ProbeMuons.clear();
  
  vector<const Muon*>::const_iterator probeCandIt   = m_TightMuons.begin();
  vector<const Muon*>::const_iterator tightMuonsEnd = m_TightMuons.end();

  for (; probeCandIt!=tightMuonsEnd; ++probeCandIt) {
    
    bool tagHasTrig = false;
    vector<const Muon*>::const_iterator tagCandIt  = m_TightMuons.begin();
    
    for (; tagCandIt!=tightMuonsEnd; ++tagCandIt) {
      if ((*tagCandIt) == (*probeCandIt)) continue; // CB has a little bias for closed-by muons
      tagHasTrig |= matchHlt(trigEvent,(*tagCandIt));
    }
    
    if (tagHasTrig) m_ProbeMuons.push_back((*probeCandIt));
    
  }      
      
  m_ControlHistos["NProbesVsTight"]->Fill(m_TightMuons.size(),m_ProbeMuons.size());
  
}

//_____________________________________________________________________
void L1TEfficiencyMuons_Offline::getMuonGmtPairs(edm::Handle<L1MuGMTReadoutCollection> & gmtCands) {

  m_MuonGmtPairs.clear();
  
  cout << "[L1TEfficiencyMuons_Offline:] Getting muon GMT pairs" << endl;  

  vector<const Muon*>::const_iterator probeMuIt  = m_ProbeMuons.begin();
  vector<const Muon*>::const_iterator probeMuEnd = m_ProbeMuons.end();

  vector<L1MuGMTExtendedCand> gmtContainer = gmtCands->getRecord(0).getGMTCands();
  
  vector<L1MuGMTExtendedCand>::const_iterator gmtIt;
  vector<L1MuGMTExtendedCand>::const_iterator gmtEnd = gmtContainer.end();
  
  for (; probeMuIt!=probeMuEnd; ++probeMuIt) {
    
    MuonGmtPair pairBestCand((*probeMuIt),0);
    pairBestCand.propagate(m_BField,m_propagatorAlong,m_propagatorOpposite);
    
    gmtIt = gmtContainer.begin();
    
    for(; gmtIt!=gmtEnd; ++gmtIt) {
      
      MuonGmtPair pairTmpCand((*probeMuIt),&(*gmtIt));
      pairTmpCand.propagate(m_BField,m_propagatorAlong,m_propagatorOpposite);

      if (pairTmpCand.dR() < m_MaxGmtMuonDR && pairTmpCand.gmtPt() > pairBestCand.gmtPt())
	pairBestCand = pairTmpCand;

    }
    
    m_MuonGmtPairs.push_back(pairBestCand);
  
    m_ControlHistos["MuonGmtDeltaR"]->Fill(pairBestCand.dR());

  }

}

//_____________________________________________________________________
bool L1TEfficiencyMuons_Offline::matchHlt(edm::Handle<TriggerEvent>  & triggerEvent, const Muon * mu) {


  double matchDeltaR = 9999;

  TriggerObjectCollection trigObjs = triggerEvent->getObjects();

  vector<int>::const_iterator trigIndexIt  = m_trigIndices.begin();
  vector<int>::const_iterator trigIndexEnd = m_trigIndices.end();
  
  for(; trigIndexIt!=trigIndexEnd; ++trigIndexIt) {

    const vector<string> moduleLabels(m_hltConfig.moduleLabels(*trigIndexIt));
    const unsigned moduleIndex = m_hltConfig.size((*trigIndexIt))-2;
    const unsigned hltFilterIndex = triggerEvent->filterIndex(InputTag(moduleLabels[moduleIndex],
								       "",m_trigProcess));
    
    if (hltFilterIndex < triggerEvent->sizeFilters()) {
      const Keys triggerKeys(triggerEvent->filterKeys(hltFilterIndex));
      const Vids triggerVids(triggerEvent->filterIds(hltFilterIndex));
      
      const unsigned nTriggers = triggerVids.size();
      for (size_t iTrig = 0; iTrig < nTriggers; ++iTrig) {
        const TriggerObject trigObject = trigObjs[triggerKeys[iTrig]];
	
        double dRtmp = deltaR((*mu),trigObject);
        if (dRtmp < matchDeltaR) matchDeltaR = dRtmp;
	
      }
    }
  }
  
  return (matchDeltaR < m_MaxHltMuonDR);

}


//define this as a plug-in
DEFINE_FWK_MODULE(L1TEfficiencyMuons_Offline);
