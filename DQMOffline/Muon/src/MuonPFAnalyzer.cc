/** \class MuRecoAnalyzer
 *
 *  DQM monitoring source for PF muons
 *
 *  $Date: 2013/06/17 09:17:17 $
 *  $Revision: 1.2 $
 *  \author C. Battilana - CIEMAT
 */
//Base class
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMOffline/Muon/interface/MuonPFAnalyzer.h"

#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/Math/interface/deltaR.h"

//System included files
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>

//Root included files
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"


//Event framework included files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace edm;
using namespace std;
using namespace reco;


MuonPFAnalyzer::MuonPFAnalyzer(const ParameterSet& pSet)
			       
{

  LogTrace("MuonPFAnalyzer") << 
    "[MuonPFAnalyzer] Initializing configuration from parameterset.\n";

  theGenLabel      = pSet.getParameter<InputTag>("inputTagGenParticles");  
  theRecoLabel     = pSet.getParameter<InputTag>("inputTagMuonReco");  
  theBeamSpotLabel = pSet.getParameter<InputTag>("inputTagBeamSpot");  
  theVertexLabel   = pSet.getParameter<InputTag>("inputTagVertex");  
  
  theHighPtTh   = pSet.getParameter<double>("highPtThreshold");
  theRecoGenR   = pSet.getParameter<double>("recoGenDeltaR");
  theIsoCut     = pSet.getParameter<double>("relCombIsoCut");
  theRunOnMC    = pSet.getParameter<bool>("runOnMC");

  theFolder = pSet.getParameter<string>("folder");
  
  theMuonKinds.push_back("");          // all TUNEP/PF muons
  theMuonKinds.push_back("Tight");     // tight TUNEP/PF muons
  theMuonKinds.push_back("TightIso");  // tight/iso TUNEP/PF muons
 

}


MuonPFAnalyzer::~MuonPFAnalyzer() 
{

  LogTrace("MuonPFAnalyzer") << 
    "[MuonPFAnalyzer] Destructor called.\n";
  
}



// ------------ method called when starting to processes a run  ------------

void MuonPFAnalyzer::beginRun(edm::Run const &, edm::EventSetup const &) {

  LogTrace("MuonPFAnalyzer") << 
    "[MuonPFAnalyzer] Booking histograms.\n";

  //Set up DAQ
  theDbe = 0;
  theDbe = edm::Service<DQMStore>().operator->();
  theDbe->cd();

  if(theRunOnMC)
    {
      bookHistos("PF");
      bookHistos("PFTight");
      bookHistos("PFTightIso");
      bookHistos("TUNEP");
      bookHistos("TUNEPTight");
      bookHistos("TUNEPTightIso");
    }
 
    bookHistos("PFvsTUNEP");
    bookHistos("PFvsTUNEPTight");
    bookHistos("PFvsTUNEPTightIso");
  

}

void MuonPFAnalyzer::analyze(const Event& event, 
			     const EventSetup& context) 
{
  
  Handle<reco::MuonCollection> muons;
  event.getByLabel(theRecoLabel, muons);

  Handle<GenParticleCollection> genMuons;
  event.getByLabel(theGenLabel, genMuons);

  Handle<BeamSpot> beamSpot;
  event.getByLabel(theBeamSpotLabel, beamSpot);

  Handle<VertexCollection> vertex;
  event.getByLabel(theVertexLabel, vertex);

  const Vertex primaryVertex = getPrimaryVertex(vertex, beamSpot);

  recoToGenMatch(muons, genMuons);
 
  RecoGenCollection::const_iterator recoGenIt  = theRecoGen.begin();
  RecoGenCollection::const_iterator recoGenEnd = theRecoGen.end();
    
  for (;recoGenIt!=recoGenEnd;++recoGenIt) 
    {
    
      const Muon *muon = recoGenIt->first;
      TrackRef tunePTrack = muon->tunePMuonBestTrack();

     const GenParticle *genMuon = recoGenIt->second;
	  
      vector<string>::const_iterator kindIt  = theMuonKinds.begin();
      vector<string>::const_iterator kindEnd = theMuonKinds.end();

      for (;kindIt!=kindEnd;++kindIt) 
	{ 

	  const string & kind = (*kindIt);

	  if (kind.find("Tight") != string::npos && 
	      !muon::isTightMuon((*muon), primaryVertex)) continue;
	  
	  if (kind.find("Iso") != string::npos &&
	      combRelIso(muon) > theIsoCut) continue;
	  
	  if (theRunOnMC && genMuon && !muon->innerTrack().isNull() ) // has matched gen muon
	    {

	      if (!tunePTrack.isNull())
		{ 

		  string group = "TUNEP" + kind;

		  float pt  = tunePTrack->pt();
		  float phi = tunePTrack->phi();
		  float eta = tunePTrack->eta();

		  float genPt  = genMuon->pt();
		  float genPhi = genMuon->p4().phi();
		  float genEta = genMuon->p4().eta();

		  float dPtOverPt = (pt / genPt) - 1;
		  
		  if (pt < theHighPtTh)
		    {

		      fillInRange(getPlot(group,"code"),1,muonTrackType(muon, false));
		      fillInRange(getPlot(group,"deltaPtOverPt"),1,dPtOverPt);
		    }
		  else
		    {
		      fillInRange(getPlot(group,"codeHighPt"),1,muonTrackType(muon, false));
		      fillInRange(getPlot(group,"deltaPtOverPtHighPt"),1,dPtOverPt);
		    }
    
		  fillInRange(getPlot(group,"deltaPt"),1,(pt - genPt));
		  fillInRange(getPlot(group,"deltaPhi"),1,fDeltaPhi(genPhi,phi));
		  fillInRange(getPlot(group,"deltaEta"),1,genEta - eta);
	  
		}
	      
	      if (muon->isPFMuon()) 
		{
		  
		  string group = "PF" + kind;
		  
		  // Assumes that default in muon is PF
		  float pt  = muon->pt();
		  float phi = muon->p4().phi();
		  float eta = muon->p4().eta();
		  
		  float genPt  = genMuon->pt();
		  float genPhi = genMuon->p4().phi();
		  float genEta = genMuon->p4().eta();

		  float dPtOverPt = (pt / genPt) - 1;
		  
		  if (pt < theHighPtTh)
		    {
		      fillInRange(getPlot(group,"code"),1,muonTrackType(muon, true));
		      fillInRange(getPlot(group,"deltaPtOverPt"),1,dPtOverPt);
		    }
		  else
		    { 
		      fillInRange(getPlot(group,"codeHighPt"),1,muonTrackType(muon, true));
		      fillInRange(getPlot(group,"deltaPtOverPtHighPt"),1,dPtOverPt);
		    }
		  
		  
		  fillInRange(getPlot(group,"deltaPt"),1,pt - genPt);
		  fillInRange(getPlot(group,"deltaPhi"),1,fDeltaPhi(genPhi,phi));
		  fillInRange(getPlot(group,"deltaEta"),1,genEta - eta);
		  
		}
	    
	    }

	

	    if (muon->isPFMuon() && !tunePTrack.isNull() &&          
		!muon->innerTrack().isNull()) // Compare PF with TuneP + Tracker 
	      {                               // No gen matching needed
	
	      string group = "PFvsTUNEP" + kind;

	      float pt  = tunePTrack->pt();
	      float phi = tunePTrack->phi();
	      float eta = tunePTrack->eta();
	      
	      // Assumes that default in muon is PF
	      float pfPt  = muon->pt();
	      float pfPhi = muon->p4().phi();
	      float pfEta = muon->p4().eta();
	      float dPtOverPt = (pfPt / pt) - 1; // TUNEP vs PF pt used as denum.
	     

	      if (pt < theHighPtTh) 
		{
		  fillInRange(getPlot(group,"code"),2,
			      muonTrackType(muon, false),muonTrackType(muon, true));
		  fillInRange(getPlot(group,"deltaPtOverPt"),1,dPtOverPt);
		}
	      else 
		{
		  fillInRange(getPlot(group,"codeHighPt"),
			      2,muonTrackType(muon, false),muonTrackType(muon, true));
		  fillInRange(getPlot(group,"deltaPtOverPtHighPt"),1,dPtOverPt);
		}
	      
	      fillInRange(getPlot(group,"deltaPt"),1,pfPt - pt);
	      fillInRange(getPlot(group,"deltaPhi"),1,fDeltaPhi(pfPhi,phi));
	      fillInRange(getPlot(group,"deltaEta"),1,pfEta - eta);
	      

	      if (theRunOnMC && genMuon) // has a matched gen muon
		
		{
		  
		  float genPt     = genMuon->pt();
		  float dPtOverPtGen = (pt / genPt) - 1;
		  float dPtOverPtGenPF = (pfPt / genPt) - 1;
		  
		  if (pt < theHighPtTh) 
		    {
		      fillInRange(getPlot(group,"deltaPtOverPtPFvsTUNEP"),
				  2,dPtOverPtGen,dPtOverPtGenPF);
		    }
		  else 
		    {
		      fillInRange(getPlot(group,"deltaPtOverPtHighPtPFvsTUNEP"),
				  2,dPtOverPtGen,dPtOverPtGenPF);
		    }
		}		  
	      
	      }
	    
	}
      
    }

}




void MuonPFAnalyzer::bookHistos(const string & group) { 

  

  LogTrace("MuonPFAnalyzer") << "[MuonPFAnalyzer] Booking histos for group :"
			     << group << "\n";

  theDbe->setCurrentFolder(string(theFolder) + group);
 

    bool isPFvsTUNEP = group.find("PFvsTUNEP") != string::npos;
    
    string hName;
    
      
    hName  = "deltaPtOverPt" + group;
    thePlots[group]["deltaPtOverPt"] = theDbe->book1D(hName.c_str(),hName.c_str(),101,-1.01,1.01);
    
    hName = "deltaPtOverPtHighPt" + group;
    thePlots[group]["deltaPtOverPtHighPt"] = theDbe->book1D(hName.c_str(),hName.c_str(),101,-1.01,1.01);
    
    hName = "deltaPt" + group;
    thePlots[group]["deltaPt"] = theDbe->book1D(hName.c_str(),hName.c_str(),201.,-10.25,10.25);
    
    hName = "deltaPhi"+group;
    thePlots[group]["deltaPhi"] = theDbe->book1D(hName.c_str(),hName.c_str(),51.,0,.0102);
    
    hName = "deltaEta"+group;
    thePlots[group]["deltaEta"] = theDbe->book1D(hName.c_str(),hName.c_str(),101.,-.00505,.00505);
    


    if (isPFvsTUNEP) {

     
      hName = "code"+group;
      MonitorElement * plot = theDbe->book2D(hName.c_str(),hName.c_str(),7,-.5,6.5,7,-.5,6.5);
      thePlots[group]["code"] = plot;
      setCodeLabels(plot,1);
      setCodeLabels(plot,2);
      
      hName = "codeHighPt"+group;
      plot = theDbe->book2D(hName.c_str(),hName.c_str(),7,-.5,6.5,7,-.5,6.5);
      thePlots[group]["codeHighPt"] = plot; 
      setCodeLabels(plot,1);
      setCodeLabels(plot,2);
    

      if (theRunOnMC)
	{	
	  hName = "deltaPtOverPtPFvsTUNEP" + group;
	  thePlots[group]["deltaPtOverPtPFvsTUNEP"] =  
	    theDbe->book2D(hName.c_str(),hName.c_str(),
			   101,-1.01,1.01,101,-1.01,1.01);

	  hName = "deltaPtOverPtHighPtPFvsTUNEP" + group;
	  thePlots[group]["deltaPtOverPtHighPtPFvsTUNEP"] =  
	    theDbe->book2D(hName.c_str(),hName.c_str(),
			   101,-1.01,1.01,101,-1.01,1.01);
	}
    } else {
      hName = "code"+group;
      MonitorElement * plot = theDbe->book1D(hName.c_str(),hName.c_str(),7,-.5,6.5);
      thePlots[group]["code"] = plot;
      setCodeLabels(plot,1);

      hName = "codeHighPt"+group;
      plot = theDbe->book1D(hName.c_str(),hName.c_str(),7,-.5,6.5);
      thePlots[group]["codeHighPt"] = plot;  
      setCodeLabels(plot,1);
    }
  
}


MonitorElement * MuonPFAnalyzer::getPlot(const string & group,
					 const string & type) {

  map<string,map<string,MonitorElement *> >::iterator groupIt = thePlots.find(group);
  if (groupIt == thePlots.end()) {
    LogTrace("MuonPFAnalyzer") << "[MuonPFAnalyzer] GROUP : " << group 
			       << " is not a valid plot group. Returning 0.\n";
    return 0;
  }
  
  map<string,MonitorElement *>::iterator typeIt = groupIt->second.find(type);
  if (typeIt == groupIt->second.end()) {
    LogTrace("MuonPFAnalyzer") << "[MuonPFAnalyzer] TYPE : " << type 
			       << " is not a valid type for GROUP : " << group 
			       << ". Returning 0.\n";
    return 0;
  }
  
  return typeIt->second;

} 


inline float MuonPFAnalyzer::combRelIso(const reco::Muon * muon)
{
  
  MuonIsolation iso = muon->isolationR03();
  float combRelIso = (iso.emEt + iso.hadEt + iso.sumPt) / muon->pt();  

  return combRelIso;
  
}


inline float MuonPFAnalyzer::fDeltaPhi(float phi1, float phi2) {
  
  float fPhiDiff = fabs(acos(cos(phi1-phi2)));
  return fPhiDiff;
  
}


void MuonPFAnalyzer::setCodeLabels(MonitorElement *plot, int nAxis) 
{

  TAxis *axis = 0;
  
  TH1 * histo = plot->getTH1();
  if(!histo) return;
  
  if (nAxis==1) 
    axis =histo->GetXaxis();
  else if (nAxis == 2)
    axis =histo->GetYaxis();

  if(!axis) return;

  axis->SetBinLabel(1,"Inner Track");
  axis->SetBinLabel(2,"Outer Track");
  axis->SetBinLabel(3,"Combined");
  axis->SetBinLabel(4,"TPFMS");
  axis->SetBinLabel(5,"Picky");
  axis->SetBinLabel(6,"DYT");
  axis->SetBinLabel(7,"None");

}


void MuonPFAnalyzer::fillInRange(MonitorElement *plot, int nAxis, double x, double y) 
{

  TH1 * histo =  plot->getTH1();
  
  TAxis *axis[2] = {0, 0};
  axis[0] = histo->GetXaxis();
  if (nAxis == 2)
    axis[1] = histo->GetYaxis();

  double value[2] = {0, 0};
  value[0] = x;
  value[1] = y;

  for (int i = 0;i<nAxis;++i)
    {
      double min = axis[i]->GetXmin();
      double max = axis[i]->GetXmax();

      if (value[i] <= min)
	value[i] = axis[i]->GetBinCenter(1);

      if (value[i] >= max)
	value[i] = axis[i]->GetBinCenter(axis[i]->GetNbins());
    }

  if (nAxis == 2)
    plot->Fill(value[0],value[1]);
  else
    plot->Fill(value[0]);
  
}


int MuonPFAnalyzer::muonTrackType(const Muon * muon, bool usePF) {

  switch ( usePF ? muon->muonBestTrackType() : muon->tunePMuonBestTrackType() ) {
  case Muon::InnerTrack :
    return 0;
  case Muon::OuterTrack :
    return 1;
  case Muon::CombinedTrack :
    return 2;
  case Muon::TPFMS :
    return 3;
  case Muon::Picky :
    return 4;
  case Muon::DYT :
    return 5;
  case Muon::None :
    return 6;
  }

  return 6;

}


void MuonPFAnalyzer::recoToGenMatch( Handle<MuonCollection>        & muons, 
				     Handle<GenParticleCollection> & gens ) 
{

  theRecoGen.clear();  

  if (muons.isValid())
    {

      MuonCollection::const_iterator muonIt  = muons->begin();
      MuonCollection::const_iterator muonEnd = muons->end();

      for(; muonIt!=muonEnd; ++muonIt) 
	{
      
	  float bestDR = 999.;
	  const GenParticle *bestGen = 0;

	  if (theRunOnMC && gens.isValid()) 
	    {
	  
	      GenParticleCollection::const_iterator genIt  = gens->begin();
	      GenParticleCollection::const_iterator genEnd = gens->end();
      
	      for(; genIt!=genEnd; ++genIt) 
		{
	  
		  if (abs(genIt->pdgId()) == 13 ) 
		    {
		  
		      float muonPhi = muonIt->phi();
		      float muonEta = muonIt->eta();
		  
		      float genPhi = genIt->phi();
		      float genEta = genIt->eta();
		  
		      float dR = deltaR(muonEta,muonPhi,
					genEta,genPhi);
		  
		      if (dR < theRecoGenR && dR < bestDR) 
			{
			  bestDR = dR;
			  bestGen = &(*genIt);
			}
		      
		    }	
		  
		}
	    }
      
	  theRecoGen.push_back(RecoGenPair(&(*muonIt), bestGen));

	}
    }
  
}

const reco::Vertex MuonPFAnalyzer::getPrimaryVertex( Handle<VertexCollection> &vertex,
						     Handle<BeamSpot> &beamSpot ) 
{

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

    LogInfo("MuonPFAnalyzer") << 
      "[MuonPFAnalyzer] PrimaryVertex not found, use BeamSpot position instead.\n";

    posVtx = beamSpot->position();
    errVtx(0,0) = beamSpot->BeamWidthX();
    errVtx(1,1) = beamSpot->BeamWidthY();
    errVtx(2,2) = beamSpot->sigmaZ();
    
  }

  const Vertex primaryVertex(posVtx,errVtx);

  return primaryVertex;

}


//define this as a plug-in
DEFINE_FWK_MODULE(MuonPFAnalyzer);
