#include "DQMServices/Examples/interface/DQMExample_Step1.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


// Geometry
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TLorentzVector.h"


#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>

//
// -------------------------------------- Constructor --------------------------------------------
//
DQMExample_Step1::DQMExample_Step1(const edm::ParameterSet& ps)
{
  edm::LogInfo("DQMExample_Step1") <<  "Constructor  DQMExample_Step1::DQMExample_Step1 " << std::endl;
  
  // Get parameters from configuration file
  theElectronCollection_   = consumes<reco::GsfElectronCollection>(ps.getParameter<edm::InputTag>("electronCollection"));
  theCaloJetCollection_    = consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
  thePfMETCollection_      = consumes<reco::PFMETCollection>(ps.getParameter<edm::InputTag>("pfMETCollection"));
  theConversionCollection_ = consumes<reco::ConversionCollection>(ps.getParameter<edm::InputTag>("conversionsCollection"));
  thePVCollection_         = consumes<reco::VertexCollection>(ps.getParameter<edm::InputTag>("PVCollection"));
  theBSCollection_         = consumes<reco::BeamSpot>(ps.getParameter<edm::InputTag>("beamSpotCollection"));
  triggerEvent_            = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("TriggerEvent"));
  triggerResults_          = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  triggerFilter_           = ps.getParameter<edm::InputTag>("TriggerFilter");
  triggerPath_             = ps.getParameter<std::string>("TriggerPath");


  // cuts:
  ptThrL1_ = ps.getUntrackedParameter<double>("PtThrL1");
  ptThrL2_ = ps.getUntrackedParameter<double>("PtThrL2");
  ptThrJet_ = ps.getUntrackedParameter<double>("PtThrJet");
  ptThrMet_ = ps.getUntrackedParameter<double>("PtThrMet");
 
}

//
// -- Destructor
//
DQMExample_Step1::~DQMExample_Step1()
{
  edm::LogInfo("DQMExample_Step1") <<  "Destructor DQMExample_Step1::~DQMExample_Step1 " << std::endl;
}

//
// -------------------------------------- beginRun --------------------------------------------
//
void DQMExample_Step1::dqmBeginRun(edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::beginRun" << std::endl;
}
//
// -------------------------------------- bookHistos --------------------------------------------
//
void DQMExample_Step1::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::bookHistograms" << std::endl;
  
  //book at beginRun
  bookHistos(ibooker_);
}
//
// -------------------------------------- Analyze --------------------------------------------
//
void DQMExample_Step1::analyze(edm::Event const& e, edm::EventSetup const& eSetup)
{
  edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::analyze" << std::endl;


  //-------------------------------
  //--- Vertex Info
  //-------------------------------
  edm::Handle<reco::VertexCollection> vertexHandle;
  e.getByToken(thePVCollection_, vertexHandle);
  if ( !vertexHandle.isValid() ) 
    {
      edm::LogError ("DQMClientExample") << "invalid collection: vertex" << "\n";
      return;
    }
  
  int vertex_number = vertexHandle->size();
  reco::VertexCollection::const_iterator v = vertexHandle->begin();

  math::XYZPoint PVPoint(-999, -999, -999);
  if(vertex_number != 0)
    PVPoint = math::XYZPoint(v->position().x(), v->position().y(), v->position().z());
  
  PVPoint_=PVPoint;

  //-------------------------------
  //--- MET
  //-------------------------------
  edm::Handle<reco::PFMETCollection> pfMETCollection;
  e.getByToken(thePfMETCollection_, pfMETCollection);
  if ( !pfMETCollection.isValid() )    
    {
      edm::LogError ("DQMClientExample") << "invalid collection: MET" << "\n";
      return;
    }
  //-------------------------------
  //--- Electrons
  //-------------------------------
  edm::Handle<reco::GsfElectronCollection> electronCollection;
  e.getByToken(theElectronCollection_, electronCollection);
  if ( !electronCollection.isValid() )
    {
      edm::LogError ("DQMClientExample") << "invalid collection: electrons" << "\n";
      return;
    }

  float nEle=0;
  int posEle=0, negEle=0;
  const reco::GsfElectron* ele1 = nullptr;
  const reco::GsfElectron* ele2 = nullptr;
  for (reco::GsfElectronCollection::const_iterator recoElectron=electronCollection->begin(); recoElectron!=electronCollection->end(); ++recoElectron)
    {
      //decreasing pT
      if( MediumEle(e,eSetup,*recoElectron) )
	{
	  if(!ele1 && recoElectron->pt() > ptThrL1_)
	    ele1 = &(*recoElectron);
	  
	  else if(!ele2 && recoElectron->pt() > ptThrL2_)
	    ele2 = &(*recoElectron);

	}
      
      if(recoElectron->charge()==1)
	posEle++;
      else if(recoElectron->charge()==-1)
	negEle++;

    } // end of loop over electrons
  
  nEle = posEle+negEle;
  
  //-------------------------------
  //--- Jets
  //-------------------------------
  edm::Handle<reco::CaloJetCollection> caloJetCollection;
  e.getByToken (theCaloJetCollection_,caloJetCollection);
  if ( !caloJetCollection.isValid() ) 
    {
      edm::LogError ("DQMClientExample") << "invalid collection: jets" << "\n";
      return;
    }

  int   nJet = 0;
  const reco::CaloJet* jet1 = nullptr;
  const reco::CaloJet* jet2 = nullptr;
  
  for (reco::CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin(); i_calojet != caloJetCollection->end(); ++i_calojet) 
    {
      //remove jet-ele matching
      if(ele1)
	if (Distance(*i_calojet,*ele1) < 0.3) continue;
      
      if(ele2)
	if (Distance(*i_calojet,*ele2) < 0.3) continue;
      
      if (i_calojet->pt() < ptThrJet_) continue;

      nJet++;
      
      if (!jet1) 
	jet1 = &(*i_calojet);
      
      else if (!jet2)
	jet2 = &(*i_calojet);
    }
  
  // ---------------------------
  // ---- Analyze Trigger Event
  // ---------------------------

  //check what is in the menu
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_,hltresults);
  
  if(!hltresults.isValid())
    {
      edm::LogError ("DQMClientExample") << "invalid collection: TriggerResults" << "\n";
      return;
    }
  
  bool hasFired = false;
  const edm::TriggerNames& trigNames = e.triggerNames(*hltresults);
  unsigned int numTriggers = trigNames.size();
  
  for( unsigned int hltIndex=0; hltIndex<numTriggers; ++hltIndex )
    {
      if (trigNames.triggerName(hltIndex)==triggerPath_ &&  hltresults->wasrun(hltIndex) &&  hltresults->accept(hltIndex))
	hasFired = true;
    }
  


  //access the trigger event
  edm::Handle<trigger::TriggerEvent> triggerEvent;
  e.getByToken(triggerEvent_, triggerEvent);
  if( triggerEvent.failedToGet() )
    {
      edm::LogError ("DQMClientExample") << "invalid collection: TriggerEvent" << "\n";
      return;
    }


  reco::Particle* ele1_HLT = nullptr;
  int nEle_HLT = 0;

  size_t filterIndex = triggerEvent->filterIndex( triggerFilter_ );
  trigger::TriggerObjectCollection triggerObjects = triggerEvent->getObjects();
  if( !(filterIndex >= triggerEvent->sizeFilters()) )
    {
      const trigger::Keys& keys = triggerEvent->filterKeys( filterIndex );
      std::vector<reco::Particle> triggeredEle;
      
      for( size_t j = 0; j < keys.size(); ++j ) 
	{
	  trigger::TriggerObject foundObject = triggerObjects[keys[j]];
	  if( abs( foundObject.particle().pdgId() ) != 11 )  continue; //make sure that it is an electron
	  
	  triggeredEle.push_back( foundObject.particle() );
	  ++nEle_HLT;
	}
      
      if( !triggeredEle.empty() ) 
	ele1_HLT = &(triggeredEle.at(0));
    }

  //-------------------------------
  //--- Fill the histos
  //-------------------------------

  //vertex
  h_vertex_number -> Fill( vertex_number );

  //met
  h_pfMet -> Fill( pfMETCollection->begin()->et() );

  //multiplicities
  h_eMultiplicity->Fill(nEle);       
  h_jMultiplicity->Fill(nJet);
  h_eMultiplicity_HLT->Fill(nEle_HLT);

  //leading not matched
  if(ele1)
    {
      h_ePt_leading->Fill(ele1->pt());
      h_eEta_leading->Fill(ele1->eta());
      h_ePhi_leading->Fill(ele1->phi());
    }
  if(ele1_HLT)
    {
      h_ePt_leading_HLT->Fill(ele1_HLT->pt());
      h_eEta_leading_HLT->Fill(ele1_HLT->eta());
      h_ePhi_leading_HLT->Fill(ele1_HLT->phi());
    }
  //leading Jet
  if(jet1)
    {
      h_jPt_leading->Fill(jet1->pt());
      h_jEta_leading->Fill(jet1->eta());
      h_jPhi_leading->Fill(jet1->phi());
    }


  //fill only when the trigger candidate mathes with the reco one
  if( ele1 && ele1_HLT && deltaR(*ele1_HLT,*ele1) < 0.3 && hasFired==true )
    {
      h_ePt_leading_matched->Fill(ele1->pt());
      h_eEta_leading_matched->Fill(ele1->eta());
      h_ePhi_leading_matched->Fill(ele1->phi());
      
      h_ePt_leading_HLT_matched->Fill(ele1_HLT->pt());
      h_eEta_leading_HLT_matched->Fill(ele1_HLT->eta());
      h_ePhi_leading_HLT_matched->Fill(ele1_HLT->phi());

      h_ePt_diff->Fill(ele1->pt()-ele1_HLT->pt());
    }
}

//
// -------------------------------------- endRun --------------------------------------------
//
void DQMExample_Step1::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::endRun" << std::endl;
}


//
// -------------------------------------- book histograms --------------------------------------------
//
void DQMExample_Step1::bookHistos(DQMStore::IBooker & ibooker_)
{
  ibooker_.cd();
  ibooker_.setCurrentFolder("Physics/TopTest");

  h_vertex_number = ibooker_.book1D("Vertex_number", "Number of event vertices in collection", 40,-0.5,   39.5 );

  h_pfMet        = ibooker_.book1D("pfMet",        "Pf Missing E_{T}; GeV"          , 20,  0.0 , 100);

  h_eMultiplicity = ibooker_.book1D("NElectrons","# of electrons per event",10,0.,10.);
  h_ePt_leading_matched = ibooker_.book1D("ElePt_leading_matched","Pt of leading electron",50,0.,100.);
  h_eEta_leading_matched = ibooker_.book1D("EleEta_leading_matched","Eta of leading electron",50,-5.,5.);
  h_ePhi_leading_matched = ibooker_.book1D("ElePhi_leading_matched","Phi of leading electron",50,-3.5,3.5);

  h_ePt_leading = ibooker_.book1D("ElePt_leading","Pt of leading electron",50,0.,100.);
  h_eEta_leading = ibooker_.book1D("EleEta_leading","Eta of leading electron",50,-5.,5.);
  h_ePhi_leading = ibooker_.book1D("ElePhi_leading","Phi of leading electron",50,-3.5,3.5);

  h_jMultiplicity = ibooker_.book1D("NJets","# of electrons per event",10,0.,10.);
  h_jPt_leading = ibooker_.book1D("JetPt_leading","Pt of leading Jet",150,0.,300.);
  h_jEta_leading = ibooker_.book1D("JetEta_leading","Eta of leading Jet",50,-5.,5.);
  h_jPhi_leading = ibooker_.book1D("JetPhi_leading","Phi of leading Jet",50,-3.5,3.5);

  h_eMultiplicity_HLT = ibooker_.book1D("NElectrons_HLT","# of electrons per event @HLT",10,0.,10.);
  h_ePt_leading_HLT = ibooker_.book1D("ElePt_leading_HLT","Pt of leading electron @HLT",50,0.,100.);
  h_eEta_leading_HLT = ibooker_.book1D("EleEta_leading_HLT","Eta of leading electron @HLT",50,-5.,5.);
  h_ePhi_leading_HLT = ibooker_.book1D("ElePhi_leading_HLT","Phi of leading electron @HLT",50,-3.5,3.5);

  h_ePt_leading_HLT_matched = ibooker_.book1D("ElePt_leading_HLT_matched","Pt of leading electron @HLT",50,0.,100.);
  h_eEta_leading_HLT_matched = ibooker_.book1D("EleEta_leading_HLT_matched","Eta of leading electron @HLT",50,-5.,5.);
  h_ePhi_leading_HLT_matched = ibooker_.book1D("ElePhi_leading_HLT_matched","Phi of leading electron @HLT",50,-3.5,3.5);

  h_ePt_diff = ibooker_.book1D("ElePt_diff_matched","pT(RECO) - pT(HLT) for mathed candidates",100,-10,10.);

  ibooker_.cd();  

}


//
// -------------------------------------- functions --------------------------------------------
//
double DQMExample_Step1::Distance( const reco::Candidate & c1, const reco::Candidate & c2 ) {
        return  deltaR(c1,c2);
}

double DQMExample_Step1::DistancePhi( const reco::Candidate & c1, const reco::Candidate & c2 ) {
        return  deltaPhi(c1.p4().phi(),c2.p4().phi());
}

// This always returns only a positive deltaPhi
double DQMExample_Step1::calcDeltaPhi(double phi1, double phi2) {
  double deltaPhi = phi1 - phi2;
  if (deltaPhi < 0) deltaPhi = -deltaPhi;
  if (deltaPhi > 3.1415926) {
    deltaPhi = 2 * 3.1415926 - deltaPhi;
  }
  return deltaPhi;
}

//
// -------------------------------------- electronID --------------------------------------------
//
bool DQMExample_Step1::MediumEle (const edm::Event & iEvent, const edm::EventSetup & iESetup, const reco::GsfElectron & electron)
{
    
  //********* CONVERSION TOOLS
  edm::Handle<reco::ConversionCollection> conversions_h;
  iEvent.getByToken(theConversionCollection_, conversions_h);
  
  bool isMediumEle = false; 
  
  float pt = electron.pt();
  float eta = electron.eta();
    
  int isEB            = electron.isEB();
  float sigmaIetaIeta = electron.sigmaIetaIeta();
  float DetaIn        = electron.deltaEtaSuperClusterTrackAtVtx();
  float DphiIn        = electron.deltaPhiSuperClusterTrackAtVtx();
  float HOverE        = electron.hadronicOverEm();
  float ooemoop       = (1.0/electron.ecalEnergy() - electron.eSuperClusterOverP()/electron.ecalEnergy());
  
  int mishits             = electron.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
  int nAmbiguousGsfTracks = electron.ambiguousGsfTracksSize();
  
  reco::GsfTrackRef eleTrack  = electron.gsfTrack() ;
  float dxy           = eleTrack->dxy(PVPoint_);  
  float dz            = eleTrack->dz (PVPoint_);
  
  edm::Handle<reco::BeamSpot> BSHandle;
  iEvent.getByToken(theBSCollection_, BSHandle);
  const reco::BeamSpot BS = *BSHandle;
  
  bool isConverted = ConversionTools::hasMatchedConversion(electron, *conversions_h, BS.position());
  
  // default
  if(  (pt > 12.) && (fabs(eta) < 2.5) &&
       ( ( (isEB == 1) && (fabs(DetaIn)  < 0.004) ) || ( (isEB == 0) && (fabs(DetaIn)  < 0.007) ) ) &&
       ( ( (isEB == 1) && (fabs(DphiIn)  < 0.060) ) || ( (isEB == 0) && (fabs(DphiIn)  < 0.030) ) ) &&
       ( ( (isEB == 1) && (sigmaIetaIeta < 0.010) ) || ( (isEB == 0) && (sigmaIetaIeta < 0.030) ) ) &&
       ( ( (isEB == 1) && (HOverE        < 0.120) ) || ( (isEB == 0) && (HOverE        < 0.100) ) ) &&
       ( ( (isEB == 1) && (fabs(ooemoop) < 0.050) ) || ( (isEB == 0) && (fabs(ooemoop) < 0.050) ) ) &&
       ( ( (isEB == 1) && (fabs(dxy)     < 0.020) ) || ( (isEB == 0) && (fabs(dxy)     < 0.020) ) ) &&
       ( ( (isEB == 1) && (fabs(dz)      < 0.100) ) || ( (isEB == 0) && (fabs(dz)      < 0.100) ) ) &&
       ( ( (isEB == 1) && (!isConverted) ) || ( (isEB == 0) && (!isConverted) ) ) &&
       ( mishits == 0 ) &&
       ( nAmbiguousGsfTracks == 0 )      
       )
    isMediumEle=true;
  
  return isMediumEle;
}
