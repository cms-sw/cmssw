#include "JetMETCorrections/JetPlusTrack/plugins/JetPlusTrackCollisionAnalysis.h"

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Provenance/interface/Provenance.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "JetMETCorrections/Algorithms/interface/JetPlusTrackCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/JetReco/interface/JetID.h"

using namespace std;
namespace cms
{

//class MyJPT : public JetPlusTrackCorrector
//{
//  MyJPT( const edm::ParameterSet& );
//  getJet
//};

JetPlusTrackCollisionAnalysis::JetPlusTrackCollisionAnalysis(const edm::ParameterSet& iConfig)
  : jptCorrector_(0)
{

   mCone = iConfig.getParameter<double>("Cone");
//   mInputCaloTower = iConfig.getParameter<edm::InputTag>("src0");   
   mInputJetsCaloTower = iConfig.getParameter<edm::InputTag>("src1");
   mInputJetsZSPCorrected = iConfig.getParameter<edm::InputTag>("src2");   
   mInputJetsJPTCorrected = iConfig.getParameter<edm::InputTag>("src3");
   mJetsIDName            = iConfig.getParameter<std::string>("jetsID");

   m_inputTrackLabel = iConfig.getUntrackedParameter<std::string>("inputTrackLabel");

   hbhelabel_ = iConfig.getParameter<edm::InputTag>("HBHERecHitCollectionLabel");
   hbhelabelNZS_ = iConfig.getParameter<edm::InputTag>("HBHENZSRecHitCollectionLabel");  
 
   ecalLabels_=iConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs");
   
   fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile");
   
   allowMissingInputs_=iConfig.getUntrackedParameter<bool>("AllowMissingInputs",true);

   jptCorrectorName_ = iConfig.getUntrackedParameter<string>("JPTname");
	  
}

JetPlusTrackCollisionAnalysis::~JetPlusTrackCollisionAnalysis()
{
    cout<<" JetPlusTrack destructor "<<endl;
}

void JetPlusTrackCollisionAnalysis::beginJob()
{

   cout<<" Begin job "<<endl;

   hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;
   myTree = new TTree("JetPlusTrack","JetPlusTrack Tree");
   myTree->Branch("run",  &run, "run/I");
   myTree->Branch("event",  &event, "event/I");

   NumRecoJetsCaloTower = 0;
   NumRecoJetsZSPCorrected = 0;
   NumRecoJetsJPTCorrected = 0;
   NumRecoCone = 0;
   NumRecoTrack = 0;

// Jet Reco CaloTower
   myTree->Branch("NumRecoJetsCaloTower", &NumRecoJetsCaloTower, "NumRecoJetsCaloTower/I");
   myTree->Branch("JetRecoEtCaloTower",  JetRecoEtCaloTower, "JetRecoEtCaloTower[10]/F");
   myTree->Branch("JetRecoEtaCaloTower",  JetRecoEtaCaloTower, "JetRecoEtaCaloTower[10]/F");
   myTree->Branch("JetRecoPhiCaloTower",  JetRecoPhiCaloTower, "JetRecoPhiCaloTower[10]/F");
   myTree->Branch("JetRecoEtRecHit",  JetRecoEtRecHit, "JetRecoEtRecHit[10]/F");
   myTree->Branch("JetRecoEmf",  JetRecoEmf, "JetRecoEmf[10]/F");
   myTree->Branch("JetRecofHPD",  JetRecofHPD, "JetRecofHPD[10]/F");
   myTree->Branch("JetRecofRBX",  JetRecofRBX, "JetRecofRBX[10]/F");


   myTree->Branch("NumRecoCone", &NumRecoCone, "NumRecoCone/I");
   myTree->Branch("EcalEnergyCone",  EcalEnergyCone, "EcalEnergyCone[10]/F");
   myTree->Branch("HcalEnergyConeZSP", HcalEnergyConeZSP , "HcalEnergyConeZSP[10]/F");
   myTree->Branch("HcalEnergyConeNZSP", HcalEnergyConeNZSP , "HcalEnergyConeNZSP[10]/F");

// ZSP Corrected

   myTree->Branch("NumRecoJetsZSPCorrected", &NumRecoJetsZSPCorrected, "NumRecoJetsZSPCorrected/I");
   myTree->Branch("JetRecoEtZSPCorrected",  JetRecoEtZSPCorrected, "JetRecoEtZSPCorrected[10]/F");
   myTree->Branch("JetRecoEtaZSPCorrected",  JetRecoEtaZSPCorrected, "JetRecoEtaZSPCorrected[10]/F");
   myTree->Branch("JetRecoPhiZSPCorrected",  JetRecoPhiZSPCorrected, "JetRecoPhiZSPCorrected[10]/F");

// JPT Corrected

   myTree->Branch("NumRecoJetsJPTCorrected", &NumRecoJetsJPTCorrected, "NumRecoJetsJPTCorrected/I");
   myTree->Branch("JetRecoEtJPTCorrected",  JetRecoEtJPTCorrected, "JetRecoEtJPTCorrected[10]/F");
   myTree->Branch("JetRecoEtaJPTCorrected",  JetRecoEtaJPTCorrected, "JetRecoEtaJPTCorrected[10]/F");
   myTree->Branch("JetRecoPhiJPTCorrected",  JetRecoPhiJPTCorrected, "JetRecoPhiJPTCorrected[10]/F");

// Tracks block
   myTree->Branch("NumRecoTrack", &NumRecoTrack, "NumRecoTrack/I");
   myTree->Branch("TrackRecoEt",  TrackRecoEt, "TrackRecoEt[5000]/F");
   myTree->Branch("TrackRecoEta",  TrackRecoEta, "TrackRecoEta[5000]/F");
   myTree->Branch("TrackRecoPhi",  TrackRecoPhi, "TrackRecoPhi[5000]/F");

}

void JetPlusTrackCollisionAnalysis::endJob()
{

   cout << "===== Start writing user histograms =====" << endl;
   hOutputFile->SetCompressionLevel(2);
   hOutputFile->cd();
   myTree->Write();
   hOutputFile->Close() ;
   cout << "===== End writing user histograms =======" << endl;
   
}

void JetPlusTrackCollisionAnalysis::analyze(
                                         const edm::Event& iEvent,
                                         const edm::EventSetup& theEventSetup)  
{

  if ( !jptCorrector_) {
    const JetCorrector* corrector = JetCorrector::getJetCorrector(jptCorrectorName_,theEventSetup);
    
    if (!corrector) edm::LogError("JetPlusTrackCollisionAnalysis") << "Failed to get corrector with name " <<   
      jptCorrectorName_ << "from the EventSetup";
    jptCorrector_ = dynamic_cast<const JetPlusTrackCorrector*>(corrector);
    if (!jptCorrector_) edm::LogError("JetPlusTrackCollisionAnalysis") << "Corrector with name " << 
      jptCorrectorName_ << " is not a JetPlusTrackCorrector";
  }

    cout<<" JetPlusTrack analyze for Run "<<iEvent.id().run()<<" Event "
    <<iEvent.id().event()<<" Lumi block "<<iEvent.getLuminosityBlock().id().luminosityBlock()<<endl;

// Check for the proper bunch-crossing (51 or 2724). These numbers may change from run to run

    int bx = iEvent.bunchCrossing();
    if(bx != 51 && bx != 2724 ) {std::cout<<" Event with bad bunchcrossing? "<<bx<<std::endl;} 

    std::cout<<" Event with tecnical bits 40,41 and bx "<<bx<<std::endl;

// Check if Primary vertex exist

   edm::Handle<reco::VertexCollection> pvHandle; 
   iEvent.getByLabel("offlinePrimaryVertices",pvHandle);
   const reco::VertexCollection & vertices = *pvHandle.product();
   bool result = false;   

   for(reco::VertexCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it)
   {
      if(it->tracksSize() > 0 && 
         ( fabs(it->z()) <= 15. ) &&
         ( fabs(it->position().rho()) <= 2. )
       ) result = true;
   }

   if(!result) { std::cout<<" Vertex is outside 15 cms "<<std::endl; return;}

   std::cout<<" PV exists in the acceptable range (+-15 cm) and bx = "<<bx<<std::endl;  


// Calo Geometry
   edm::ESHandle<CaloGeometry> pG;
   theEventSetup.get<CaloGeometryRecord>().get(pG);
   const CaloGeometry* geo = pG.product();
/*   
  std::vector<edm::Provenance const*> theProvenance;
  iEvent.getAllProvenance(theProvenance);
  for( std::vector<edm::Provenance const*>::const_iterator ip = theProvenance.begin();
                                                      ip != theProvenance.end(); ip++)
  {
     cout<<" Print all module/label names "<<(**ip).moduleName()<<" "<<(**ip).moduleLabel()<<
     " "<<(**ip).productInstanceName()<<endl;
  }
*/
    


   run = iEvent.id().run();
   event = iEvent.id().event();

   cout<<" Run number "<<run<<" Event number "<<event<<endl;

// CaloJets

  edm::Handle<edm::ValueMap<reco::JetID> > jetsID;
  iEvent.getByLabel(mJetsIDName,jetsID);

    int ind=0;

    NumRecoJetsCaloTower = 0;

    edm::Handle<reco::CaloJetCollection> jets0;
    iEvent.getByLabel(mInputJetsCaloTower, jets0);
    if (!jets0.isValid()) {
      // can't find it!
      if (!allowMissingInputs_) {cout<<"CaloTowers are missed "<<endl; 
	*jets0;  // will throw the proper exception
      }
    } else {
      reco::CaloJetCollection::const_iterator jet = jets0->begin ();
      
      cout<<" Size of jets "<<jets0->size()<<endl;
      
      if(jets0->size() > 0 )
	{
	  for (; jet != jets0->end (); jet++)
	    {
	      
	      if( NumRecoJetsCaloTower < 10 )
		{
               edm::RefToBase<reco::Jet> jetRef(edm::Ref<reco::CaloJetCollection>(jets0,ind));

//                jpt::MatchedTracks pions;
//                jpt::MatchedTracks muons;
//                jpt::MatchedTracks electrons;
//  const bool ok = jptCorrector_->matchTracks(*jet,iEvent,theEventSetup,pions,muons,electrons);

//  jpt::JetTracks trk;
//  const bool ok1 = jptCorrector_->jetTrackAssociation((*jet),iEvent,theEventSetup,trk);
			  
		  JetRecoEtCaloTower[NumRecoJetsCaloTower] = (*jet).et();
		  JetRecoEtaCaloTower[NumRecoJetsCaloTower] = (*jet).eta();
		  JetRecoPhiCaloTower[NumRecoJetsCaloTower] = (*jet).phi();
                  JetRecoEmf[NumRecoJetsCaloTower] = (*jet).emEnergyFraction();
                  JetRecofHPD[NumRecoJetsCaloTower] = (*jetsID)[jetRef].fHPD;
                  JetRecofRBX[NumRecoJetsCaloTower] = (*jetsID)[jetRef].fRBX;

		  NumRecoJetsCaloTower++;
		  cout<<" Raw et "<<(*jet).et()<<" Eta "<<(*jet).eta()<<" Phi "<<(*jet).phi()<<endl;
                  ind++ ;
		} // CaloJets <10
	    } // Calojets cycle
	} // jets collection non-empty
    } // valid collection

//    if(NumRecoJetsCaloTower == 0) return; 

// ZSP correction
     NumRecoJetsZSPCorrected = 0;

     edm::Handle<reco::CaloJetCollection> jets1;
     iEvent.getByLabel(mInputJetsZSPCorrected, jets1);
     if (!jets1.isValid()) {
       // can't find it!
       if (!allowMissingInputs_) {cout<<"ZSP corrected are missed "<<endl; 
	 *jets1;  // will throw the proper exception
       }
     } else {
       reco::CaloJetCollection::const_iterator jet = jets1->begin ();

       cout<<" Size of ZSP jets "<<jets1->size()<<endl;
       if(jets1->size() > 0 )
	 {
	   for (; jet != jets1->end (); jet++)
	     {
	       if( NumRecoJetsZSPCorrected < 10 )
		 {
                jpt::MatchedTracks pions;
                jpt::MatchedTracks muons;
                jpt::MatchedTracks electrons;
  const bool ok = jptCorrector_->matchTracks(*jet,iEvent,theEventSetup,pions,muons,electrons);
       if(ok) {

           std::cout<<" The number of pions, muon, electrons "<<pions.inVertexInCalo_.size()<<std::endl;

        }
		   JetRecoEtZSPCorrected[NumRecoJetsZSPCorrected] = (*jet).et();
		   JetRecoEtaZSPCorrected[NumRecoJetsZSPCorrected] = (*jet).eta();
		   JetRecoPhiZSPCorrected[NumRecoJetsZSPCorrected] = (*jet).phi();
		   cout<<" ZSP et "<<(*jet).et()<<" Eta "<<(*jet).eta()<<" Phi "<<(*jet).phi()<<endl;
		   NumRecoJetsZSPCorrected++;
		 } // ZSPjets <10
	     } // ZSP jets cycle
	 } // jets non empty
     } // collection valid



// JPT correction
     NumRecoJetsJPTCorrected = 0;

     edm::Handle<reco::CaloJetCollection> jets2;
     iEvent.getByLabel(mInputJetsJPTCorrected, jets2);
     if (!jets2.isValid()) {
       // can't find it!
       if (!allowMissingInputs_) {cout<<"JPT corrected are missed "<<endl; 
	 *jets2;  // will throw the proper exception
       }
     } else {
       reco::CaloJetCollection::const_iterator jet = jets2->begin ();

       cout<<" Size of JPT jets "<<jets2->size()<<endl;
       if(jets2->size() > 0 )
	 {
	   for (; jet != jets2->end (); jet++)
	     {
	       if( NumRecoJetsJPTCorrected < 10 )
		 { 
//                   jpt::JetTracks trk;
		   JetRecoEtJPTCorrected[NumRecoJetsJPTCorrected] = (*jet).et();
		   JetRecoEtaJPTCorrected[NumRecoJetsJPTCorrected] = (*jet).eta();
		   JetRecoPhiJPTCorrected[NumRecoJetsJPTCorrected] = (*jet).phi();

		   cout<<" JPT et "<<(*jet).et()<<" Eta "<<(*jet).eta()<<" Phi "<<(*jet).phi()<<endl;

//                   const bool ok = jptCorrector_->jetTrackAssociation((*jet),iEvent,theEventSetup,trk);
/*
                   if( ok ) {
                     reco::TrackRefVector TrkatV = trk.vertex_;
                     std::cout<<" Jet "<<NumRecoJetsJPTCorrected<<" Number of associated tracks "
                                       <<TrkatV.size()<<std::endl;
                   }
*/
		   NumRecoJetsJPTCorrected++;
		 } //jets <10 
	     } // JPTjets cycle
	 }// jets non empty
     }// collection valid


// CaloTowers from RecHits
// Load EcalRecHits
// for jets and for particular axis: eta = 0.1, phi = 0.1


   std::vector<edm::InputTag>::const_iterator i;
   int iecal = 0;
    
   for(int jjj=0; jjj<NumRecoJetsCaloTower; jjj++)
   {
    JetRecoEtRecHit[jjj] = 0.;
    for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) {
     
      edm::Handle<EcalRecHitCollection> ec;
      iEvent.getByLabel(*i,ec);

      if (!ec.isValid()) {

	if (!allowMissingInputs_) {cout<<" Ecal rechits are missed "<<endl; 
	  *ec;  // will throw the proper exception
	} // missing input

      } else {
	// EcalBarrel = 1, EcalEndcap = 2
	for(EcalRecHitCollection::const_iterator recHit = (*ec).begin();
	    recHit != (*ec).end(); ++recHit)
	  {	    
	    GlobalPoint pos = geo->getPosition(recHit->detid());
	    double deta = pos.eta() - JetRecoEtaCaloTower[jjj];
	    double dphi = fabs(pos.phi() - JetRecoPhiCaloTower[jjj]); 
	    if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	    double dr = sqrt(dphi*dphi + deta*deta);

	    if(dr<mCone)
	      {
                 JetRecoEtRecHit[jjj] = JetRecoEtRecHit[jjj] + (*recHit).energy();
	      } // jet cone
	  } // ecal rechits
      } // collection valid
      iecal++;
    } // ecal labels
   } // calo jets

// Hcal Barrel and endcap for isolation
 
   {
   edm::Handle<HBHERecHitCollection> hbhe;
   iEvent.getByLabel(hbhelabel_,hbhe);
   if (!hbhe.isValid()) {
     // can't find it!
     cout<<" Exception in hbhe "<<endl;
     if (!allowMissingInputs_) {
       *hbhe;  // will throw the proper exception
     }
   } else {
     for(int jjj=0; jjj<NumRecoJetsCaloTower; jjj++)
       {	 
	 for(HBHERecHitCollection::const_iterator hbheItr = (*hbhe).begin();
	     hbheItr != (*hbhe).end(); ++hbheItr)
	   {
	     DetId id = (hbheItr)->detid();
	     GlobalPoint pos = geo->getPosition(hbheItr->detid());
	     double deta = pos.eta() - JetRecoEtaCaloTower[jjj];
	     double dphi = fabs(pos.phi() - JetRecoPhiCaloTower[jjj]);
	     if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	     double dr = sqrt(dphi*dphi + deta*deta);
	     
	     if(dr<mCone)
	       {
		 JetRecoEtRecHit[jjj] = JetRecoEtRecHit[jjj] + (*hbheItr).energy();
	       } // jet cone

	   } // hcal rechits
       } // jetcalotower
   } // valid collection
   } // end of block

// For particular direction
    iecal = 0;
     double empty_jet_energy_ecal = 0.; 
    for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) {
      edm::Handle<EcalRecHitCollection> ec;
      iEvent.getByLabel(*i,ec);
      if (!ec.isValid()) {
	// can't find it!
	if (!allowMissingInputs_) {cout<<" Ecal rechits are missed "<<endl; 
	  *ec;  // will throw the proper exception
	}
      } else {
	// EcalBarrel = 1, EcalEndcap = 2
	for(EcalRecHitCollection::const_iterator recHit = (*ec).begin();
	    recHit != (*ec).end(); ++recHit)
	  {
	     DetId id = (recHit)->detid();
	     GlobalPoint pos = geo->getPosition(recHit->detid());	   
            double deta = pos.eta() - 0.1;
	    double dphi_empty = fabs(pos.phi()+4.*atan(1.) - 0.1);
	    if(dphi_empty > 4.*atan(1.)) dphi_empty = 8.*atan(1.) - dphi_empty;
	    double dr_empty = sqrt(dphi_empty*dphi_empty + deta*deta);
	    	   
	    if(dr_empty<mCone)
	      {
		empty_jet_energy_ecal = empty_jet_energy_ecal + (*recHit).energy();         
	      } // cone
	  } // rechit
      } // valid
      iecal++;
      } // ecal collection


   double empty_jet_energy_hcal_ZS = 0.;
  
   edm::Handle<HBHERecHitCollection> hbhe;
   iEvent.getByLabel(hbhelabel_,hbhe);
   if (!hbhe.isValid()) {
     // can't find it!
     cout<<" Exception in hbhe "<<endl;
     if (!allowMissingInputs_) {
       *hbhe;  // will throw the proper exception
     }
   } else {
	 
	 for(HBHERecHitCollection::const_iterator hbheItr = (*hbhe).begin();
	     hbheItr != (*hbhe).end(); ++hbheItr)
	   {
	     DetId id = (hbheItr)->detid();
	     GlobalPoint pos = geo->getPosition(hbheItr->detid());
            double deta = pos.eta() - 0.1;
	    double dphi_empty = fabs(pos.phi()+4.*atan(1.) - 0.1);
	    if(dphi_empty > 4.*atan(1.)) dphi_empty = 8.*atan(1.) - dphi_empty;
	    double dr_empty = sqrt(dphi_empty*dphi_empty + deta*deta);	     
	     if(dr_empty<mCone)
	       {
		 empty_jet_energy_hcal_ZS = empty_jet_energy_hcal_ZS + (*hbheItr).energy();
	       } // cone
	   } // hbhe

   } // valid collection 

   double empty_jet_energy_hcal_NZS = 0.;
 
   edm::Handle<HBHERecHitCollection> hbhek;
   iEvent.getByLabel(hbhelabelNZS_,hbhek);
   if (!hbhek.isValid()) {
     // can't find it!
     cout<<" Exception in NZS hbhe "<<endl;
     if (!allowMissingInputs_) {
       *hbhek;  // will throw the proper exception
     }
   } else {
	 
	 for(HBHERecHitCollection::const_iterator hbheItr = (*hbhek).begin();
	     hbheItr != (*hbhek).end(); ++hbheItr)
	   {
	     DetId id = (hbheItr)->detid();
	     GlobalPoint pos = geo->getPosition(hbheItr->detid());
            double deta = pos.eta() - 0.1;
	    double dphi_empty = fabs(pos.phi()+4.*atan(1.) - 0.1);
	    if(dphi_empty > 4.*atan(1.)) dphi_empty = 8.*atan(1.) - dphi_empty;
	    double dr_empty = sqrt(dphi_empty*dphi_empty + deta*deta);	     
	     if(dr_empty<mCone)
	       {
		 empty_jet_energy_hcal_NZS = empty_jet_energy_hcal_NZS + (*hbheItr).energy();
	       }// cone
	   } // hbhe

   } // valid

       EcalEnergyCone[0] = empty_jet_energy_ecal;
       HcalEnergyConeZSP[0] = empty_jet_energy_hcal_ZS;
       HcalEnergyConeNZSP[0] = empty_jet_energy_hcal_NZS;

       cout<<" Ecal energy in cone "<<EcalEnergyCone[0]<<" "<<HcalEnergyConeZSP[0]<<" "<<
       HcalEnergyConeNZSP[0]<<endl;

// Tracker

    edm::Handle<reco::TrackCollection> tracks;
    iEvent.getByLabel(m_inputTrackLabel, tracks);

    reco::TrackCollection::const_iterator trk;
    int iTracks = 0;
    for ( trk = tracks->begin(); trk != tracks->end(); ++trk){
      TrackRecoEt[iTracks] = trk->pt();
      TrackRecoEta[iTracks] = trk->eta();
      TrackRecoPhi[iTracks] = trk->phi();
      iTracks++;
    }
    NumRecoTrack = iTracks;
    cout<<" Number of tracks "<<NumRecoTrack<<endl;

   myTree->Fill();
   
}
} // namespace cms

// define this class as a plugin
#include "FWCore/Framework/interface/MakerMacros.h"
using namespace cms;
DEFINE_FWK_MODULE(JetPlusTrackCollisionAnalysis);
