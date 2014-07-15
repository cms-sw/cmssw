// user include files
#include "GammaJetAnalysis.h" 

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h" 
/* #include "FWCore/Framework/interface/MakerMacros.h" */
#include "FWCore/Framework/interface/ESHandle.h" 
#include "FWCore/Framework/interface/EventSetup.h" 

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h" 

#include "Geometry/Records/interface/CaloGeometryRecord.h" 
#include "Geometry/CaloGeometry/interface/CaloGeometry.h" 
#include "DataFormats/GeometryVector/interface/GlobalPoint.h" 
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h" 
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h" 
// #include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h" 
// #include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h" 
#include "DataFormats/JetReco/interface/CaloJetCollection.h" 
#include "DataFormats/JetReco/interface/CaloJet.h" 
#include "DataFormats/JetReco/interface/GenJetCollection.h"
//#include "DataFormats/JetReco/interface/GenJetfwd.h"
#include "DataFormats/JetReco/interface/GenJet.h"

//#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"
#include "TFile.h"
#include "TTree.h"


using namespace std;
namespace cms
{
GammaJetAnalysis::GammaJetAnalysis(const edm::ParameterSet& iConfig)
{
  // get names of modules, producing object collections

  nameProd_ = iConfig.getUntrackedParameter<std::string>("nameProd");
  jetCalo_ = iConfig.getUntrackedParameter<std::string>("jetCalo","GammaJetJetBackToBackCollection");

  tok_jets_ = consumes<reco::CaloJetCollection>( edm::InputTag(nameProd_,
	jetCalo_) );

  gammaClus_ = iConfig.getUntrackedParameter<std::string>("gammaClus","GammaJetGammaBackToBackCollection");

  tok_egamma_ = consumes<reco::SuperClusterCollection>( edm::InputTag(nameProd_,
	gammaClus_) );

  ecalInput_=iConfig.getUntrackedParameter<std::string>("ecalInput","GammaJetEcalRecHitCollection");

  tok_ecal_ = consumes<EcalRecHitCollection>( edm::InputTag(nameProd_, ecalInput_) );

  hbheInput_ = iConfig.getUntrackedParameter<std::string>("hbheInput");

  tok_hbhe_ = consumes<HBHERecHitCollection>(edm::InputTag(nameProd_,hbheInput_));

  hoInput_ = iConfig.getUntrackedParameter<std::string>("hoInput");

  tok_ho_ = consumes<HORecHitCollection>(edm::InputTag(nameProd_,hoInput_));

  hfInput_ = iConfig.getUntrackedParameter<std::string>("hfInput");

  tok_hf_ = consumes<HFRecHitCollection>(edm::InputTag(nameProd_,hfInput_));

  Tracks_ = iConfig.getUntrackedParameter<std::string>("Tracks","GammaJetTracksCollection");
  CutOnEgammaEnergy_  = iConfig.getParameter<double>("CutOnEgammaEnergy");

  myName = iConfig.getParameter<std::string> ("textout");
  useMC = iConfig.getParameter<bool>("useMCInfo"); 
  allowMissingInputs_=iConfig.getUntrackedParameter<bool>("AllowMissingInputs",false);
  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile"); 
  risol[0] = 0.5;
  risol[1] = 0.7;
  risol[2] = 1.0;
  
  ecut[0][0] = 0.09;
  ecut[0][1] = 0.18;
  ecut[0][2] = 0.27;
  
  ecut[1][0] = 0.45;
  ecut[1][1] = 0.9;
  ecut[1][2] = 1.35;
  
  ecut[2][0] = 0.5;
  ecut[2][1] = 1.;
  ecut[2][2] = 1.5;
}


GammaJetAnalysis::~GammaJetAnalysis()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void GammaJetAnalysis::beginJob()
{
   hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ; 
   myTree = new TTree("GammaJet","GammaJet Tree");
   myTree->Branch("run",  &run, "run/I");
   myTree->Branch("event",  &event, "event/I");
   
   NumRecoJets = 0;
   NumGenJets = 0;
   NumRecoGamma = 0;
   NumRecoTrack = 0;
   NumPart = 0;
// Jet block
   myTree->Branch("NumRecoJets", &NumRecoJets, "NumRecoJets/I");   
   myTree->Branch("JetRecoEt",  JetRecoEt, "JetRecoEt[10]/F");
   myTree->Branch("JetRecoEta",  JetRecoEta, "JetRecoEta[10]/F");
   myTree->Branch("JetRecoPhi",  JetRecoPhi, "JetRecoPhi[10]/F");
   myTree->Branch("JetRecoType",  JetRecoType, "JetRecoType[10]/F");
   
// Gamma block for ECAL isolated gammas
   myTree->Branch("NumRecoGamma", &NumRecoGamma, "NumRecoGamma/I");
   myTree->Branch("EcalClusDet", &EcalClusDet, "EcalClusDet[20]/I");
   myTree->Branch("GammaRecoEt",  GammaRecoEt, "GammaRecoEt[20]/F");
   myTree->Branch("GammaRecoEta",  GammaRecoEta, "GammaRecoEta[20]/F");
   myTree->Branch("GammaRecoPhi",  GammaRecoPhi, "GammaRecoPhi[20]/F");
   myTree->Branch("GammaIsoEcal",  GammaIsoEcal, "GammaIsoEcal[9][20]/F");

// Tracks block
   myTree->Branch("NumRecoTrack", &NumRecoTrack, "NumRecoTrack/I");
   myTree->Branch("TrackRecoEt",  TrackRecoEt, "TrackRecoEt[200]/F");
   myTree->Branch("TrackRecoEta",  TrackRecoEta, "TrackRecoEta[200]/F");
   myTree->Branch("TrackRecoPhi",  TrackRecoPhi, "TrackRecoPhi[200]/F");

// end of tree declaration

//   edm::ESHandle<CaloGeometry> pG;
//   iSetup.get<CaloGeometryRecord>().get(pG);
//   geo = pG.product();

  myout_part = new std::ofstream((myName+"_part.dat").c_str()); 
  if(!myout_part) cout << " Output file not open!!! "<<endl;
  myout_hcal = new std::ofstream((myName+"_hcal.dat").c_str()); 
  if(!myout_hcal) cout << " Output file not open!!! "<<endl;
  myout_ecal = new std::ofstream((myName+"_ecal.dat").c_str()); 
  if(!myout_ecal) cout << " Output file not open!!! "<<endl;
  
  myout_jet = new std::ofstream((myName+"_jet.dat").c_str()); 
  if(!myout_jet) cout << " Output file not open!!! "<<endl;
  myout_photon = new std::ofstream((myName+"_photon.dat").c_str()); 
  if(!myout_photon) cout << " Output file not open!!! "<<endl;
   
   
}

void GammaJetAnalysis::endJob()
{

   cout << "===== Start writing user histograms =====" << endl;
   hOutputFile->SetCompressionLevel(2);
   hOutputFile->cd();
   myTree->Write();
   hOutputFile->Close() ;
   cout << "===== End writing user histograms =======" << endl; 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GammaJetAnalysis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);
   geo = pG.product();


  using namespace edm;
  std::vector<Provenance const*> theProvenance;
  iEvent.getAllProvenance(theProvenance);
//  for( std::vector<Provenance const*>::const_iterator ip = theProvenance.begin();
//                                                      ip != theProvenance.end(); ip++)
//  {
//     cout<<" Print all module/label names "<<(**ip).moduleName()<<" "<<(**ip).moduleLabel()<<
//     " "<<(**ip).productInstanceName()<<endl;
//  }
// Load generator information
// write HEPEVT block into file
   run = iEvent.id().run();
   event = iEvent.id().event();
  (*myout_part)<<"Event "<<iEvent.id().run()<<" "<<iEvent.id().event()<<endl;
  (*myout_jet)<<"Event "<<iEvent.id().run()<<" "<<iEvent.id().event()<<endl;
  (*myout_hcal)<<"Event "<<iEvent.id().run()<<" "<<iEvent.id().event()<<endl;
  (*myout_ecal)<<"Event "<<iEvent.id().run()<<" "<<iEvent.id().event()<<endl;
  (*myout_photon)<<"Event "<<iEvent.id().run()<<" "<<iEvent.id().event()<<endl;
  

    std::vector<edm::InputTag>::const_iterator ic;
    int jettype = 0;
    int jetexist = -100;
    int reco = 1;
    double etlost = -100.1;
    
    NumRecoJets = 0;
    
     try {
       
       edm::Handle<reco::CaloJetCollection> jets;
       iEvent.getByToken(tok_jets_, jets);
       reco::CaloJetCollection::const_iterator jet = jets->begin ();
       cout<<" Size of Calo jets "<<jets->size()<<endl;
       jettype++;
       
       if(jets->size() > 0 )
       {
         int ij = 0;
         for (; jet != jets->end (); jet++) 
         {
            cout<<" Jet et "<<(*jet).et()<<" "<<(*jet).eta()<<" "<<(*jet).phi()<<endl;
	    ij++;
	    if(ij<4) (*myout_jet)<<jettype<<" "<<reco<<" "<<ij<<" "<<(*jet).et()<<" "<<(*jet).eta()<<" "<<(*jet).phi()
	    <<" "<<iEvent.id().event()<<endl;
	    jetexist = ij;
	    if( NumRecoJets < 8 )
	    {
	     JetRecoEt[NumRecoJets] = (*jet).et();
	     JetRecoEta[NumRecoJets] = (*jet).eta();
	     JetRecoPhi[NumRecoJets] = (*jet).phi();
	     JetRecoType[NumRecoJets] = jettype;
	     NumRecoJets++;
	    }
         }
       }
     } catch (cms::Exception& e) { // can't find it!
       if (!allowMissingInputs_) {
         cout<< " Calojets are missed "<<endl;
         throw e;
        } 	 
     }
   
     cout<<" We filled CaloJet part "<<jetexist<<endl;
     
     if( jetexist < 0 ) (*myout_jet)<<jetexist<<" "<<reco<<" "<<etlost
                         <<" "<<etlost<<" "<<etlost
			 <<" "<<iEvent.id().event()<<endl;
// Load EcalRecHits
  
    std::vector<edm::InputTag>::const_iterator i;
    vector<std::pair<DetId, double> > theRecHits;
      
    try {
      
      edm::Handle<EcalRecHitCollection> ec;
      iEvent.getByToken(tok_ecal_,ec);
      
       for(EcalRecHitCollection::const_iterator recHit = (*ec).begin();
                                                recHit != (*ec).end(); ++recHit)
       {
// EcalBarrel = 1, EcalEndcap = 2

	 GlobalPoint pos = geo->getPosition(recHit->detid());
         theRecHits.push_back(std::pair<DetId, double>(recHit->detid(), recHit->energy()));

	 if( (*recHit).energy()> ecut[recHit->detid().subdetId()-1][0] )
                    (*myout_ecal)<<recHit->detid().subdetId()<<" "<<(*recHit).energy()<<" "<<pos.phi()<<" "<<pos.eta()
	                         <<" "<<iEvent.id().event()<<endl;
	     
       } 
      
    } catch (cms::Exception& e) { // can't find it!
    if (!allowMissingInputs_) {
      cout<<" Ecal collection is missed "<<endl;
      throw e;
     } 
    }

     cout<<" Fill EcalRecHits "<<endl;
//  cout<<" Start to get hbhe "<<endl;
// Hcal Barrel and endcap for isolation
    try {
      edm::Handle<HBHERecHitCollection> hbhe;
      iEvent.getByToken(tok_hbhe_,hbhe);

//      (*myout_hcal)<<(*hbhe).size()<<endl;
  for(HBHERecHitCollection::const_iterator hbheItr = (*hbhe).begin();

      hbheItr != (*hbhe).end(); ++hbheItr)
      {
        DetId id = (hbheItr)->detid();
        GlobalPoint pos = geo->getPosition(hbheItr->detid());
        (*myout_hcal)<<id.subdetId()<<" "<<(*hbheItr).energy()<<" "<<pos.phi()<<
                                      " "<<pos.eta()<<" "<<iEvent.id().event()<<endl;
        theRecHits.push_back(std::pair<DetId, double>(hbheItr->detid(), hbheItr->energy()));

      }
    } catch (cms::Exception& e) { // can't find it!
      if (!allowMissingInputs_) {
        cout<<" HBHE collection is missed "<<endl;
        throw e;
      }
    }

 
 for(int i = 0; i<9; i++)
 {
    for(int j= 0; j<10; j++) GammaIsoEcal[i][j] = 0.;
 }

// Load Ecal clusters
 jetexist = -100; 
 int barrel = 1;
 NumRecoGamma = 0;
 
 try {
 int ij = 0;
  // Get island super clusters after energy correction
  Handle<reco::SuperClusterCollection> eclus;
  iEvent.getByToken(tok_egamma_, eclus);
  const reco::SuperClusterCollection* correctedSuperClusters=eclus.product();
  // loop over the super clusters and fill the histogram
  for(reco::SuperClusterCollection::const_iterator aClus = correctedSuperClusters->begin();
                                                           aClus != correctedSuperClusters->end(); aClus++) {
    double vet = aClus->energy()/cosh(aClus->eta());
    cout<<" Supercluster " << ij<<" Et "<< vet <<" energy "<<aClus->energy()<<" eta "<<aClus->eta()<<" Cut "<<CutOnEgammaEnergy_<<endl;

    if(vet>CutOnEgammaEnergy_) {
      ij++;
      float gammaiso_ecal[9] = {0.,0.,0.,0.,0.,0.,0.,0.,0.};
     for(vector<std::pair<DetId, double> >::const_iterator it = theRecHits.begin(); it != theRecHits.end(); it++)
      {
           GlobalPoint pos = geo->getPosition(it->first);
           double eta = pos.eta();
	   double phi = pos.phi();
	   double deta = fabs(eta-aClus->eta());
	   double dphi = fabs(phi-aClus->phi());
	   if(dphi>4.*atan(1.)) dphi = 8.*atan(1.)-dphi;
	   double dr = sqrt(deta*deta+dphi*dphi);
	   
	   double rmin = 0.07;
	   if( fabs(aClus->eta()) > 1.47 ) rmin = 0.07*(fabs(aClus->eta())-.47)*1.2;
	   if( fabs(aClus->eta()) > 2.2 ) rmin = 0.07*(fabs(aClus->eta())-.47)*1.4;
	   
	   int itype_ecal = 0;
	   double ecutn = 0.;
	   for (int i = 0; i<3; i++)
	   {
	     for (int j = 0; j<3; j++)
	     {
	     
	        if(it->first.det() == DetId::Ecal ) 
		{
		  if(it->first.subdetId() == 1) ecutn = ecut[0][j];
		  if(it->first.subdetId() == 2) ecutn = ecut[1][j];
		  if( dr>rmin && dr<risol[i])
		  {
		   if((*it).second > ecutn) gammaiso_ecal[itype_ecal] = gammaiso_ecal[itype_ecal]+(*it).second/cosh(eta);
		  } 
		}
		
		if(it->first.det() == DetId::Hcal ) 
		{
		   ecutn = ecut[2][j];
		   if( dr>rmin && dr<risol[i])
		   {
		     if((*it).first > ecutn) 
		     {
		        gammaiso_ecal[itype_ecal] = gammaiso_ecal[itype_ecal]+(*it).second/cosh(eta);
		     }
		   }
		} 
		jetexist = ij;
	        itype_ecal++;
		
	     } // Ecal
	   } // cycle on iso radii      
      } // cycle on rechits
      
      
// Fill Tree      
	   if( NumRecoGamma < 10 ) 
	   {
	    for (int ii = 0; ii<9 ; ii++)
	    {
	     GammaIsoEcal[ii][NumRecoGamma] = gammaiso_ecal[ii]; 
	    } 
             EcalClusDet[NumRecoGamma] = 1;
             GammaRecoEt[NumRecoGamma] = vet;
             GammaRecoEta[NumRecoGamma] = aClus->eta();
             GammaRecoPhi[NumRecoGamma] = aClus->phi();
	     NumRecoGamma++;
	    }
    (*myout_photon)<<ij<<" "<<barrel<<" "<<vet<<" "<<aClus->eta()<<" "<<aClus->phi()<<" "<<iEvent.id().event()<<endl;
    (*myout_photon)<<ij<<" "<<gammaiso_ecal[0]<<" "<<gammaiso_ecal[1] <<" "<<gammaiso_ecal[2]<<" "<<gammaiso_ecal[3]
                   <<" "<<gammaiso_ecal[4]<<" "<<gammaiso_ecal[5]<<" "<<gammaiso_ecal[6]<<" "<<gammaiso_ecal[7]<<" "<<gammaiso_ecal[8]<<endl;
      
       jetexist = ij;
    } //vet  
  } // number of superclusters
  } catch (cms::Exception& e) { // can't find it!
    if (!allowMissingInputs_) {
       cout<<" Ecal barrel clusters are missed "<<endl;
       throw e;
    }   
  }

    cout<<" After iso cuts "<<jetexist<<endl;

    double ecluslost = -100.1;
    if(jetexist<0) (*myout_photon)<<jetexist<<" "<<barrel<<" "<<ecluslost<<" "<<ecluslost
                                  <<" "<<ecluslost<<" "<<iEvent.id().event()<<endl;
  
    cout<<" Event is ready "<<endl;
   
    myTree->Fill();
   
} // analyze method
} // namespace cms
