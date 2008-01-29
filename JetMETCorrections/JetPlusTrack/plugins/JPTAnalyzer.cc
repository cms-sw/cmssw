// system include files
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
// /CMSSW/Calibration/HcalAlCaRecoProducers/src/AlCaIsoTracksProducer.cc  track propagator
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// MC info
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "CLHEP/HepPDT/DefaultConfig.hh"
//
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "DataFormats/Math/interface/deltaR.h"
//double dR = deltaR( c1, c2 );
//
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
//jets
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetfwd.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetfwd.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
//
// muons and tracks
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
// ecal
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
// candidates
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
//
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
//
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include <vector>

using namespace std;
using namespace reco;

//
// class decleration
//

class JPTAnalyzer : public edm::EDAnalyzer {
   public:
      explicit JPTAnalyzer(const edm::ParameterSet&);
      ~JPTAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
  // output root file
  string fOutputFileName ;
  // names of modules, producing object collections
  string calojetsSrc;
  string genjetsSrc;
  //
  // MC jet energy corrections
  string JetCorrectionMCJ;
  // ZSP jet energy corrections
  string JetCorrectionZSP;
  // Jet+tracks energy corrections
  string JetCorrectionJPT;
  // variables to store in ntpl
  double  EtaGen1, PhiGen1, EtaRaw1, PhiRaw1, EtGen1, EtRaw1, EtMCJ1, EtZSP1, EtJPT1, DRMAXgjet1;
  double  EtaGen2, PhiGen2, EtaRaw2, PhiRaw2, EtGen2, EtRaw2, EtMCJ2, EtZSP2, EtJPT2, DRMAXgjet2;
  // output root file and tree
  TFile*      hOutputFile ;
  TTree*      t1;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

// ------------ method called once each job just before starting event loop  ------------
void 
JPTAnalyzer::beginJob(const edm::EventSetup&)
{
  using namespace edm;
  // creating a simple tree

  hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;

  t1 = new TTree("t1","analysis tree");

  t1->Branch("EtaGen1",&EtaGen1,"EtaGen1/D");
  t1->Branch("PhiGen1",&PhiGen1,"PhiGen1/D");
  t1->Branch("EtaRaw1",&EtaRaw1,"EtaRaw1/D");
  t1->Branch("PhiRaw1",&PhiRaw1,"PhiRaw1/D");
  t1->Branch("EtGen1",&EtGen1,"EtGen1/D");
  t1->Branch("EtRaw1",&EtRaw1,"EtRaw1/D");
  t1->Branch("EtMCJ1",&EtMCJ1,"EtMCJ1/D");
  t1->Branch("EtZSP1",&EtZSP1,"EtZSP1/D");
  t1->Branch("EtJPT1",&EtJPT1,"EtJPT1/D");
  t1->Branch("DRMAXgjet1",&DRMAXgjet1,"DRMAXgjet1/D");

  t1->Branch("EtaGen2",&EtaGen2,"EtaGen2/D");
  t1->Branch("PhiGen2",&PhiGen2,"PhiGen2/D");
  t1->Branch("EtaRaw2",&EtaRaw2,"EtaRaw2/D");
  t1->Branch("PhiRaw2",&PhiRaw2,"PhiRaw2/D");
  t1->Branch("EtGen2",&EtGen2,"EtGen2/D");
  t1->Branch("EtRaw2",&EtRaw2,"EtRaw2/D");
  t1->Branch("EtMCJ2",&EtMCJ2,"EtMCJ2/D");
  t1->Branch("EtZSP2",&EtZSP2,"EtZSP2/D");
  t1->Branch("EtJPT2",&EtJPT2,"EtJPT2/D");
  t1->Branch("DRMAXgjet2",&DRMAXgjet2,"DRMAXgjet2/D");

  return ;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
JPTAnalyzer::endJob() {

  hOutputFile->Write() ;
  hOutputFile->Close() ;
  
  return ;
}

//
// constructors and destructor
//
JPTAnalyzer::JPTAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  using namespace edm;
  // 
  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile");
  //
  // get names of input object collections
  calojetsSrc   = iConfig.getParameter< std::string > ("calojets");
  genjetsSrc    = iConfig.getParameter< std::string > ("genjets");
  //
  // MC jet energy corrections
  JetCorrectionMCJ = iConfig.getParameter< std::string > ("JetCorrectionMCJ");
  // ZSP jet energy corrections
  JetCorrectionZSP = iConfig.getParameter< std::string > ("JetCorrectionZSP");
  // Jet+tracks energy corrections
  JetCorrectionJPT = iConfig.getParameter< std::string > ("JetCorrectionJPT");
}


JPTAnalyzer::~JPTAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
JPTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // initialize vector containing two highest Et gen jets > 20 GeV
   // in this example they are checked not to be leptons from Z->ll decay (DR match)
   vector<HepLorentzVector> gjets;
   gjets.clear();

   // initialize tree variables
   EtaGen1 = 0.;
   PhiGen1 = 0.;
   EtaRaw1 = 0.;
   PhiRaw1 = 0.;
   EtGen1  = 0.;
   EtRaw1  = 0.;
   EtMCJ1  = 0.;
   EtZSP1  = 0.;
   EtJPT1  = 0.;
   DRMAXgjet1 = 1000.;

   EtaGen2 = 0.;
   PhiGen2 = 0.;
   EtaRaw2 = 0.;
   PhiRaw2 = 0.;
   EtGen2  = 0.;
   EtRaw2  = 0.;
   EtMCJ2  = 0.;
   EtZSP2  = 0.;
   EtJPT2  = 0.;
   DRMAXgjet2 = 1000.;

   //   edm::ESHandle<CaloGeometry> geometry;
   //   iSetup.get<IdealGeometryRecord>().get(geometry);


   // get MC info
   edm::Handle<HepMCProduct> EvtHandle ;
   iEvent.getByLabel( "source", EvtHandle ) ;
   //  iEvent.getByLabel( "VtxSmeared", EvtHandle ) ;

   // l1 and l2 are leptons from Z->ll to be checked they are not gen jets (DR match)
   HepLorentzVector l1;
   HepLorentzVector l2;
   const HepMC::GenEvent* evt = EvtHandle->GetEvent() ;
   ESHandle<ParticleDataTable> pdt;
   iSetup.getData( pdt );
   for ( HepMC::GenEvent::particle_const_iterator p = evt->particles_begin();
	 p != evt->particles_end(); ++p ) {
     /*
     cout <<" status : " << (*p)->status() 
	  <<" pid = " << (*p)->pdg_id() 
	  <<" px = " << (*p)->momentum().px()
	  <<" py = " << (*p)->momentum().py() 
	  <<" charge = " << (pdt->particle((*p)->pdg_id()))->charge()
	  <<" charge3 = " << (pdt->particle((*p)->pdg_id()))->ID().threeCharge() << endl;
     */
     if((*p)->status() == 3 && (*p)->pdg_id() == 23) {
       //Z 
       /*
       cout <<" Z status : " << (*p)->status() 
	    <<" pid = " << (*p)->pdg_id() 
	    <<" px = " << (*p)->momentum().px()
	    <<" py = " << (*p)->momentum().py() 
	    <<" charge = " << (pdt->particle((*p)->pdg_id()))->charge()
	    <<" charge3 = " << (pdt->particle((*p)->pdg_id()))->ID().threeCharge() << endl;
       */
       // l1 doc lines
       ++p;
       /*
       cout <<" l1 status : " << (*p)->status() 
	    <<" pid = " << (*p)->pdg_id() 
	    <<" px = " << (*p)->momentum().px()
	    <<" py = " << (*p)->momentum().py() 
	    <<" pz = " << (*p)->momentum().pz() 
	    <<" e = " << (*p)->momentum().e() 
	    <<" charge = " << (pdt->particle((*p)->pdg_id()))->charge()
	    <<" charge3 = " << (pdt->particle((*p)->pdg_id()))->ID().threeCharge() << endl;
       */
       HepLorentzVector l1c((*p)->momentum().px(),(*p)->momentum().py(),(*p)->momentum().pz(),(*p)->momentum().e());
       l1 = l1c;
       // l2 doc lines
       ++p;
       /*
       cout <<" l2 status : " << (*p)->status() 
	    <<" pid = " << (*p)->pdg_id() 
	    <<" px = " << (*p)->momentum().px()
	    <<" py = " << (*p)->momentum().py() 
	    <<" pz = " << (*p)->momentum().pz() 
	    <<" e = " << (*p)->momentum().e() 
	    <<" charge = " << (pdt->particle((*p)->pdg_id()))->charge()
	    <<" charge3 = " << (pdt->particle((*p)->pdg_id()))->ID().threeCharge() << endl;
       */
       HepLorentzVector l2c((*p)->momentum().px(),(*p)->momentum().py(),(*p)->momentum().pz(),(*p)->momentum().e());
       l2 = l2c;
     }
   }
     
   // get collection of towers
   /*
   edm::Handle<CandidateCollection> calotowers;
   iEvent.getByLabel(calotowersSrc, calotowers);   
   const CandidateCollection* inputCol = calotowers.product();
   CandidateCollection::const_iterator candidate;
   for( candidate = inputCol->begin(); candidate != inputCol->end(); ++candidate )
     {
       double phi   = candidate->phi();
       double theta = candidate->theta();
       double eta   = candidate->eta();
       double e     = candidate->energy();
       double et    = e*sin(theta);
       cout <<" towers: phi = " << phi
	    <<" eta = " << eta
	    <<" et = " << et << endl;
    }
   */

   /*
   vector<HepLorentzVector> cjetsRaw;
   cjetsRaw.clear();

   vector<HepLorentzVector> cjetsMCJ;
   cjetsMCJ.clear();

   vector<HepLorentzVector> cjetsZSP;
   cjetsZSP.clear();

   vector<HepLorentzVector> cjetsJPT;
   cjetsJPT.clear();
   */

   // get gen jets collection
   Handle<GenJetCollection> genjets;
   iEvent.getByLabel(genjetsSrc, genjets);
   int jg = 0;
   for(GenJetCollection::const_iterator gjet = genjets->begin(); 
       gjet != genjets->end(); ++gjet ) {
     if(gjet->pt() >= 20.) {
       HepLorentzVector jet(gjet->px(), gjet->py(), gjet->pz(), gjet->energy());
       double drjl1 = l1.deltaR(jet);
       double drjl2 = l2.deltaR(jet);
       /*
       cout <<" Gen Jet " << jg
	    <<" pt = " << gjet->pt()
	    <<" px = " << gjet->px()
	    <<" py = " << gjet->py()
	    <<" pz = " << gjet->pz()
	    <<" energy = " << gjet->energy()
	    <<" j eta = " << gjet->eta()
	    <<" j phi = " << gjet->phi() 
	    <<" l1 eta = " << l1.eta() 
	    <<" l1 phi = " << l1.phi() 
	    <<" l2 eta = " << l2.eta() 
	    <<" l2 phi = " << l2.phi() 
	    <<" dr1 = " << drjl1 
	    <<" dr2 = " << drjl2 << endl;
       */
       if(drjl1 > 1.0 && drjl2 > 1.0) 
	 {
	   jg++;
	   if(jg <= 2) {
	     gjets.push_back(jet);
	   }
	 }
     }
   }

   //   cout <<" ==> NUMBER OF GOOD GEN JETS " << gjets.size() << endl;

   if(gjets.size() > 0) {
   // get calo jet collection
     edm::Handle<CaloJetCollection> calojets;
     iEvent.getByLabel(calojetsSrc, calojets);
     //     cout << "====> number of jets "<< calojets->size() << endl;
     if(calojets->size() > 0) {

       // MC jet energy corrections
       const JetCorrector* correctorMCJ = JetCorrector::getJetCorrector (JetCorrectionMCJ, iSetup);
       // ZSP jet energy corrections
       const JetCorrector* correctorZSP = JetCorrector::getJetCorrector (JetCorrectionZSP, iSetup);
       // Jet+tracks energy corrections
       const JetCorrector* correctorJPT = JetCorrector::getJetCorrector (JetCorrectionJPT, iSetup);
       
       // loop over jets and do matching with gen jets
       int jc = 0;
       
       for( CaloJetCollection::const_iterator cjet = calojets->begin(); 
	    cjet != calojets->end(); ++cjet ){ 
	 //
	 HepLorentzVector cjetc(cjet->px(), cjet->py(), cjet->pz(), cjet->energy());
	 
	 // MC jet energy corrections
	 double scaleMCJ = correctorMCJ->correction (*cjet);
	 Jet::LorentzVector jetscaleMCJ(cjet->px()*scaleMCJ, cjet->py()*scaleMCJ,
					cjet->pz()*scaleMCJ, cjet->energy()*scaleMCJ);
	 CaloJet cjetMCJ(jetscaleMCJ, cjet->getSpecific(), cjet->getJetConstituents());
	 
	 //       
	 //ZSP jet energy corrections
	 double scaleZSP = correctorZSP->correction (*cjet);
	 Jet::LorentzVector jetscaleZSP(cjet->px()*scaleZSP, cjet->py()*scaleZSP,
					cjet->pz()*scaleZSP, cjet->energy()*scaleZSP);
	 CaloJet cjetZSP(jetscaleZSP, cjet->getSpecific(), cjet->getJetConstituents());
	 //
	 // ZSP+JPT
	 double scaleJPT = correctorJPT->correction (cjetZSP,iEvent,iSetup);
	 Jet::LorentzVector jetscaleJPT(cjetZSP.px()*scaleJPT, cjetZSP.py()*scaleJPT,
					cjetZSP.pz()*scaleJPT, cjetZSP.energy()*scaleJPT);
	 CaloJet cjetJPT(jetscaleJPT, cjet->getSpecific(), cjet->getJetConstituents());

	 /*	 
	 cout <<" Jet " << jc
	      <<" raw pt = " << cjet->pt()
	      <<" mcj scale = " << scaleMCJ
	      <<" mcj pt = " << cjetMCJ.pt() 
	      <<" zsp scale = " << scaleZSP
	      <<" zsp pt = " << cjetZSP.pt() 
	      <<" jpt scale = " << scaleJPT
	      <<" jpt pt = " << cjetJPT.pt() 
	      <<" raw eta = " << cjet->eta() 
	      <<" raw phi = " << cjet->phi() << endl; 
	 */

	 double DRgjet1 = gjets[0].deltaR(cjetc);
	 /*
	 cout <<" --> DRgjet1 = " << DRgjet1 
	      <<" gen get1 pt = " << gjets[0].perp() <<" eta = " << gjets[0].eta() <<" phi = " << gjets[0].phi() << endl;   
	 */
	 if(DRgjet1 < DRMAXgjet1) {
	   DRMAXgjet1 = DRgjet1;
 
	   EtaGen1 = gjets[0].eta();
	   PhiGen1 = gjets[0].phi();
	   EtGen1  = gjets[0].perp();

	   EtaRaw1 = cjet->eta(); 
	   PhiRaw1 = cjet->phi();
	   EtRaw1  = cjet->pt();
	   EtMCJ1  = cjetMCJ.pt(); 
	   EtZSP1  = cjetZSP.pt(); 
	   EtJPT1  = cjetJPT.pt(); 
	 }
	 if(gjets.size() == 2) {
	   double DRgjet2 = gjets[1].deltaR(cjetc);
	   /*
	   cout <<" --> DRgjet2 = " << DRgjet2 
		<<" gen2 get pt = " << gjets[1].perp() <<" eta = " << gjets[1].eta() <<" phi = " << gjets[1].phi() << endl;   
	   */
	   if(DRgjet2 < DRMAXgjet2) { 
	     DRMAXgjet2 = DRgjet2;

	     EtaGen2 = gjets[1].eta();
	     PhiGen2 = gjets[1].phi();
	     EtGen2  = gjets[1].perp();

	     EtaRaw2 = cjet->eta(); 
	     PhiRaw2 = cjet->phi();
	     EtRaw2  = cjet->pt();
	     EtMCJ2  = cjetMCJ.pt(); 
	     EtZSP2  = cjetZSP.pt(); 
	     EtJPT2  = cjetJPT.pt(); 
	   }
	 }
	 jc++;
       }
       /*
        cout <<" best match to 1st gen get = " << DRMAXgjet1
	    <<" raw jet pt = " << EtRaw1 <<" eta = " << EtaRaw1 <<" phi " << PhiRaw1 
	    <<" mcj pt = " << EtMCJ1 << " zsp pt = " << EtZSP1 <<" jpt = " << EtJPT1 << endl; 
       if(gjets.size() == 2) {
	 cout <<" best match to 2st gen get = " << DRMAXgjet2
	      <<" raw jet pt = " << EtRaw2 <<" eta = " << EtaRaw2 <<" phi " << PhiRaw2 
	      <<" mcj pt = " << EtMCJ2 << " zsp pt = " << EtZSP2 <<" jpt = " << EtJPT2 << endl; 
       }
       */
     }
   }
   // fill tree
   t1->Fill();
}


//define this as a plug-in
DEFINE_FWK_MODULE(JPTAnalyzer);
