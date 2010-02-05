#include "CLHEP/Vector/LorentzVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
//#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace reco;
using namespace edm;

// -----------------------------------------------------------------------------
//
class JPTAnalyzer : public edm::EDAnalyzer {

public:

  explicit JPTAnalyzer(const edm::ParameterSet&);
  ~JPTAnalyzer();
  
private:

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------

  string fOutputFileName;
  string calojetsSrc;
  string zspjetsSrc;
  string genjetsSrc;

  string JetCorrectionJPT;

  double  EtaGen1, PhiGen1, EtaRaw1, PhiRaw1, EtGen1, EtRaw1, EtMCJ1, EtZSP1, EtJPT1, DRMAXgjet1, EtaZSP1, PhiZSP1, EtaJPT1, PhiJPT1; 
  double  EtaGen2, PhiGen2, EtaRaw2, PhiRaw2, EtGen2, EtRaw2, EtMCJ2, EtZSP2, EtJPT2, DRMAXgjet2, EtaZSP2, PhiZSP2, EtaJPT2, PhiJPT2; 

  TFile* hOutputFile ;
  TTree* t1;

  bool scalar_;

};

// -----------------------------------------------------------------------------
//
void JPTAnalyzer::beginJob( ) {

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

  t1->Branch("EtaZSP1",&EtaZSP1,"EtaZSP1/D");
  t1->Branch("PhiZSP1",&PhiZSP1,"PhiZSP1/D");
  t1->Branch("EtaJPT1",&EtaJPT1,"EtaJPT1/D");
  t1->Branch("PhiJPT1",&PhiJPT1,"PhiJPT1/D");

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

  t1->Branch("EtaZSP2",&EtaZSP2,"EtaZSP2/D");
  t1->Branch("PhiZSP2",&PhiZSP2,"PhiZSP2/D");
  t1->Branch("EtaJPT2",&EtaJPT2,"EtaJPT2/D");
  t1->Branch("PhiJPT2",&PhiJPT2,"PhiJPT2/D");

  return ;
}

// -----------------------------------------------------------------------------
//
void JPTAnalyzer::endJob() {
  hOutputFile->Write() ;
  hOutputFile->Close() ;
  return ;
}

// -----------------------------------------------------------------------------
//
JPTAnalyzer::JPTAnalyzer( const edm::ParameterSet& iConfig ) {

  //now do what ever initialization is needed
  using namespace edm;
  // 
  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile");
  //
  // get names of input object collections
  // raw calo jets
  calojetsSrc   = iConfig.getParameter< std::string > ("calojets");
  // calo jets after zsp corrections
  zspjetsSrc    = iConfig.getParameter< std::string > ("zspjets");
  genjetsSrc    = iConfig.getParameter< std::string > ("genjets");
  //
  // MC jet energy corrections
  //  JetCorrectionMCJ = iConfig.getParameter< std::string > ("JetCorrectionMCJ");
  // ZSP jet energy corrections
  //  JetCorrectionZSP = iConfig.getParameter< std::string > ("JetCorrectionZSP");
  // Jet+tracks energy corrections
  JetCorrectionJPT = iConfig.getParameter< std::string > ("JetCorrectionJPT");

  scalar_ = iConfig.getUntrackedParameter<bool> ("UseScalarMethod",false);
  
}

// -----------------------------------------------------------------------------
//
JPTAnalyzer::~JPTAnalyzer() {;}

// -----------------------------------------------------------------------------
//
void JPTAnalyzer::analyze( const edm::Event& iEvent, 
			      const edm::EventSetup& iSetup ) {

  using namespace edm;

   // initialize vector containing two highest Et gen jets > 20 GeV
   // in this example they are checked not to be leptons from Z->ll decay (DR match)
   vector<CLHEP::HepLorentzVector> gjets;
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
   
   PhiZSP1 = 0.;
   EtaZSP1 = 0.;
   PhiJPT1 = 0.;
   EtaJPT1 = 0.;

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

   PhiZSP2 = 0.;
   EtaZSP2 = 0.;
   PhiJPT2 = 0.;
   EtaJPT2 = 0.;

   //   edm::ESHandle<CaloGeometry> geometry;
   //   iSetup.get<IdealGeometryRecord>().get(geometry);


   // get MC info
   edm::Handle<HepMCProduct> EvtHandle ;
   iEvent.getByLabel( "source", EvtHandle ) ;
   //  iEvent.getByLabel( "VtxSmeared", EvtHandle ) ;

   // l1 and l2 are leptons from Z->ll to be checked they are not gen jets (DR match)
   CLHEP::HepLorentzVector l1(0.,0.,1.,1.);
   CLHEP::HepLorentzVector l2(0.,0.,1.,1.);
 
  //
     
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
   vector<CLHEP::HepLorentzVector> cjetsRaw;
   cjetsRaw.clear();

   vector<CLHEP::HepLorentzVector> cjetsMCJ;
   cjetsMCJ.clear();

   vector<CLHEP::HepLorentzVector> cjetsZSP;
   cjetsZSP.clear();

   vector<CLHEP::HepLorentzVector> cjetsJPT;
   cjetsJPT.clear();
   */

   // get gen jets collection
   Handle<GenJetCollection> genjets;
   iEvent.getByLabel(genjetsSrc, genjets);
   int jg = 0;
   for(GenJetCollection::const_iterator gjet = genjets->begin(); 
       gjet != genjets->end(); ++gjet ) {
     if(gjet->pt() >= 20.) {
       CLHEP::HepLorentzVector jet(gjet->px(), gjet->py(), gjet->pz(), gjet->energy());
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
   // get calo jet after zsp collection
     edm::Handle<CaloJetCollection> zspjets;
     iEvent.getByLabel(zspjetsSrc, zspjets);
     /*
     cout << "====> number of calo jets "<< calojets->size() 
	  << " number if zsp jets = " << zspjets->size() << endl;
     */
     if(calojets->size() > 0) {

       // MC jet energy corrections
       //       const JetCorrector* correctorMCJ = JetCorrector::getJetCorrector (JetCorrectionMCJ, iSetup);
       // ZSP jet energy corrections
       //       const JetCorrector* correctorZSP = JetCorrector::getJetCorrector (JetCorrectionZSP, iSetup);
       // Jet+tracks energy corrections

       const JetCorrector* correctorJPT = JetCorrector::getJetCorrector (JetCorrectionJPT, iSetup);
       
       // loop over jets and do matching with gen jets
       int jc = 0;
       
       for( CaloJetCollection::const_iterator cjet = calojets->begin(); 
	    cjet != calojets->end(); ++cjet ){ 
	 //
	 CLHEP::HepLorentzVector cjetc(cjet->px(), cjet->py(), cjet->pz(), cjet->energy());
	 /*
	 cout <<" ==> calo jet Et = " << cjet->pt()
	      <<" eta = " << cjet->eta()
	      <<" phi = " << cjet->phi() << endl;
	 */
	 CaloJetCollection::const_iterator zspjet;
	 for( zspjet = zspjets->begin(); 
	      zspjet != zspjets->end(); ++zspjet ){ 
	   CLHEP::HepLorentzVector zspjetc(zspjet->px(), zspjet->py(), zspjet->pz(), zspjet->energy());
	   double dr = zspjetc.deltaR(cjetc);
	   /*
	   cout <<"      zspjet Et = " << zspjet->pt()
		<<" eta = " << zspjet->eta()
		<<" phi = " << zspjet->phi()
		<<" dr = " << dr << endl;
	   */
	   if(dr < 0.001) break;
	 }
	 /*
	 cout <<" --> matched zsp jet found Et = " << zspjet->pt()
	      <<" eta = " << zspjet->eta()
	      <<" phi = " << zspjet->phi() << endl;
	 */
	 
	 // ZSP JetRef
	 edm::RefToBase<reco::Jet> zspRef( edm::Ref<CaloJetCollection>( zspjets, zspjet - zspjets->begin() ) );
	 
	 // JPT corrections
	 double scaleJPT = -1.;
	 Jet::LorentzVector jetscaleJPT;
	 if ( scalar_ ) {
	   
	   scaleJPT = correctorJPT->correction ( (*zspjet), zspRef, iEvent, iSetup );
	   jetscaleJPT = Jet::LorentzVector( zspjet->px()*scaleJPT, 
					     zspjet->py()*scaleJPT,
					     zspjet->pz()*scaleJPT, 
					     zspjet->energy()*scaleJPT );
	 } else {
	   JetCorrector::LorentzVector p4;
	   scaleJPT = correctorJPT->correction( *zspjet, zspRef, iEvent, iSetup, p4 );
	   jetscaleJPT = Jet::LorentzVector( p4.Px(), p4.Py(), p4.Pz(), p4.E() );
	 }	   
	 
	 CaloJet cjetJPT(jetscaleJPT, cjet->getSpecific(), cjet->getJetConstituents());

// 	 cout <<" TEST pt=" << jetscaleJPT.pt()
// 	      <<" scale=" << scaleJPT
// 	      <<" e=" << jetscaleJPT.E()
// 	      <<" et=" << jetscaleJPT.Et()
// 	      <<" eta=" << jetscaleJPT.eta()
// 	      <<" phi=" << jetscaleJPT.phi()
//  	      <<" e=" << cjetJPT.energy()
//  	      <<" et=" << cjetJPT.et()
//  	      <<" eta=" << cjetJPT.eta()
//  	      <<" phi=" << cjetJPT.phi()
// 	      << endl;
	 
	 double DRgjet1 = gjets[0].deltaR(cjetc);

	 if(DRgjet1 < DRMAXgjet1) {
	   DRMAXgjet1 = DRgjet1;
 
	   EtaGen1 = gjets[0].eta();
	   PhiGen1 = gjets[0].phi();
	   EtGen1  = gjets[0].perp();

	   EtaRaw1 = cjet->eta(); 
	   PhiRaw1 = cjet->phi();
	   EtRaw1  = cjet->pt();
	   //	   EtMCJ1  = cjetMCJ.pt(); 
	   EtZSP1  = zspjet->pt(); 
	   EtJPT1  = cjetJPT.pt(); 
	   
	   EtaZSP1 = zspjet->eta(); 
	   PhiZSP1 = zspjet->phi(); 
	   EtaJPT1 = cjetJPT.eta(); 
	   PhiJPT1 = cjetJPT.phi(); 

	 }
	 if(gjets.size() == 2) {
	   double DRgjet2 = gjets[1].deltaR(cjetc);
	   if(DRgjet2 < DRMAXgjet2) { 
	     DRMAXgjet2 = DRgjet2;

	     EtaGen2 = gjets[1].eta();
	     PhiGen2 = gjets[1].phi();
	     EtGen2  = gjets[1].perp();

	     EtaRaw2 = cjet->eta(); 
	     PhiRaw2 = cjet->phi();
	     EtRaw2  = cjet->pt();
	     EtZSP2  = zspjet->pt(); 
	     EtJPT2  = cjetJPT.pt(); 
	     
	     EtaZSP2 = zspjet->eta(); 
	     PhiZSP2 = zspjet->phi(); 
	     EtaJPT2 = cjetJPT.eta(); 
	     PhiJPT2 = cjetJPT.phi(); 
	     
	   }
	 }
	 jc++;
       }

       /*
        cout <<" best1 match to 1st gen get = " << DRMAXgjet1
	     <<" raw jet pt = " << EtRaw1 <<" eta = " << EtaRaw1 <<" phi " << PhiRaw1 
	     << " phizsp " << PhiZSP1 << " phijpt " << PhiJPT1 
	    <<" mcj pt = " << EtMCJ1 << " zsp pt = " << EtZSP1 <<" jpt = " << EtJPT1 << endl; 
       if(gjets.size() == 2) {
	 cout <<" best1 match to 2st gen get = " << DRMAXgjet2
	      <<" raw jet pt = " << EtRaw2 <<" eta = " << EtaRaw2 <<" phi " << PhiRaw2 
	      << " phizsp " << PhiZSP2 << " phijpt " << PhiJPT2 
	      <<" mcj pt = " << EtMCJ2 << " zsp pt = " << EtZSP2 <<" jpt = " << EtJPT2 << endl; 
       }
       */
       
     }
   }
 
   // fill tree
   t1->Fill();

//         cout <<" best2 match to 1st gen get = " << DRMAXgjet1
// 	     <<" raw jet pt = " << EtRaw1 <<" eta = " << EtaRaw1 <<" phi " << PhiRaw1 
// 	     << " phizsp " << PhiZSP1 << " phijpt " << PhiJPT1 
// 	    <<" mcj pt = " << EtMCJ1 << " zsp pt = " << EtZSP1 <<" jpt = " << EtJPT1 << endl; 
// 	cout <<" best2 match to 2st gen get = " << DRMAXgjet2
// 	     <<" raw jet pt = " << EtRaw2 <<" eta = " << EtaRaw2 <<" phi " << PhiRaw2 
// 	     << " phizsp " << PhiZSP2 << " phijpt " << PhiJPT2 
// 	     <<" mcj pt = " << EtMCJ2 << " zsp pt = " << EtZSP2 <<" jpt = " << EtJPT2 << endl; 

}

// -----------------------------------------------------------------------------
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JPTAnalyzer);
