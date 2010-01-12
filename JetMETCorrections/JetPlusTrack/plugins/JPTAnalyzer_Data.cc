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
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
//#include "CLHEP/HepPDT/DefaultConfig.hh"
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
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Algorithms/interface/JetPlusTrackCorrector.h"
#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"
#include "DataFormats/JetReco/interface/JetID.h"
//
// muons and tracks
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
// ecal
//#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
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

class JPTAnalyzer_Data : public edm::EDAnalyzer {
   public:
      explicit JPTAnalyzer_Data(const edm::ParameterSet&);
      ~JPTAnalyzer_Data();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  // ----------member data ---------------------------
  // JPT corrector
  const JetPlusTrackCorrector* jptCorrector_;
  //
  // output root file
  string fOutputFileName ;
  // names of modules, producing object collections
  // raw calo jets
  string calojetsSrc;
  string jetsIDSrc;
  string jetExtenderSrc;
  // calo jets after zsp corrections
  string zspjetsSrc;
  //
  // MC jet energy corrections
  //  string JetCorrectionMCJ;
  // ZSP jet energy corrections
  //  string JetCorrectionZSP;
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
JPTAnalyzer_Data::beginJob(const edm::EventSetup&)
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
JPTAnalyzer_Data::endJob() {

  hOutputFile->Write() ;
  hOutputFile->Close() ;
  
  return ;
}

//
// constructors and destructor
//
JPTAnalyzer_Data::JPTAnalyzer_Data(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  using namespace edm;
  // 
  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile");
  //
  // get names of input object collections
  // raw calo jets
  calojetsSrc      = iConfig.getParameter< std::string > ("calojets");
  jetsIDSrc        = iConfig.getParameter< std::string > ("jetsID");
  //  jetExtenderSrc   = iConfig.getParameter< std::string > ("jetExtender");
  // calo jets after zsp corrections
  zspjetsSrc    = iConfig.getParameter< std::string > ("zspjets");
  //
  // MC jet energy corrections
  //  JetCorrectionMCJ = iConfig.getParameter< std::string > ("JetCorrectionMCJ");
  // ZSP jet energy corrections
  //  JetCorrectionZSP = iConfig.getParameter< std::string > ("JetCorrectionZSP");
  // Jet+tracks energy corrections
  JetCorrectionJPT = iConfig.getParameter< std::string > ("JetCorrectionJPT");
}


JPTAnalyzer_Data::~JPTAnalyzer_Data()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
JPTAnalyzer_Data::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
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

   edm::Handle<CaloJetCollection> calojets;
   iEvent.getByLabel(calojetsSrc, calojets);

   //   Handle<JetExtendedAssociation::Container> jetExtender;
   //   iEvent.getByLabel(jetExtenderSrc,jetExtender);

   Handle<ValueMap<reco::JetID> > jetsID;
   iEvent.getByLabel(jetsIDSrc,jetsID);

   // get calo jet after zsp collection
   edm::Handle<CaloJetCollection> zspjets;
   iEvent.getByLabel(zspjetsSrc, zspjets);
     /*
     cout << "====> number of calo jets "<< calojets->size() 
	  << " number if zsp jets = " << zspjets->size() << endl;
     */

   // get vertex
   Handle<reco::VertexCollection> recVtxs;
   iEvent.getByLabel("offlinePrimaryVertices",recVtxs);
   int nvtx = 0;
   double mPVx, mPVy, mPVz;
   int ntrkV = 0;
   for(unsigned int ind = 0; ind < recVtxs->size(); ind++) 
     {
       if (!((*recVtxs)[ind].isFake())) 
	 {
	   nvtx = nvtx + 1;
	   if(nvtx == 1) {
	     mPVx  = (*recVtxs)[ind].x();
	     mPVy  = (*recVtxs)[ind].y();
	     mPVz  = (*recVtxs)[ind].z();
	     ntrkV = (*recVtxs)[ind].tracksSize();
	   }
	 }
     }
   
   if( (nvtx == 1) && (ntrkV > 3) ) {

     /*
     cout <<"   Vertex found, X = " << mPVx
          <<" Y = " << mPVy
	  <<" Z = " << mPVz 
	  <<" ntrk = " << ntrk << endl;
     
     */

     if( (calojets->size() > 0) && (zspjets->size() > 0) ) {

       // MC jet energy corrections
       //       const JetCorrector* correctorMCJ = JetCorrector::getJetCorrector (JetCorrectionMCJ, iSetup);
       // ZSP jet energy corrections
       //       const JetCorrector* correctorZSP = JetCorrector::getJetCorrector (JetCorrectionZSP, iSetup);
       // Jet+tracks energy corrections
       
       const JetCorrector* correctorJPT = JetCorrector::getJetCorrector (JetCorrectionJPT, iSetup);
       
       // loop over jets and do matching with gen jets
       // number of all raw calo jets
       int jc = 0;
       // number of raw calo jets passed jet ID and eta < 2.0
       int jcgood = 0;
       // number of jtp jets > 10 GeV       
       int jjpt = 0;

       for( CaloJetCollection::const_iterator cjet = calojets->begin(); 
	    cjet != calojets->end(); ++cjet ){ 
	 
	 // raw jet selection 
	 RefToBase<Jet> jetRef(Ref<CaloJetCollection>(calojets,jc));
	 double mN90  = (*calojets)[jc].n90();
	 double mEmf  = (*calojets)[jc].emEnergyFraction(); 	
	 double mfHPD = (*jetsID)[jetRef].fHPD;
	 double mfRBX = (*jetsID)[jetRef].fRBX; 
	 
	 jc++;
	 
	 // good jet selections
	 
	 if(mEmf < 0.01) continue;
	 if(mfHPD>0.98) continue;
	 if(mfRBX>0.98) continue;
	 if(mN90 < 2) continue;
	 if(fabs(cjet->eta()) > 2.0) continue;
	 
	 //
	 CLHEP::HepLorentzVector cjetc(cjet->px(), cjet->py(), cjet->pz(), cjet->energy());
	 /*
	 cout <<" ==> calo jet Et = " << cjet->pt()
	      <<" eta = " << cjet->eta()
	      <<" phi = " << cjet->phi() << endl;
	 
	 cout <<"  == jet N = " << jcgood
	      <<" pt = " << cjet->pt()
	      <<" eta = " << cjet->eta()
	      <<" mEmf = " << mEmf
	      <<" mfHPD = " << mfHPD
	      <<" mfRBX = " << mfRBX << endl;
	 */
	 int iczsp = 0;
	 
	 CaloJetCollection::const_iterator zspjet;
	 for( zspjet = zspjets->begin(); 
	      zspjet != zspjets->end(); ++zspjet ){ 
	   CLHEP::HepLorentzVector zspjetc(zspjet->px(), zspjet->py(), zspjet->pz(), zspjet->energy());
	   double dr = zspjetc.deltaR(cjetc);

	   if(dr < 0.001) {
	     iczsp = 1;
	     break;
	   }
	 }
	 
	 if(iczsp == 0) continue;
	 
	 jcgood = jcgood + 1;

	 // JPT corrections
	 double scaleJPT = correctorJPT->correction ((*zspjet),iEvent,iSetup);
	 Jet::LorentzVector jetscaleJPT(zspjet->px()*scaleJPT, zspjet->py()*scaleJPT,
					zspjet->pz()*scaleJPT, zspjet->energy()*scaleJPT);
	 /* 
	 cout <<" ....> corrected jpt jet Et = " << jetscaleJPT.pt()
	      <<" eta = " << jetscaleJPT.eta()
	      <<" phi = " << jetscaleJPT.phi() << endl;
	 */
	 CaloJet cjetJPT(jetscaleJPT, cjet->getSpecific(), cjet->getJetConstituents());

	 jpt::MatchedTracks pions;
	 jpt::MatchedTracks muons;
	 jpt::MatchedTracks electrons;
	 const bool particlesOK = true;
	 jptCorrector_ = dynamic_cast<const JetPlusTrackCorrector*>(correctorJPT);

	 jptCorrector_->matchTracks((*zspjet),iEvent,iSetup,pions,muons,electrons);
	 int NtrkJPT = pions.inVertexOutOfCalo_.size() + pions.inVertexInCalo_.size();
	 for (reco::TrackRefVector::const_iterator iInConeVtxTrk = pions.inVertexOutOfCalo_.begin(); 
	      iInConeVtxTrk != pions.inVertexOutOfCalo_.end(); ++iInConeVtxTrk) {
	   const double pt  = (*iInConeVtxTrk)->pt();
	   const double eta = (*iInConeVtxTrk)->eta();
	   const double phi = (*iInConeVtxTrk)->phi();

	   int trkNVhits  = (*iInConeVtxTrk)->numberOfValidHits();
	   int Nlayers    = (*iInConeVtxTrk)->hitPattern().trackerLayersWithMeasurement();
	   int NpxlHits   = (*iInConeVtxTrk)->hitPattern().pixelLayersWithMeasurement();
	   int NoutLayers = (*iInConeVtxTrk)->hitPattern().stripTOBLayersWithMeasurement() +
	                    (*iInConeVtxTrk)->hitPattern().stripTECLayersWithMeasurement();

	   cout <<"  --> track pT = " << pt
		<<"  eta = " << eta
		<<"  phi = " << phi << endl;
	 }

	 if(cjetJPT.pt() > 10.) {

	   jjpt = jjpt + 1;
	   
	   if(jjpt == 1) {
	     EtaRaw1 = cjet->eta(); 
	     PhiRaw1 = cjet->phi();
	     EtRaw1  = cjet->pt();
	     EtZSP1  = zspjet->pt(); 
	     EtJPT1  = cjetJPT.pt(); 
	   }
	   if(jjpt == 2) {
	     EtaRaw2 = cjet->eta(); 
	     PhiRaw2 = cjet->phi();
	     EtRaw2  = cjet->pt();
	     EtZSP2  = zspjet->pt(); 
	     EtJPT2  = cjetJPT.pt(); 
	   }
	   t1->Fill();
	 }
       }
     }
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
   // fill tree
}

//define this as a plug-in
DEFINE_FWK_MODULE(JPTAnalyzer_Data);
