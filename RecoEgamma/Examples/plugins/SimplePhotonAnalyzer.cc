/**\class PhotonSimpleAnalyzer
 **
 ** $Date: 2012/01/28 10:29:35 $
 ** $Revision: 1.25 $
 ** \author Nancy Marinelli, U. of Notre Dame, US
*/

#include "RecoEgamma/Examples/plugins/SimplePhotonAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
//
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
//
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "TFile.h"

//========================================================================
SimplePhotonAnalyzer::SimplePhotonAnalyzer( const edm::ParameterSet& ps )
//========================================================================
{
  photonCollectionProducer_ = ps.getParameter<std::string>("phoProducer");
  photonCollection_ = ps.getParameter<std::string>("photonCollection");

  barrelEcalHits_   = ps.getParameter<edm::InputTag>("barrelEcalHits");
  endcapEcalHits_   = ps.getParameter<edm::InputTag>("endcapEcalHits");



  mcProducer_ = ps.getParameter<std::string>("mcProducer");
  //mcCollection_ = ps.getParameter<std::string>("mcCollection");
  vertexProducer_ = ps.getParameter<std::string>("primaryVertexProducer");
  sample_ = ps.getParameter<int>("sample");


}


//========================================================================
SimplePhotonAnalyzer::~SimplePhotonAnalyzer()
//========================================================================
{


}

//========================================================================
void
SimplePhotonAnalyzer::beginJob() {
//========================================================================

  edm::Service<TFileService> fs;

  float hiE=0;
  float loE=0;
  float hiEt=0;
  float loEt=0;
  float dPhi=0;
  float loRes=0;
  float hiRes=0;
  if ( sample_ ==1 ) {
    loE=0.;
    hiE=30.;
    loEt=0.;
    hiEt=30.;
    dPhi=0.2;
    loRes=0.;
    hiRes=1.2;
  } else if ( sample_ ==2 ) {
    loE=0.;
    hiE=200.;
    loEt=0.;
    hiEt=50.;
    dPhi=0.05;
    loRes=0.7;
    hiRes=1.2;
  } else if ( sample_ ==3 ) {
    loE=0.;
    hiE=500.;
    loEt=0.;
    hiEt=500.;
    dPhi=0.05;
    loRes=0.7;
    hiRes=1.2;
  }  else if (sample_==4) {
    loE=0.;
    hiE=6000.;
    loEt=0.;
    hiEt=1200.;
    dPhi=0.05;
    loRes=0.7;
    hiRes=1.2;
  }


  effEta_ = fs->make<TProfile> ("effEta"," Photon reconstruction efficiency",50,-2.5,2.5);
  effPhi_ = fs->make<TProfile> ("effPhi"," Photon reconstruction efficiency",80, -3.14, 3.14);

  h1_deltaEta_ = fs->make<TH1F>("deltaEta"," Reco photon Eta minus Generated photon Eta  ",100,-0.2, 0.2);
  h1_deltaPhi_ = fs->make<TH1F>("deltaPhi","Reco photon Phi minus Generated photon Phi ",100,-dPhi, dPhi);
  h1_pho_Eta_ = fs->make<TH1F>("phoEta","Photon  Eta ",40,-3., 3.);
  h1_pho_Phi_ = fs->make<TH1F>("phoPhi","Photon  Phi ",40,-3.14, 3.14);
  h1_pho_E_ = fs->make<TH1F>("phoE","Photon Energy ",100,loE,hiE);
  h1_pho_Et_ = fs->make<TH1F>("phoEt","Photon Et ",100,loEt,hiEt);

  h1_scEta_ = fs->make<TH1F>("scEta"," SC Eta ",40,-3., 3.);
  h1_deltaEtaSC_ = fs->make<TH1F>("deltaEtaSC"," SC Eta minus Generated photon Eta  ",100,-0.02, 0.02);

 //
  h1_recEoverTrueEBarrel_ = fs->make<TH1F>("recEoverTrueEBarrel"," Reco photon Energy over Generated photon Energy: Barrel ",100,loRes, hiRes);
  h1_recEoverTrueEEndcap_ = fs->make<TH1F>("recEoverTrueEEndcap"," Reco photon Energy over Generated photon Energy: Endcap ",100,loRes, hiRes);

  //

  h1_pho_R9Barrel_ = fs->make<TH1F>("phoR9Barrel","Photon  3x3 energy / SuperCluster energy : Barrel ",100,0.,1.2);
  h1_pho_R9Endcap_ = fs->make<TH1F>("phoR9Endcap","Photon  3x3 energy / SuperCluster energy : Endcap ",100,0.,1.2);
  h1_pho_sigmaIetaIetaBarrel_ = fs->make<TH1F>("sigmaIetaIetaBarrel",   "sigmaIetaIeta: Barrel",100,0., 0.05) ;
  h1_pho_sigmaIetaIetaEndcap_ = fs->make<TH1F>("sigmaIetaIetaEndcap" ,  "sigmaIetaIeta: Endcap",100,0., 0.1) ;
  h1_pho_hOverEBarrel_ = fs->make<TH1F>("hOverEBarrel",   "H/E: Barrel",100,0., 0.1) ;
  h1_pho_hOverEEndcap_ = fs->make<TH1F>("hOverEEndcap",   "H/E: Endcap",100,0., 0.1) ;
  h1_pho_ecalIsoBarrel_ = fs->make<TH1F>("ecalIsolBarrel",   "isolation et sum in Ecal: Barrel",100,0., 100.) ;
  h1_pho_ecalIsoEndcap_ = fs->make<TH1F>("ecalIsolEndcap",   "isolation et sum in Ecal: Endcap",100,0., 100.) ;
  h1_pho_hcalIsoBarrel_ = fs->make<TH1F>("hcalIsolBarrel",   "isolation et sum in Hcal: Barrel",100,0., 100.) ;
  h1_pho_hcalIsoEndcap_ = fs->make<TH1F>("hcalIsolEndcap",   "isolation et sum in Hcal: Endcap",100,0., 100.) ;
  h1_pho_trkIsoBarrel_ = fs->make<TH1F>("trkIsolBarrel",   "isolation pt sum in the tracker: Barrel",100,0., 100.) ;
  h1_pho_trkIsoEndcap_ = fs->make<TH1F>("trkIsolEndcap",   "isolation pt sum in the tracker: Endcap",100,0., 100.) ;





}


//========================================================================
void
SimplePhotonAnalyzer::analyze( const edm::Event& evt, const edm::EventSetup& es ) {
//========================================================================

  using namespace edm; // needed for all fwk related classes
  edm::LogInfo("PhotonAnalyzer") << "Analyzing event number: " << evt.id() << "\n";


 // get the  calo topology  from the event setup:
  edm::ESHandle<CaloTopology> pTopology;
  es.get<CaloTopologyRecord>().get(theCaloTopo_);



  // Get the  corrected  photon collection (set in the configuration) which also contains infos about conversions

  Handle<reco::PhotonCollection> photonHandle;
  evt.getByLabel(photonCollectionProducer_, photonCollection_ , photonHandle);
  const reco::PhotonCollection photonCollection = *(photonHandle.product());

  Handle< HepMCProduct > hepProd ;
  evt.getByLabel( mcProducer_.c_str(),  hepProd ) ;
  const HepMC::GenEvent * myGenEvent = hepProd->GetEvent();


  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p ) {
    if ( !( (*p)->pdg_id() == 22 && (*p)->status()==1 )  )  continue;

    // single primary photons or photons from Higgs or RS Graviton
    HepMC::GenParticle* mother = 0;
    if ( (*p)->production_vertex() )  {
      if ( (*p)->production_vertex()->particles_begin(HepMC::parents) !=
           (*p)->production_vertex()->particles_end(HepMC::parents))
	mother = *((*p)->production_vertex()->particles_begin(HepMC::parents));
    }
    if ( ((mother == 0) || ((mother != 0) && (mother->pdg_id() == 25))
	  || ((mother != 0) && (mother->pdg_id() == 22)))) {

      float minDelta=10000.;
      std::vector<reco::Photon> localPhotons;
      int index=0;
      int iMatch=-1;

      float phiPho=(*p)->momentum().phi();
      float etaPho=(*p)->momentum().eta();
      etaPho = etaTransformation(etaPho, (*p)->production_vertex()->position().z()/10. );

      bool matched=false;
      // loop  Photon candidates
      for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {
	reco::Photon localPho = reco::Photon(*iPho);
	localPhotons.push_back(localPho);

	/// Match reconstructed photon candidates with the nearest generated photonPho;
	float phiClu=localPho.phi();
	float etaClu=localPho.eta();
	float deltaPhi = phiClu-phiPho;
	float deltaEta = etaClu-etaPho;

	if ( deltaPhi > pi )  deltaPhi -= twopi;
	if ( deltaPhi < -pi) deltaPhi += twopi;
	deltaPhi=std::pow(deltaPhi,2);
	deltaEta=std::pow(deltaEta,2);
	float delta = sqrt( deltaPhi+deltaEta);
	if ( delta<0.1 && delta < minDelta ) {
	  minDelta=delta;
	  iMatch=index;

	}
	index++;
      } // End loop over photons

      double wt=0.;
      if ( iMatch>-1 ) matched = true;

      /// Plot kinematic disctributions for matched photons
      if (matched ) {
        wt=1.;

	effEta_ ->Fill ( etaPho, wt);
	effPhi_ ->Fill ( phiPho, wt);
	reco::Photon matchingPho = localPhotons[iMatch];

	bool  phoIsInBarrel=false;
	if ( fabs(matchingPho.superCluster()->position().eta() ) <  1.479 ) {
	  phoIsInBarrel=true;
	}
	edm::Handle<EcalRecHitCollection>   ecalRecHitHandle;


	h1_scEta_->Fill( matchingPho.superCluster()->position().eta() );
	float trueEta=  (*p)->momentum().eta() ;
	trueEta = etaTransformation(trueEta, (*p)->production_vertex()->position().z()/10. );
	h1_deltaEtaSC_ -> Fill(  localPhotons[iMatch].superCluster()->eta()- trueEta  );

	float photonE = matchingPho.energy();
	float photonEt= matchingPho.et() ;
	float photonEta= matchingPho.eta() ;
	float photonPhi= matchingPho.phi() ;

	float r9 = matchingPho.r9();
	float sigmaIetaIeta =  matchingPho.sigmaIetaIeta();
	float hOverE = matchingPho.hadronicOverEm();
	float ecalIso = matchingPho.ecalRecHitSumEtConeDR04();
	float hcalIso = matchingPho.hcalTowerSumEtConeDR04();
	float trkIso =  matchingPho.trkSumPtSolidConeDR04();


	h1_pho_E_->Fill( photonE  );
	h1_pho_Et_->Fill( photonEt );
	h1_pho_Eta_->Fill( photonEta );
	h1_pho_Phi_->Fill( photonPhi );

	h1_deltaEta_ -> Fill(  photonEta - (*p)->momentum().eta()  );
	h1_deltaPhi_ -> Fill(  photonPhi - (*p)->momentum().phi()  );

	if ( phoIsInBarrel ) {
	  h1_recEoverTrueEBarrel_ -> Fill ( photonE / (*p)->momentum().e() );
	  h1_pho_R9Barrel_->Fill( r9 );
          h1_pho_sigmaIetaIetaBarrel_->Fill ( sigmaIetaIeta );
          h1_pho_hOverEBarrel_-> Fill ( hOverE );
	  h1_pho_ecalIsoBarrel_-> Fill ( ecalIso );
	  h1_pho_hcalIsoBarrel_-> Fill ( hcalIso );
	  h1_pho_trkIsoBarrel_-> Fill ( trkIso );

	} else {
	  h1_recEoverTrueEEndcap_ -> Fill ( photonE / (*p)->momentum().e() );
	  h1_pho_R9Endcap_->Fill( r9 );
          h1_pho_sigmaIetaIetaEndcap_->Fill ( sigmaIetaIeta );
          h1_pho_hOverEEndcap_-> Fill ( hOverE );
	  h1_pho_ecalIsoEndcap_-> Fill ( ecalIso );
	  h1_pho_hcalIsoEndcap_-> Fill ( hcalIso );
	  h1_pho_trkIsoEndcap_-> Fill ( trkIso );


	}


      } //  reco photon matching MC truth 




    } // End loop over MC particles

  }


}


float SimplePhotonAnalyzer::etaTransformation(  float EtaParticle , float Zvertex)  {

  //---Definitions
  const float PI    = 3.1415927;
  //UNUSED const float TWOPI = 2.0*PI;

  //---Definitions for ECAL
  const float R_ECAL           = 136.5;
  const float Z_Endcap         = 328.0;
  const float etaBarrelEndcap  = 1.479;

  //---ETA correction

  float Theta = 0.0  ;
  float ZEcal = R_ECAL*sinh(EtaParticle)+Zvertex;

  if(ZEcal != 0.0) Theta = atan(R_ECAL/ZEcal);
  if(Theta<0.0) Theta = Theta+PI ;
  float ETA = - log(tan(0.5*Theta));

  if( fabs(ETA) > etaBarrelEndcap )
    {
      float Zend = Z_Endcap ;
      if(EtaParticle<0.0 )  Zend = -Zend ;
      float Zlen = Zend - Zvertex ;
      float RR = Zlen/sinh(EtaParticle);
      Theta = atan(RR/Zend);
      if(Theta<0.0) Theta = Theta+PI ;
      ETA = - log(tan(0.5*Theta));
    }
  //---Return the result
  return ETA;
  //---end
}




//========================================================================
void
SimplePhotonAnalyzer::endJob() {
//========================================================================



}
