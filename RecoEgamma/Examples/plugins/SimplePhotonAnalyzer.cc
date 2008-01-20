/**\class PhotonSimpleAnalyzer
 **
 ** $Date: 2007/11/14 15:12:17 $ 
 ** $Revision: 1.5 $
 ** \author Nancy Marinelli, U. of Notre Dame, US
*/

#include "RecoEgamma/Examples/plugins/SimplePhotonAnalyzer.h"

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
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
//
#include "CLHEP/Units/PhysicalConstants.h"
#include "TFile.h"

//========================================================================
SimplePhotonAnalyzer::SimplePhotonAnalyzer( const edm::ParameterSet& ps )
//========================================================================
{
  photonCollectionProducer_ = ps.getParameter<std::string>("phoProducer");
  photonCollection_ = ps.getParameter<std::string>("photonCollection");


  mcProducer_ = ps.getParameter<std::string>("mcProducer");
  //mcCollection_ = ps.getParameter<std::string>("mcCollection");
  vertexProducer_ = ps.getParameter<std::string>("primaryVertexProducer");
 

  outputFile_   = ps.getParameter<std::string>("outputFile");
  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE"); // open output file to store histograms

}


//========================================================================
SimplePhotonAnalyzer::~SimplePhotonAnalyzer()
//========================================================================
{

  delete rootFile_;
}

//========================================================================
void
SimplePhotonAnalyzer::beginJob(edm::EventSetup const&) {
//========================================================================

  // go to *OUR* rootfile and book histograms
  rootFile_->cd();

  h1_scEt_ = new TH1F("scEt","Uncorrected photons : SC Et ",100,0.,100.);
  h1_scE_ = new TH1F("scE","Uncorrected photons : SC Energy ",100,0.,100.);
  h1_scEta_ = new TH1F("scEta","Uncorrected photons:  SC Eta ",40,-3., 3.);
  h1_scPhi_ = new TH1F("scPhi","Uncorrected photons: SC Phi ",40,-3.14, 3.14);
  //
  h1_phoE_ = new TH1F("phoE","Uncorrected photons : photon Energy ",100,0., 100.);
  h1_phoEta_ = new TH1F("phoEta","Uncorrected photons:  photon Eta ",40,-3., 3.);
  h1_phoPhi_ = new TH1F("phoPhi","Uncorrected photons: photon Phi ",40,-3.14, 3.14);
  //
  h1_recEoverTrueE_ = new TH1F("recEoverTrueE"," Reco photon Energy over Generated photon Energy ",100,0., 3);
  h1_deltaEta_ = new TH1F("deltaEta"," Reco photon Eta minus Generated photon Eta  ",100,-0.2, 0.2);
  h1_deltaPhi_ = new TH1F("deltaPhi","Reco photon Phi minus Generated photon Phi ",100,-0.2, 0.2);
  //
  h1_corrPho_E_ = new TH1F("corrPhoE","Corrected photons : Energy ",100,0., 100.);
  h1_corrPho_Eta_ = new TH1F("corrPhoEta","Corrected photons:  Eta ",40,-3., 3.);
  h1_corrPho_Phi_ = new TH1F("corrPhoPhi","Corrected photons:  Phi ",40,-3.14, 3.14);
  h1_corrPho_R9_ = new TH1F("corrPhoR9","Corrected photons:  3x3 energy / SuperCluster energy",100,0.,1.2);


}


//========================================================================
void
SimplePhotonAnalyzer::analyze( const edm::Event& evt, const edm::EventSetup& es ) {
//========================================================================

  using namespace edm; // needed for all fwk related classes
  edm::LogInfo("PhotonAnalyzer") << "Analyzing event number: " << evt.id() << "\n";

  //  ----- barrel with island


  // Get the  corrected  photon collection (set in the configuration) which also contains infos about conversions

  Handle<reco::PhotonCollection> photonHandle; 
  evt.getByLabel(photonCollectionProducer_, photonCollection_ , photonHandle);
  const reco::PhotonCollection photonCollection = *(photonHandle.product());


  // Get the primary event vertex
  Handle<reco::VertexCollection> vertexHandle;
  evt.getByLabel(vertexProducer_, vertexHandle);
  reco::VertexCollection vertexCollection = *(vertexHandle.product());
  math::XYZPoint vtx(0.,0.,0.);
  if (vertexCollection.size()>0) vtx = vertexCollection.begin()->position();


  /// Get the MC truth
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

      // loop over uncorrected  Photon candidates 
      for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {

	/////  Set event vertex
	reco::Photon localPho = reco::Photon(*iPho);
	localPho.setVertex(vtx);
	localPhotons.push_back(localPho);

	/// Match reconstructed photon candidates with the nearest generated photonPho;
	float phiClu=localPho.phi();
	float etaClu=localPho.eta();
	float phiPho=(*p)->momentum().phi();
	float etaPho=(*p)->momentum().eta();
	float deltaPhi = phiClu-phiPho;
	float deltaEta = etaClu-etaPho;

	if ( deltaPhi > pi )  deltaPhi -= twopi;
	if ( deltaPhi < -pi) deltaPhi += twopi;
	deltaPhi=pow(deltaPhi,2);
	deltaEta=pow(deltaEta,2);
	float delta = sqrt( deltaPhi+deltaEta); 
	if ( delta<0.1 && delta < minDelta ) {
	  minDelta=delta;
	  iMatch=index;
	}
	index++;
      } // End loop over uncorrected photons

      /// Plot kinematic disctributions for matched photons
      if (iMatch>-1) {
	std::cout << "h1" << std::endl;
	h1_scE_->Fill( localPhotons[iMatch].superCluster()->energy() );
	h1_scEt_->Fill( localPhotons[iMatch].superCluster()->energy()/cosh(localPhotons[iMatch].superCluster()->position().eta()) );
	h1_scEta_->Fill( localPhotons[iMatch].superCluster()->position().eta() );
	h1_scPhi_->Fill( localPhotons[iMatch].superCluster()->position().phi() );
	std::cout << "h2" << std::endl;
	
	h1_phoE_->Fill( localPhotons[iMatch].energy() );
	h1_phoEta_->Fill( localPhotons[iMatch].eta() );
	h1_phoPhi_->Fill( localPhotons[iMatch].phi() );
      }    

      minDelta=10000.;
      localPhotons.clear();
      index=0;
      iMatch=-1;



    } // End loop over MC particles

  }

}

//========================================================================
void
SimplePhotonAnalyzer::endJob() {
//========================================================================


  rootFile_->cd();

  h1_scE_  -> Write();
  h1_scEt_  -> Write();
  h1_scEta_-> Write();
  h1_scPhi_-> Write();


  h1_phoE_  -> Write();
  h1_phoEta_-> Write();
  h1_phoPhi_-> Write();

  h1_recEoverTrueE_     ->  Write();
  h1_deltaEta_ ->  Write();
  h1_deltaPhi_ ->  Write();


  h1_corrPho_E_->Write();
  h1_corrPho_Eta_->Write();
  h1_corrPho_Phi_->Write();
  h1_corrPho_R9_->Write();


  rootFile_->Close();
}
