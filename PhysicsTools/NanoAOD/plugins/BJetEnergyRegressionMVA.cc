//
// Original Author:  Andre Rizzi
//         Created:  Mon, 07 Sep 2017 09:18:03 GMT
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"


#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

#include "PhysicsTools/PatAlgos/interface/BaseMVAValueMapProducer.h"
#include <vector>

class BJetEnergyRegressionMVA : public BaseMVAValueMapProducer<pat::Jet> {
	public:
	  explicit BJetEnergyRegressionMVA(const edm::ParameterSet &iConfig):
		BaseMVAValueMapProducer<pat::Jet>(iConfig),
    		pvsrc_(edm::stream::EDProducer<>::consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("pvsrc"))),
    		svsrc_(edm::stream::EDProducer<>::consumes<edm::View<reco::VertexCompositePtrCandidate>> (iConfig.getParameter<edm::InputTag>("svsrc"))),
    		rhosrc_(edm::stream::EDProducer<>::consumes<double> (iConfig.getParameter<edm::InputTag>("rhosrc")))

	  {

		
	  }
	  void readAdditionalCollections(edm::Event&iEvent, const edm::EventSetup&) override {
		iEvent.getByToken(pvsrc_, pvs_);
		iEvent.getByToken(svsrc_, svs_);
        iEvent.getByToken(rhosrc_,rho_);

      }

    void fillAdditionalVariables(const pat::Jet&j)  override {

    this->setValue("nPVs",pvs_->size());
    this->setValue("rho",*(rho_.product()));

    
    float cone_boundaries[] = { 0.05, 0.1, 0.2, 0.3,0.4 }; 
    size_t ncone_boundaries = sizeof(cone_boundaries)/sizeof(float);
    std::vector<float> emFractionEnergyRings(ncone_boundaries+1);
    std::vector<float> chFractionEnergyRings(ncone_boundaries+1);
    std::vector<float> neFractionEnergyRings(ncone_boundaries+1);
    std::vector<float> muFractionEnergyRings(ncone_boundaries+1);
    float jetRawEnergy=j.p4().E()*j.jecFactor("Uncorrected");
    int numDaughtersPt03=0;
        for (unsigned int ijcone = 0; ijcone<ncone_boundaries; ijcone++){
            emFractionEnergyRings[ijcone] = 0;
            muFractionEnergyRings[ijcone] = 0;
            chFractionEnergyRings[ijcone] = 0;
            neFractionEnergyRings[ijcone] = 0;
        }
        for(const auto & d : j.daughterPtrVector()){
           float candDr   = Geom::deltaR(d->p4(),j.p4());
           size_t icone = std::lower_bound(&cone_boundaries[0],&cone_boundaries[ncone_boundaries],candDr) - &cone_boundaries[0];
           float candEnergy = d->energy()/jetRawEnergy;
           int pdgid = abs(d->pdgId()) ;
           if( pdgid == 22 || pdgid == 11 ) {
               emFractionEnergyRings[icone] += candEnergy;
            } else if ( pdgid == 13 ) { 
               muFractionEnergyRings[icone] += candEnergy;
           } else if ( d->charge() != 0 ) {
               chFractionEnergyRings[icone] += candEnergy;
           } else {
               neFractionEnergyRings[icone] += candEnergy;
           } 
           if(d->pt()>0.3) numDaughtersPt03+=1;
         } // end of jet daughters loop

    this->setValue("Jet_energyRing_dR0_em_Jet_rawEnergy", emFractionEnergyRings[0]);
    this->setValue("Jet_energyRing_dR1_em_Jet_rawEnergy", emFractionEnergyRings[1]);
    this->setValue("Jet_energyRing_dR2_em_Jet_rawEnergy", emFractionEnergyRings[2]);
    this->setValue("Jet_energyRing_dR3_em_Jet_rawEnergy", emFractionEnergyRings[3]);
    this->setValue("Jet_energyRing_dR4_em_Jet_rawEnergy", emFractionEnergyRings[4]);
  //  this->setValue("Jet_energyRing_dR5_em_Jet_rawEnergy", emFractionEnergyRings[5]);
    
    this->setValue("Jet_energyRing_dR0_ch_Jet_rawEnergy", chFractionEnergyRings[0]);
    this->setValue("Jet_energyRing_dR1_ch_Jet_rawEnergy", chFractionEnergyRings[1]);
    this->setValue("Jet_energyRing_dR2_ch_Jet_rawEnergy", chFractionEnergyRings[2]);
    this->setValue("Jet_energyRing_dR3_ch_Jet_rawEnergy", chFractionEnergyRings[3]);
    this->setValue("Jet_energyRing_dR4_ch_Jet_rawEnergy", chFractionEnergyRings[4]);
  //  this->setValue("Jet_energyRing_dR5_ch_Jet_rawEnergy", chFractionEnergyRings[5]);
    
    this->setValue("Jet_energyRing_dR0_mu_Jet_rawEnergy", muFractionEnergyRings[0]);
    this->setValue("Jet_energyRing_dR1_mu_Jet_rawEnergy", muFractionEnergyRings[1]);
    this->setValue("Jet_energyRing_dR2_mu_Jet_rawEnergy", muFractionEnergyRings[2]);
    this->setValue("Jet_energyRing_dR3_mu_Jet_rawEnergy", muFractionEnergyRings[3]);
    this->setValue("Jet_energyRing_dR4_mu_Jet_rawEnergy", muFractionEnergyRings[4]);
  //  this->setValue("Jet_energyRing_dR5_mu_Jet_rawEnergy", muFractionEnergyRings[5]);
    
    this->setValue("Jet_energyRing_dR0_neut_Jet_rawEnergy", neFractionEnergyRings[0]);
    this->setValue("Jet_energyRing_dR1_neut_Jet_rawEnergy", neFractionEnergyRings[1]);
    this->setValue("Jet_energyRing_dR2_neut_Jet_rawEnergy", neFractionEnergyRings[2]);
    this->setValue("Jet_energyRing_dR3_neut_Jet_rawEnergy", neFractionEnergyRings[3]);
    this->setValue("Jet_energyRing_dR4_neut_Jet_rawEnergy", neFractionEnergyRings[4]);
//    this->setValue("Jet_energyRing_dR5_neut_Jet_rawEnergy", neFractionEnergyRings[5]);

    this->setValue("Jet_numDaughters_pt03",numDaughtersPt03);

	  }

          static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
            edm::ParameterSetDescription desc = BaseMVAValueMapProducer<pat::Jet>::getDescription();
            desc.add<edm::InputTag>("pvsrc")->setComment("primary vertices input collection");
            desc.add<edm::InputTag>("svsrc")->setComment("secondary vertices input collection");
            desc.add<edm::InputTag>("rhosrc")->setComment("rho  input collection");
            descriptions.add("BJetEnergyRegressionMVA",desc);
          }

        private:
	  const edm::EDGetTokenT<std::vector<reco::Vertex>> pvsrc_;
 	  edm::Handle<std::vector<reco::Vertex>> pvs_;
      const edm::EDGetTokenT<edm::View<reco::VertexCompositePtrCandidate> > svsrc_;
 	  edm::Handle<edm::View<reco::VertexCompositePtrCandidate>> svs_;
      edm::EDGetTokenT<double> rhosrc_;
      edm::Handle<double> rho_;

	  
};

//define this as a plug-in
DEFINE_FWK_MODULE(BJetEnergyRegressionMVA);

