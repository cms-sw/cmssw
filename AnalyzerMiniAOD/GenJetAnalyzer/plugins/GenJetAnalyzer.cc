// -*- C++ -*-
//
// Package:    AnalyzerMiniAOD/GenJetAnalyzer
// Class:      GenJetAnalyzer
// 
/**\class GenJetAnalyzer GenJetAnalyzer.cc AnalyzerMiniAOD/GenJetAnalyzer/plugins/GenJetAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Giulio Mandorli
//         Created:  Wed, 19 Jul 2017 15:14:32 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TTree.h"
#include <TVector3.h>
#include <TLorentzVector.h>



#include <DataFormats/JetReco/interface/GenJet.h>
#include <DataFormats/HepMCCandidate/interface/GenParticle.h>
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

class GenJetAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit GenJetAnalyzer(const edm::ParameterSet&);
      ~GenJetAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------
           
      
      
        edm::EDGetTokenT<std::vector<reco::GenJet> > slimmedGenJetToken;
        edm::EDGetTokenT<std::vector<reco::GenParticle> > prunedGenParticlesToken;

  
  
        TTree *tree;
        edm::Service<TFileService> file;	
      
             
        
        
        int nGenMu;
        float GenMu_pt[30];
        float GenMu_eta[30];
        float GenMu_phi[30];
        float GenMu_E[30];
        float GenMu_mass[30];   
        float Mll;

        int nJet;
        int nJet_iso03;
        float Jet_pt[30];
        float Jet_eta[30];
        float Jet_phi[30];
        float Jet_E[30];
        float Jet_mass[30];
        float Mjj;
        float Mjj_iso03;


        
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
GenJetAnalyzer::GenJetAnalyzer(const edm::ParameterSet& iConfig)

{
  slimmedGenJetToken = consumes<std::vector<reco::GenJet> >(edm::InputTag("ak4GenJets"));
  prunedGenParticlesToken = consumes<std::vector<reco::GenParticle> >(edm::InputTag("genParticles"));

     
  
    usesResource("TFileService");
   
   
    tree=file->make<TTree>("tree","tree");

    
    tree->Branch("nGenMu", &nGenMu, "numberOfMu/I"); //"Jet_pt[10]/F"

    tree->Branch("GenMu_pt", &GenMu_pt, "GenMu_pt[30]/F"); //"Jet_pt[10]/F"
    tree->Branch("GenMu_eta", &GenMu_eta, "GenMu_eta[30]/F");
    tree->Branch("GenMu_phi", &GenMu_phi, "GenMu_phi[30]/F");
    tree->Branch("GenMu_E", &GenMu_E, "GenMu_E[30]/F");
    tree->Branch("GenMu_mass", &GenMu_mass, "GenMu_mass[30]/F");
    
    tree->Branch("GenMu_Mll", &Mll, "GenMu_Mll/F");


    tree->Branch("nGenJet", &nJet, "numberOfJet/I"); 
    tree->Branch("nGenJet_iso03", &nJet_iso03, "numberOfJet_iso03/I"); 

    tree->Branch("GenJet_pt", &Jet_pt, "GenJet_pt[30]/F"); //"Jet_pt[10]/F"
    tree->Branch("GenJet_eta", &Jet_eta, "GenJet_eta[30]/F");
    tree->Branch("GenJet_phi", &Jet_phi, "GenJet_phi[30]/F");
    tree->Branch("GenJet_E", &Jet_E, "GenJet_E[30]/F");
    tree->Branch("GenJet_mass", &Jet_mass, "GenJet_mass[30]/F");


    tree->Branch("GenJet_Mjj", &Mjj, "GenJet_Mjj/F");
    tree->Branch("GenJet_Mjj_iso03", &Mjj_iso03, "GenJet_Mjj_iso03/F");

}


GenJetAnalyzer::~GenJetAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
GenJetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;


        nGenMu = 0;
        nJet = 0;
        nJet_iso03 = 0;
        Mjj = 0;
        Mjj_iso03 = 0;
        int MuIdx=0;
        
        for (int n=0; n<30;++n) {
            Jet_pt[n]=0;
            Jet_eta[n]=0;
            Jet_phi[n]=0;
            Jet_E[n]=0;
            Jet_mass[n]=0;

            


            GenMu_pt[n]=0;
            GenMu_eta[n]=0;
            GenMu_phi[n]=0;
            GenMu_E[n]=0;
            GenMu_mass[n]=0;  
        }



        Handle<std::vector<reco::GenJet>> jetsCollection;
        iEvent.getByToken(slimmedGenJetToken, jetsCollection);
        std::vector<reco::GenJet>  jets = *jetsCollection.product();
        

        
        Handle<std::vector<reco::GenParticle>> genParticelesCollection;
        iEvent.getByToken(prunedGenParticlesToken, genParticelesCollection);
        auto  genParticles = *genParticelesCollection.product();
        
        
        
        
        
        std::vector<TLorentzVector> mu_vector;
        for(std::vector<reco::GenParticle>::const_iterator pit = genParticles.begin() ; pit != genParticles.end() && MuIdx < 30 ; ++pit) {

            if (abs(pit->pdgId()) == 13 ){//&& pit->isHardProcess()) {
                GenMu_pt[MuIdx] = pit->p4().pt();
                GenMu_eta[MuIdx] = pit->p4().Eta();
                GenMu_phi[MuIdx] = pit->p4().Phi();
                GenMu_E[MuIdx] = pit->p4().E();
                GenMu_mass[MuIdx] = pit->p4().M(); 
                ++MuIdx;

                TLorentzVector tmpVector;
                tmpVector.SetPtEtaPhiM(pit->p4().pt(), pit->p4().Eta(), pit->p4().Phi(), pit->p4().M());
                mu_vector.push_back(tmpVector);
            }
        }
        
        
        
        nGenMu = mu_vector.size();
        
        
        std::vector<TLorentzVector> jets_noLep03; 
        nJet = jets.size();
        int jet_iso03_idx=0;
        int n = 0;
        if (nGenMu > 1 ) {
            Mll = (mu_vector[0] + mu_vector[1]).M();
            for(std::vector<reco::GenJet>::const_iterator jit = jets.begin() ; jit != jets.end() && jet_iso03_idx < 30 ; ++jit) {

            if ( jit->pt()>5 && abs(jit->eta())<5) {


                if (jit->pt() < -1000) std::cout  << std::endl << n << " \t " << jit->pt() << std::endl;
        
                TLorentzVector tmpVector;
                tmpVector.SetPtEtaPhiM(jit->pt(),jit->eta(),jit->phi(),jit->mass());
                

                
                    float theta1 = mu_vector[0].DeltaR(tmpVector);
                    float theta2 = mu_vector[1].DeltaR(tmpVector);
                    if (theta1 > 0.3 && theta2 > 0.3) {
                        jets_noLep03.push_back(tmpVector);
                        Jet_pt[jet_iso03_idx] = jit->pt();
                        Jet_eta[jet_iso03_idx] = jit->eta();
                        Jet_phi[jet_iso03_idx] = jit->phi();
                        Jet_E[jet_iso03_idx] = jit->energy();
                        Jet_mass[jet_iso03_idx] = jit->mass();   
                        ++jet_iso03_idx;
                    }
                    
                
                
                    if (jets_noLep03.size() == 2 ) {
                        Mjj_iso03 = (jets_noLep03[0] + jets_noLep03[1]).M();
                    }
                
                }
            ++n;            
                
            }         
        } //end nGenMu > 1 condition
        
        nJet_iso03 = jets_noLep03.size();
    
        tree->Fill();
        
        
}


// ------------ method called once each job just before starting event loop  ------------
void 
GenJetAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
GenJetAnalyzer::endJob() 
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
GenJetAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GenJetAnalyzer);
