#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h" 
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetProducers/interface/MVAJetPuId.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"


class MVAJetPuIdProducer : public edm::stream::EDProducer <> {
public:
   explicit MVAJetPuIdProducer(const edm::ParameterSet&);

   static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
   virtual void produce(edm::Event&, const edm::EventSetup&) override;


   void initJetEnergyCorrector(const edm::EventSetup &iSetup, bool isData);

   edm::InputTag jets_, vertexes_, jetids_, rho_;
   std::string jec_;
   bool runMvas_, produceJetIds_, inputIsCorrected_, applyJec_;
   std::vector<std::pair<std::string, MVAJetPuId *> > algos_;

   bool residualsFromTxt_;
   edm::FileInPath residualsTxt_;
   FactorizedJetCorrector *jecCor_;
   std::vector<JetCorrectorParameters> jetCorPars_;

   edm::EDGetTokenT<edm::View<reco::Jet> > input_jet_token_;
   edm::EDGetTokenT<reco::VertexCollection> input_vertex_token_;
   edm::EDGetTokenT<edm::ValueMap<StoredPileupJetIdentifier> > input_vm_pujetid_token_;
   edm::EDGetTokenT<double> input_rho_token_;

};

 
MVAJetPuIdProducer::MVAJetPuIdProducer(const edm::ParameterSet& iConfig)
{
     runMvas_ = iConfig.getParameter<bool>("runMvas");
     produceJetIds_ = iConfig.getParameter<bool>("produceJetIds");
     jets_ = iConfig.getParameter<edm::InputTag>("jets");
     vertexes_ = iConfig.getParameter<edm::InputTag>("vertexes");
     jetids_  = iConfig.getParameter<edm::InputTag>("jetids");
     inputIsCorrected_ = iConfig.getParameter<bool>("inputIsCorrected");
     applyJec_ = iConfig.getParameter<bool>("applyJec");
     jec_ =  iConfig.getParameter<std::string>("jec");
     rho_ = iConfig.getParameter<edm::InputTag>("rho");
     residualsFromTxt_ = iConfig.getParameter<bool>("residualsFromTxt");
     if(residualsFromTxt_) residualsTxt_ = iConfig.getParameter<edm::FileInPath>("residualsTxt");
     std::vector<edm::ParameterSet> algos = iConfig.getParameter<std::vector<edm::ParameterSet> >("algos");
     
     jecCor_ = 0;
 
     if( ! runMvas_ ) assert( algos.size() == 1 );
     
     if( produceJetIds_ ) {
         produces<edm::ValueMap<StoredPileupJetIdentifier> > ("");
     }
     for(std::vector<edm::ParameterSet>::iterator it=algos.begin(); it!=algos.end(); ++it) {
         std::string label = it->getParameter<std::string>("label");
         algos_.push_back( std::make_pair(label,new MVAJetPuId(*it)) );
         if( runMvas_ ) {
             produces<edm::ValueMap<float> > (label+"Discriminant");
             produces<edm::ValueMap<int> > (label+"Id");
         }
     }
 
     input_jet_token_ = consumes<edm::View<reco::Jet> >(jets_);
     input_vertex_token_ = consumes<reco::VertexCollection>(vertexes_);
     input_vm_pujetid_token_ = consumes<edm::ValueMap<StoredPileupJetIdentifier> >(jetids_);
     input_rho_token_ = consumes<double>(rho_); 
 
 }
 
 
 void
 MVAJetPuIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
 {
     using namespace edm;
     using namespace std;
     using namespace reco;
     
     Handle<View<Jet> > jetHandle;
     iEvent.getByToken(input_jet_token_,jetHandle);
     const View<Jet> & jets = *jetHandle;
     Handle<VertexCollection> vertexHandle;
     if(  produceJetIds_ ) {
             iEvent.getByToken(input_vertex_token_, vertexHandle);
     }
     const VertexCollection & vertexes = *(vertexHandle.product());
     Handle<ValueMap<StoredPileupJetIdentifier> > vmap;
     if( ! produceJetIds_ ) {
         iEvent.getByToken(input_vm_pujetid_token_, vmap);
     }
     edm::Handle< double > rhoH;
     double rho = 0.;
     
     vector<StoredPileupJetIdentifier> ids; 
     map<string, vector<float> > mvas;
     map<string, vector<int> > idflags;
 
     VertexCollection::const_iterator vtx;
     if( produceJetIds_ ) {
         vtx = vertexes.begin();
         while( vtx != vertexes.end() && ( vtx->isFake() || vtx->ndof() < 4 ) ) {
             ++vtx;
         }
         if( vtx == vertexes.end() ) { vtx = vertexes.begin(); }
    }
    
    for ( unsigned int i=0; i<jets.size(); ++i ) {
        vector<pair<string,MVAJetPuId *> >::iterator algoi = algos_.begin();
        MVAJetPuId * ialgo = algoi->second;
        
        const Jet & jet = jets[i];
         
         float jec = 0.;
         if( applyJec_ ) {
             if( rho == 0. ) {
                 iEvent.getByToken(input_rho_token_,rhoH);
                 rho = *rhoH;
             }
             jecCor_->setJetPt(jet.pt());
             jecCor_->setJetEta(jet.eta());
             jecCor_->setJetA(jet.jetArea());
             jecCor_->setRho(rho);
             jec = jecCor_->getCorrection();
         }
         
         bool applyJec = applyJec_ || !inputIsCorrected_;
         reco::Jet * corrJet = 0;
         if( applyJec ) {
             float scale = jec;
                         corrJet = dynamic_cast<reco::Jet *>( jet.clone() );
             corrJet->scaleEnergy(scale);
         }
         const reco::Jet * theJet = ( applyJec ? corrJet : &jet );
         PileupJetIdentifier puIdentifier;
         if( produceJetIds_ ) {
             puIdentifier = ialgo->computeIdVariables(theJet, jec,  &(*vtx), vertexes, runMvas_);
             ids.push_back( puIdentifier );
         } else {
             puIdentifier = (*vmap)[jets.refAt(i)]; 
             puIdentifier.jetPt(theJet->pt());    /*make sure JEC is applied when computing the MVA*/
             puIdentifier.jetEta(theJet->eta());
             puIdentifier.jetPhi(theJet->phi());
             ialgo->set(puIdentifier); 
             puIdentifier = ialgo->computeMva();
         }
         
         if( runMvas_ ) {
             for( ; algoi!=algos_.end(); ++algoi) {
                 ialgo = algoi->second;
                 ialgo->set(puIdentifier);
                 PileupJetIdentifier id = ialgo->computeMva();
                 mvas[algoi->first].push_back( id.mva() );
                 idflags[algoi->first].push_back( id.idFlag() );
             }
         }
         
         if( corrJet ) { delete corrJet; }
     }
     
     if( runMvas_ ) {
         for(vector<pair<string,MVAJetPuId *> >::iterator ialgo = algos_.begin(); ialgo!=algos_.end(); ++ialgo) {
             vector<float> & mva = mvas[ialgo->first];
             auto_ptr<ValueMap<float> > mvaout(new ValueMap<float>());
             ValueMap<float>::Filler mvafiller(*mvaout);
             mvafiller.insert(jetHandle,mva.begin(),mva.end());
             mvafiller.fill();
             iEvent.put(mvaout,ialgo->first+"Discriminant");
             
             vector<int> & idflag = idflags[ialgo->first];
             auto_ptr<ValueMap<int> > idflagout(new ValueMap<int>());
             ValueMap<int>::Filler idflagfiller(*idflagout);
             idflagfiller.insert(jetHandle,idflag.begin(),idflag.end());
             idflagfiller.fill();
             iEvent.put(idflagout,ialgo->first+"Id");
         }
     }
     if( produceJetIds_ ) {
         assert( jetHandle->size() == ids.size() );
         auto_ptr<ValueMap<StoredPileupJetIdentifier> > idsout(new ValueMap<StoredPileupJetIdentifier>());
         ValueMap<StoredPileupJetIdentifier>::Filler idsfiller(*idsout);
         idsfiller.insert(jetHandle,ids.begin(),ids.end());
         idsfiller.fill();
         iEvent.put(idsout);
     }
 }
 
 
 
 void
 MVAJetPuIdProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("runMvas", true);
  desc.add<bool>("inputIsCorrected", true);
  desc.add<edm::InputTag>("vertexes", edm::InputTag("hltPixelVertices"));
  desc.add<bool>("produceJetIds", true);
  desc.add<std::string>("jec", "AK4PF");
  desc.add<bool>("residualsFromTxt", false);
  desc.add<bool>("applyJec", false);
  desc.add<edm::InputTag>("jetids", edm::InputTag(""));
  desc.add<edm::InputTag>("rho", edm::InputTag("hltFixedGridRhoFastjetAll"));
  desc.add<edm::InputTag>("jets", edm::InputTag("hltAK4PFJetsCorrected"));
    edm::ParameterSetDescription vpsd1;
    vpsd1.add<std::vector<std::string>>("tmvaVariables", {
      "rho",
      "nTot",
      "nCh",
      "axisMajor",
      "axisMinor",
      "fRing0",
      "fRing1",
      "fRing2",
      "fRing3",
      "ptD",
      "beta",
      "betaStar",
      "DR_weighted",
      "pull",
      "jetR",
      "jetRchg",
    });
    vpsd1.add<std::string>("tmvaMethod", "JetID");
    vpsd1.add<bool>("cutBased", false);
    vpsd1.add<std::string>("tmvaWeights", "RecoJets/JetProducers/data/MVAJetPuID.weights.xml.gz");
    vpsd1.add<std::vector<std::string>>("tmvaSpectators", {
      "jetEta",
      "jetPt",
    });
    vpsd1.add<std::string>("label", "CATEv0");
    vpsd1.add<int>("version", -1);
    {
      edm::ParameterSetDescription psd0;
      psd0.add<std::vector<double>>("Pt2030_Tight", {
        0.73,
        0.05,
        -0.26,
        -0.42,
      });
      psd0.add<std::vector<double>>("Pt2030_Loose", {
        -0.63,
        -0.6,
        -0.55,
        -0.45,
      });
      psd0.add<std::vector<double>>("Pt3050_Medium", {
        0.1,
        -0.36,
        -0.54,
        -0.54,
      });
      psd0.add<std::vector<double>>("Pt1020_Tight", {
        -0.83,
        -0.81,
        -0.74,
        -0.81,
      });
      psd0.add<std::vector<double>>("Pt2030_Medium", {
        0.1,
        -0.36,
        -0.54,
        -0.54,
      });
      psd0.add<std::vector<double>>("Pt010_Tight", {
        -0.83,
        -0.81,
        -0.74,
        -0.81,
      });
      psd0.add<std::vector<double>>("Pt1020_Loose", {
        -0.95,
        -0.96,
        -0.94,
        -0.95,
      });
      psd0.add<std::vector<double>>("Pt010_Medium", {
        -0.83,
        -0.92,
        -0.9,
        -0.92,
      });
      psd0.add<std::vector<double>>("Pt1020_Medium", {
        -0.83,
        -0.92,
        -0.9,
        -0.92,
      });
      psd0.add<std::vector<double>>("Pt010_Loose", {
        -0.95,
        -0.96,
        -0.94,
        -0.95,
      });
      psd0.add<std::vector<double>>("Pt3050_Loose", {
        -0.63,
        -0.6,
        -0.55,
        -0.45,
      });
      psd0.add<std::vector<double>>("Pt3050_Tight", {
        0.73,
        0.05,
        -0.26,
        -0.42,
      });
      vpsd1.add<edm::ParameterSetDescription>("JetIdParams", psd0);
    }
    vpsd1.add<double>("impactParTkThreshold", 1.0);
    std::vector<edm::ParameterSet> temp1;
    temp1.reserve(1);

    desc.addVPSet("algos", vpsd1, temp1);

  descriptions.add("MVAJetPuIdProducer", desc);

 }
 
DEFINE_FWK_MODULE(MVAJetPuIdProducer);


