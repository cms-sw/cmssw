 #include <memory>
 #include "HLTrigger/JetMET/plugins/MVAJetPuIdProducer.h"
 
 
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
 
 
 
 MVAJetPuIdProducer::~MVAJetPuIdProducer()
 {
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
        
        const Jet & jet = jets.at(i);
         
         float jec = 0.;
         if( applyJec_ ) {
             if( rho == 0. ) {
                 iEvent.getByToken(input_rho_token_,rhoH);
                 rho = *rhoH;
             }
             if( jecCor_ == 0 ) {
                 //initJetEnergyCorrector( iSetup, iEvent.isRealData() );
             }
                       jecCor_->setJetPt(jet.pt());
             jecCor_->setJetEta(jet.eta());
             jecCor_->setJetA(jet.jetArea());
             jecCor_->setRho(rho);
             jec = jecCor_->getCorrection();
         }
         
         bool applyJec = applyJec_ || !inputIsCorrected_;  //( ! ispat && ! inputIsCorrected_ );
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
             puIdentifier.jetPt(theJet->pt());    // make sure JEC is applied when computing the MVA
             puIdentifier.jetEta(theJet->eta());
             puIdentifier.jetPhi(theJet->phi());
             ialgo->set(puIdentifier); 
             puIdentifier = ialgo->computeMva();
         }
         
         if( runMvas_ ) {
             mvas[algoi->first].push_back( puIdentifier.mva() );
             idflags[algoi->first].push_back( puIdentifier.idFlag() );
             for( ++algoi; algoi!=algos_.end(); ++algoi) {
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
     desc.setUnknown();
     descriptions.addDefault(desc);
 }
/* 
 
 void 
 MVAJetPuIdProducer::initJetEnergyCorrector(const edm::EventSetup &iSetup, bool isData)
 {
     std::vector<std::string> jecLevels;
     jecLevels.push_back("L1FastJet");
     jecLevels.push_back("L2Relative");
     jecLevels.push_back("L3Absolute");
     if(isData && ! residualsFromTxt_ ) jecLevels.push_back("L2L3Residual");
 
     edm::ESHandle<JetCorrectorParametersCollection> parameters;
     iSetup.get<JetCorrectionsRecord>().get(jec_,parameters);
     for(std::vector<std::string>::const_iterator ll = jecLevels.begin(); ll != jecLevels.end(); ++ll)
     { 
         const JetCorrectorParameters& ip = (*parameters)[*ll];
         jetCorPars_.push_back(ip); 
     } 
     if( isData && residualsFromTxt_ ) {
         jetCorPars_.push_back(JetCorrectorParameters(residualsTxt_.fullPath())); 
     }
     
     jecCor_ = new FactorizedJetCorrector(jetCorPars_);
 }
*/
 DEFINE_FWK_MODULE(MVAJetPuIdProducer);

