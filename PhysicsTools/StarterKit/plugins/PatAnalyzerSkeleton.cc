#include "PhysicsTools/StarterKit/interface/PatAnalyzerSkeleton.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PatAnalyzerSkeleton::PatAnalyzerSkeleton(const edm::ParameterSet& iConfig):
  histocontainer_(),
  eleLabel_(iConfig.getUntrackedParameter<edm::InputTag>("electronTag")),
  muoLabel_(iConfig.getUntrackedParameter<edm::InputTag>("muonTag")),
  jetLabel_(iConfig.getUntrackedParameter<edm::InputTag>("jetTag")),
  tauLabel_(iConfig.getUntrackedParameter<edm::InputTag>("tauTag")),
  metLabel_(iConfig.getUntrackedParameter<edm::InputTag>("metTag")),
  phoLabel_(iConfig.getUntrackedParameter<edm::InputTag>("photonTag"))

{
   //now do what ever initialization is needed

}


PatAnalyzerSkeleton::~PatAnalyzerSkeleton()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
PatAnalyzerSkeleton::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;


   //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}
   // first: get all objects from the event.
   //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}

   edm::Handle<edm::View<pat::Muon> > muonHandle;
   iEvent.getByLabel(muoLabel_,muonHandle);
   edm::View<pat::Muon> muons = *muonHandle;
   
   edm::Handle<edm::View<pat::Jet> > jetHandle;
   iEvent.getByLabel(jetLabel_,jetHandle);
   edm::View<pat::Jet> jets = *jetHandle;

   edm::Handle<edm::View<pat::Electron> > electronHandle;
   iEvent.getByLabel(eleLabel_,electronHandle);
   edm::View<pat::Electron> electrons = *electronHandle;

   edm::Handle<edm::View<pat::MET> > metHandle;
   iEvent.getByLabel(metLabel_,metHandle);
   edm::View<pat::MET> mets = *metHandle;

   edm::Handle<edm::View<pat::Photon> > phoHandle;
   iEvent.getByLabel(phoLabel_,phoHandle);
   edm::View<pat::Photon> photons = *phoHandle;

   edm::Handle<edm::View<pat::Tau> > tauHandle;
   iEvent.getByLabel(tauLabel_,tauHandle);
   edm::View<pat::Tau> taus = *tauHandle;

   //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}
   // example of a loop over objects... this works identical for all vectors defined above
   //   once you have a jet object you can use all methods defined in the header file 
   // (DataFormats/PatCandidates/interface/Jet.h and equivalent for other objects)
   //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}
   size_t njetscounter=0;
   for(edm::View<pat::Jet>::const_iterator jet_iter = jets.begin(); jet_iter!=jets.end(); ++jet_iter){
     if(jet_iter->pt()>50)
       njetscounter++;
     
   }

   //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}
   //histocontainer_ is of type std::map<edm::View, TH1D*>. This means you can use it with this syntax:
   // histocontainer_["histname"]->Fill(x); 
   // histocontainer_["histname"]->Draw(); 
   // etc, etc. Essentially you use the histname string to look up a pointer to a TH1D* 
   // which you can do everything to you would normally do in ROOT.
   //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}
   
   histocontainer_["njets"]->Fill(njetscounter);
   //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}
   // for the other objects just quickly book the multiplicity. Again, just use the same infrastructure as for jets if you want to loop over them.
   //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}
   histocontainer_["nelectrons"]->Fill(electrons.size());
   histocontainer_["nphotons"]->Fill(photons.size());
   histocontainer_["nmuons"]->Fill(muons.size());
   histocontainer_["ntaus"]->Fill(taus.size());
}
// ------------ method called once each job just before starting event loop  ------------
void 
PatAnalyzerSkeleton::beginJob(const edm::EventSetup&)
{
  //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}
  // define some histograms using the framework tfileservice. Define the output file name in your .cfg.
  //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}
  edm::Service<TFileService> fs;
  
  //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}
  //histocontainer_ is of type std::map<std::string, TH1D*>. This means you can use it with this syntax:
  // histocontainer_["histname"]->Fill(x); 
  // histocontainer_["histname"]->Draw(); 
  // etc, etc. Essentially you use the histname string to look up a pointer to a TH1D* 
  // which you can do everything to you would normally do in ROOT.
  //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}

  
  //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}
  // here we book new histograms:
  //{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}

  histocontainer_["njets"]=fs->make<TH1D>("njets","jet multiplicity for jets with p_{T} > 50 GeV/c",10,0,10);
  histocontainer_["nelectrons"]=fs->make<TH1D>("nelectrons","electron multiplicity",10,0,10);
  histocontainer_["ntaus"]=fs->make<TH1D>("ntaus","tau multiplicity",10,0,10);
  histocontainer_["nphotons"]=fs->make<TH1D>("nphotons","photon multiplicity",10,0,10);
  histocontainer_["nmuons"]=fs->make<TH1D>("nmuons","muon multiplicity",10,0,10);

}

// ------------ method called once each job just after ending the event loop  ------------
void 
PatAnalyzerSkeleton::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(PatAnalyzerSkeleton);
