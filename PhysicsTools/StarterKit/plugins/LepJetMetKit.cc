#include "PhysicsTools/StarterKit/interface/LepJetMetKit.h"

#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"

using namespace std;
using namespace pat;

//
// constructors and destructor
//
LepJetMetKit::LepJetMetKit(const edm::ParameterSet& iConfig) 
  :
  StarterKit( iConfig )
{
  lepJetMetEvent_   = false;

  // ----- Get user supplied parameters: -----

  // --- Histogram parameters ---

  // number of bins on x axis
  hNbinsJetMult_    = iConfig.getUntrackedParameter<unsigned int>("HNbinsJetMult",16);
  hNbinsMuonMult_   = iConfig.getUntrackedParameter<unsigned int>("HNbinsMuonMult",11);
  hNbinsElecMult_   = iConfig.getUntrackedParameter<unsigned int>("HNbinsElecMult",11);
  hNbinsLead2_      = iConfig.getUntrackedParameter<unsigned int>("HNbinsLead2",100);
  hNbinsLead3_      = iConfig.getUntrackedParameter<unsigned int>("HNbinsLead3",100);
  hNbinsLead4_      = iConfig.getUntrackedParameter<unsigned int>("HNbinsLead4",100);

  // x axis minimum
  hMinJetMult_      = iConfig.getUntrackedParameter<double>("HMinJetMult",-0.5);
  hMinMuonMult_     = iConfig.getUntrackedParameter<double>("HMinMuonMult",-0.5);
  hMinElecMult_     = iConfig.getUntrackedParameter<double>("HMinElecMult",-0.5);
  hMinLead2_        = iConfig.getUntrackedParameter<double>("HMinLead2",0.);
  hMinLead3_        = iConfig.getUntrackedParameter<double>("HMinLead3",0.);
  hMinLead4_        = iConfig.getUntrackedParameter<double>("HMinLead4",0.);

  // x axis maximum
  hMaxJetMult_      = iConfig.getUntrackedParameter<double>("HMaxJetMult",15.5);
  hMaxMuonMult_     = iConfig.getUntrackedParameter<double>("HMaxMuonMult",10.5);
  hMaxElecMult_     = iConfig.getUntrackedParameter<double>("HMaxElecMult",10.5);
  hMaxLead2_        = iConfig.getUntrackedParameter<double>("HMaxLead2",1000.);
  hMaxLead3_        = iConfig.getUntrackedParameter<double>("HMaxLead3",1000.);
  hMaxLead4_        = iConfig.getUntrackedParameter<double>("HMaxLead4",1000.);

  // ----- Book histograms -----

  leading2JetsHist_ = new HistoComposite("Leading2Jets", "Two leading jets", "sum");
  leading3JetsHist_ = new HistoComposite("Leading3Jets", "Three leading jets", "sum");
  leading4JetsHist_ = new HistoComposite("Leading4Jets", "Four leading jets", "sum");

  leading2JetsHist_->setNBins(hNbinsLead2_);
  leading3JetsHist_->setNBins(hNbinsLead3_);
  leading4JetsHist_->setNBins(hNbinsLead4_);

  leading2JetsHist_->setPtRange(hMinLead2_, hMaxLead2_);
  leading3JetsHist_->setPtRange(hMinLead3_, hMaxLead3_);
  leading4JetsHist_->setPtRange(hMinLead4_, hMaxLead4_);

  leading2JetsHist_->setMassRange(hMinLead2_, hMaxLead2_);
  leading3JetsHist_->setMassRange(hMinLead3_, hMaxLead3_);
  leading4JetsHist_->setMassRange(hMinLead4_, hMaxLead4_);

  // ----- Name production branch -----

  std::string alias;
  // true if event contains at least one 'valid' lepton+jet+met
  produces<bool>( alias = "LepJetMetEvent" ).setBranchAlias( alias );
  produces<double>( alias = "Leading2JetsMass" ).setBranchAlias( alias );
  produces<double>( alias = "Leading3JetsMass" ).setBranchAlias( alias );
  produces<double>( alias = "Leading4JetsMass" ).setBranchAlias( alias );
}

LepJetMetKit::~LepJetMetKit() 
{
  delete leading2JetsHist_;
  delete leading3JetsHist_;
  delete leading4JetsHist_;
}

//
// member functions
//

// ------------ method called to for each event  ------------
void LepJetMetKit::produce( edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
  using namespace edm;

  StarterKit::produce( iEvent, iSetup );

  // INSIDE OF StarterKit::produce:

  // --------------------------------------------------
  //    Step 1: Retrieve objects from data stream
  // --------------------------------------------------

  // --------------------------------------------------
  //    Step 2: invoke PhysicsHistograms to deal with all this.
  //
  //    Note that each handle will dereference into a vector<>,
  //    however the fillCollection() method takes a reference,
  //    so the collections are not copied...
  // --------------------------------------------------

  // BEGIN LEP+JET+MET CODE

  // --------------------------------------------------
  //    Step 3: Plot event data
  // --------------------------------------------------


  lepJetMetEvent_ = true;

  // Composite objects
  reco::NamedCompositeCandidate leadingJets("leadingJets");
  AddFourMomenta addFourMomenta;

  leadingJets.addDaughter(auto_ptr<reco::Candidate>(new pat::Jet((*jetHandle_)[0])), "jet1");
  if ( jetHandle_->size() > 1 ) {
    leadingJets.addDaughter(auto_ptr<reco::Candidate>(new pat::Jet((*jetHandle_)[1])), "jet2");
    addFourMomenta.set(leadingJets);

    auto_ptr<double> leadingJetsMass( new double(leadingJets.mass()) );
    iEvent.put(leadingJetsMass, "Leading2JetsMass");

    leading2JetsHist_->fill(leadingJets);
  }
  if ( jetHandle_->size() > 2 ) {
    leadingJets.addDaughter(auto_ptr<reco::Candidate>(new pat::Jet((*jetHandle_)[2])), "jet3");
    addFourMomenta.set(leadingJets);

    auto_ptr<double> leadingJetsMass( new double(leadingJets.mass()) );
    iEvent.put(leadingJetsMass, "Leading3JetsMass");

    leading3JetsHist_->fill(leadingJets);
  }
  if ( jetHandle_->size() > 3 ) {
    leadingJets.addDaughter(auto_ptr<reco::Candidate>(new pat::Jet((*jetHandle_)[3])), "jet4");
    addFourMomenta.set(leadingJets);

    auto_ptr<double> leadingJetsMass( new double(leadingJets.mass()) );
    iEvent.put(leadingJetsMass, "Leading4JetsMass");

    leading4JetsHist_->fill(leadingJets);
  }


  iEvent.put(auto_ptr<bool>(new bool(lepJetMetEvent_)), "LepJetMetEvent");
}


// ------------ method called once each job just before starting event loop  ------------
void
LepJetMetKit::beginJob(const edm::EventSetup& iSetup)
{
  StarterKit::beginJob(iSetup);
}



// ------------ method called once each job just after ending the event loop  ------------
void
LepJetMetKit::endJob() {
  StarterKit::endJob();
}

//define this as a plug-in
DEFINE_FWK_MODULE(LepJetMetKit);
