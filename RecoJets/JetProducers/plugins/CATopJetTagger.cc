#include "CATopJetTagger.h"
#include "RecoJets/JetAlgorithms/interface/CATopJetHelper.h"
#include "DataFormats/BTauReco/interface/CATopJetTagInfo.h"

using namespace std;
using namespace reco;
using namespace edm;
//
// constants, enums and typedefs
//

//
// static data member definitions
//


//
// constructors and destructor
//
CATopJetTagger::CATopJetTagger(const edm::ParameterSet& iConfig):
  src_(iConfig.getParameter<InputTag>("src") ),
  TopMass_(iConfig.getParameter<double>("TopMass") ),
  WMass_(iConfig.getParameter<double>("WMass") ),
  verbose_(iConfig.getParameter<bool>("verbose") ),
  input_jet_token_(consumes<edm::View<reco::Jet> >(src_))
{
  produces<CATopJetTagInfoCollection>();
}


CATopJetTagger::~CATopJetTagger()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
CATopJetTagger::produce( edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{

  // Set up output list
  auto_ptr<CATopJetTagInfoCollection> tagInfos(new CATopJetTagInfoCollection() );

  // Get the input list of basic jets corresponding to the hard jets
  Handle<View<Jet> > pBasicJets;
  iEvent.getByToken(input_jet_token_, pBasicJets);

  // Get a convenient handle
  View<Jet> const & hardJets = *pBasicJets;

  CATopJetHelper helper( TopMass_, WMass_ );
   
  // Now loop over the hard jets and do kinematic cuts
  View<Jet>::const_iterator ihardJet = hardJets.begin(),
    ihardJetEnd = hardJets.end();
  size_t iihardJet = 0;
  for ( ; ihardJet != ihardJetEnd; ++ihardJet, ++iihardJet ) {

    if ( verbose_ ) edm::LogInfo("CATopJetTagger") << "Processing ihardJet with pt = " << ihardJet->pt() << endl;

    // Initialize output variables
    // Get a ref to the hard jet
    RefToBase<Jet> ref( pBasicJets, iihardJet );    
    // Get properties
    CATopJetProperties properties = helper( *ihardJet );
    
    CATopJetTagInfo tagInfo;
    tagInfo.insert( ref, properties );
    tagInfos->push_back( tagInfo );
  }// end loop over hard jets
  
  iEvent.put( tagInfos );
 
  return;   
}

//define this as a plug-in
DEFINE_FWK_MODULE(CATopJetTagger);
