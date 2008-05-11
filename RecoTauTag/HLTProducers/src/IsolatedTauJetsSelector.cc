#include "RecoTauTag/HLTProducers/interface/IsolatedTauJetsSelector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "Math/GenVector/VectorUtil.h"
//
// class decleration
//


IsolatedTauJetsSelector::IsolatedTauJetsSelector(const edm::ParameterSet& iConfig)
{

  jetSrc = iConfig.getParameter<vtag>("JetSrc");
  pt_min_leadTrack   = iConfig.getParameter<double>("MinimumTransverseMomentumLeadingTrack"); 
useIsolationDiscriminator = iConfig.getParameter<bool>("UseIsolationDiscriminator");
 useInHLTOpen       = iConfig.getParameter<bool>("UseInHLTOpen");
 
  produces<reco::CaloJetCollection>();
  //  produces<reco::IsolatedTauTagInfoCollection>();  
}

IsolatedTauJetsSelector::~IsolatedTauJetsSelector(){ }

void IsolatedTauJetsSelector::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

  using namespace reco;
  using namespace edm;
  using namespace std;
  
  CaloJetCollection* myJetCollection = new CaloJetCollection;
  CaloJetCollection * jetCollection =new CaloJetCollection;
  CaloJetCollection * jetCollectionTmp = new CaloJetCollection;
    IsolatedTauTagInfoCollection * extendedCollection = new IsolatedTauTagInfoCollection;

    //matching cone
    float mc_cone = 0.1;

  for( vtag::const_iterator s = jetSrc.begin(); s != jetSrc.end(); ++ s ) {
    edm::Handle<IsolatedTauTagInfoCollection> tauJets;
    iEvent.getByLabel( * s, tauJets );
    IsolatedTauTagInfoCollection::const_iterator i = tauJets->begin();
    for(;i !=tauJets->end(); i++ ) {

      if(useInHLTOpen) {
	const CaloJet* pippo = dynamic_cast<const CaloJet*>((i->jet().get()));
	jetCollectionTmp->push_back(*pippo);
	extendedCollection->push_back(*(i)); //to  be used later
	//	delete pippo;
      }else{
	const TrackRef leadTk = i->leadingSignalTrack(mc_cone, pt_min_leadTrack);
	if( !leadTk ) 
	  {}else{	   
	  float discriminator = i->discriminator();	  
	  const CaloJet* pippo = dynamic_cast<const CaloJet*>((i->jet().get()));
	  if(useIsolationDiscriminator && (discriminator > 0) ) {
	    jetCollectionTmp->push_back(*pippo);
	    extendedCollection->push_back(*(i)); //to  be used later
	    // delete pippo;
	  }else if(!useIsolationDiscriminator){
	    jetCollectionTmp->push_back(*pippo);
	    extendedCollection->push_back(*(i)); //to  be used later
	    
	  }
	}
      }
    }	  
  }


  /*
  if(useVertex){//We have to select jets which comes from the same vertex

    //Using vertex constraint needed for Pixel
    VertexCollection::const_iterator myVertex = vertices->begin();

    for(;myVertex!= vertices->end();myVertex++){
      

      IsolatedTauTagInfoCollection::const_iterator myIsolJet = extendedCollection->begin();
      int taggedJets=0;
      for(;myIsolJet!=extendedCollection->end();myIsolJet++){
	const TrackRef leadTk = myIsolJet->leadingSignalTrack(matching_cone, pt_min_leadTrack);
	double deltaZLeadTrackVertex = fabs(myVertex->z() - leadTk->dz());
	//Check leadingTrack_z_imp and PV list
	if(deltaZLeadTrackVertex <dZ_vertex)
	  {
	    JetTracksAssociationRef     jetTracks = myIsolJet->jtaRef();
	    taggedJets++;
	    const CaloJet* pippo = dynamic_cast<const CaloJet*>(myIsolJet->jet().get());
	    myJetCollection->push_back(*pippo);
	    //	    delete pippo;
	  }
      }
      //check if we have at least 2 jets from the same vertex
      //      cout <<"Tagged Jets with vertex constr "<<taggedJets<<endl;
      if(taggedJets > 1) 
	{
	  jetCollection = myJetCollection;
	  break;
	}
    }
  }else{// no Vertex constraint is used 
    
jetCollection = jetCollectionTmp;
  }
  */
  jetCollection = jetCollectionTmp;


  
  auto_ptr<reco::CaloJetCollection> selectedTaus(jetCollection);
  auto_ptr<reco::IsolatedTauTagInfoCollection> extColl(extendedCollection);
  
  iEvent.put(extColl);
  iEvent.put(selectedTaus);


}
