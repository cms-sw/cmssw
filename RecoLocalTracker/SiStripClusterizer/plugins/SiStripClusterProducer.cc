#include "RecoLocalTracker/SiStripClusterizer/plugins/SiStripClusterProducer.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerFactory.h"

// -----------------------------------------------------------------------------
//
SiStripClusterProducer::SiStripClusterProducer( edm::ParameterSet const& pset )
  : clusterizer_( new SiStripClusterizerFactory(pset) ),
    tag_( pset.getParameter<edm::InputTag>("ProductLabel") ),
    edmNew_( pset.getParameter<bool>("DetSetVectorNew") )
{
  if ( !edmNew_ ) { produces<ClustersDSV>(); }
  else { produces<ClustersDSVnew>(); } 
}

// -----------------------------------------------------------------------------
//
SiStripClusterProducer::~SiStripClusterProducer() { 
  if ( clusterizer_ ) { delete clusterizer_; }
}  

// -----------------------------------------------------------------------------
//
void SiStripClusterProducer::beginRun( edm::Run& event,
				       const edm::EventSetup& setup ) {
  if ( clusterizer_ ) { clusterizer_->eventSetup( setup ); }
}

// -----------------------------------------------------------------------------
//
void SiStripClusterProducer::produce( edm::Event& event,
				      const edm::EventSetup& setup ) {
  edm::Handle<DigisDSV> input;
  event.getByLabel( tag_, input ); 
  if ( !edmNew_ ) {
    std::auto_ptr<ClustersDSV> output( new ClustersDSV() );
    if ( clusterizer_ ) {
      clusterizer_->eventSetup( setup );
      clusterizer_->clusterize( *input, *output );
    }
    event.put(output);
  } else {
    std::auto_ptr<ClustersDSVnew> output( new ClustersDSVnew() );
    if ( clusterizer_ ) {
      clusterizer_->eventSetup( setup );
      clusterizer_->clusterize( *input, *output );
    }
    event.put(output);
  }
}
