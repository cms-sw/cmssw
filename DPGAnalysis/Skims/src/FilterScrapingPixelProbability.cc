
// Original Author: Gavril Giurgiu (JHU) 

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "DPGAnalysis/Skims/interface/FilterScrapingPixelProbability.h"

using namespace edm;
using namespace std;

FilterScrapingPixelProbability::FilterScrapingPixelProbability(const edm::ParameterSet& iConfig)
{
  //std::cout << "FilterScrapingPixelProbability::FilterScrapingPixelProbability " << std::endl;

  apply_filter                 = iConfig.getUntrackedParameter<bool>  ( "apply_filter"                , true  );
  select_collision             = iConfig.getUntrackedParameter<bool>  ( "select_collision"            , true  );
  select_pkam                  = iConfig.getUntrackedParameter<bool>  ( "select_pkam"                 , false );
  select_other                 = iConfig.getUntrackedParameter<bool>  ( "select_other"                , false );
  low_probability              = iConfig.getUntrackedParameter<double>( "low_probability"             , 0.0   );
  low_probability_fraction_cut = iConfig.getUntrackedParameter<double>( "low_probability_fraction_cut", 0.4   ); 
  tracks_ = iConfig.getUntrackedParameter<edm::InputTag>("src",edm::InputTag("generalTracks"));
}

FilterScrapingPixelProbability::~FilterScrapingPixelProbability()
{
}

bool FilterScrapingPixelProbability::filter( edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  bool accepted = false;
 
  double low_probability_fraction = -999999.9;  
 
  float n_hits_low_prob = 0.0;
  float n_hits_barrel   = 0.0;

  // Loop over generalTracks
  edm::Handle<reco::TrackCollection> trackCollection;
  iEvent.getByLabel(tracks_, trackCollection);
  const reco::TrackCollection *tracks = trackCollection.product();
  reco::TrackCollection::const_iterator tciter;

  //std::cout << "(int)tracks->size() = " << (int)tracks->size() << std::endl;
  
  if ( (int)tracks->size() > 0 )
    {
      // Loop on tracks
      for ( tciter=tracks->begin(); tciter!=tracks->end(); ++tciter)
	{
	  // First loop on hits: find pixel hits
	  for ( trackingRecHit_iterator it = tciter->recHitsBegin(); it != tciter->recHitsEnd(); ++it) 
	    {
	      const TrackingRecHit &thit = **it;
	      
	      const SiPixelRecHit* matchedhit = dynamic_cast<const SiPixelRecHit*>(&thit);
              
	      // Check if the RecHit is a PixelRecHit
	      if ( matchedhit ) 
		{
		  DetId detId = (*it)->geographicalId();
		  
		  // Only consider barrel pixel hits
		  if ( (int)detId.subdetId() == (int)PixelSubdetector::PixelBarrel ) 
		    {
		      n_hits_barrel = n_hits_barrel + 1.0;

		      double pixel_hit_probability = matchedhit->clusterProbability(0);
		       
		      if ( pixel_hit_probability <= 0.0 )
			n_hits_low_prob = n_hits_low_prob + 1.0; 
		    
		    } // if ( (int)detId.subdetId() == (int)PixelSubdetector::PixelBarrel ) 

		} // if ( matchedhit )

	    } // for ( trackingRecHit_iterator it = tciter->recHitsBegin(); it != tciter->recHitsEnd(); ++it) 
	  
	} //  for ( tciter=tracks->begin(); tciter!=tracks->end(); ++tciter) 
      
    } // if ( (int)tracks->size() > 0 )

  bool is_collision = false;
  bool is_pkam      = false;
  bool is_other     = false;

  if ( n_hits_barrel > 0.0 )
    {
      low_probability_fraction = n_hits_low_prob / n_hits_barrel;

      if ( low_probability_fraction < 0.4 )
	is_collision = true;
      else 
	is_pkam = true;
    }
  else
    is_other = true;
  
  if ( ( select_collision && is_collision ) || 
       ( select_pkam      && is_pkam      ) ||
       ( select_other     && is_other     ) )
    accepted = true;
 
  if ( apply_filter )
    return accepted;
  else
    return true;

}

//define this as a plug-in
DEFINE_FWK_MODULE(FilterScrapingPixelProbability);
