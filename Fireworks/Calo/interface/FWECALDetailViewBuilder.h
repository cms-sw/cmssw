
#include "TEveViewer.h"
#include "TEveScene.h"
#include "TEveManager.h"
#include "TEveCaloData.h"
#include "TEveCalo.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/FWLite/interface/Event.h"

#include <map>
#include <vector>
#include <algorithm>

// Less than operator for sorting clusters according to eta
class superClsterEtaLess : public std::binary_function<const reco::CaloCluster&, const reco::CaloCluster&, bool>
{
public:
	bool operator()(const reco::CaloCluster &lhs, const reco::CaloCluster &rhs)
    {
		return ( lhs.eta() < rhs.eta()) ;
    }
};

// builder class for ecal detail view
class FWECALDetailViewBuilder {
	
	public:

		// construct an ecal detail view builder
		// the arguments are the event, the eta and phi position to show,
		// the half width of the region (in indices, e.g. iEta) and
		// the default color for the hits.
		FWECALDetailViewBuilder(const FWEventItem *item, 
								float eta, float phi, int size = 50, 
								Color_t defaultColor = kMagenta)  
									: m_item(item), m_eta(eta), 
									m_phi(phi), m_size(size), 
									m_defaultColor(defaultColor) 
								{
									m_event = m_item->getEvent();
								}

		// draw the ecal information with the preset colors
		// (if any colors have been preset)
		TEveCaloLego* build();
	
		// set colors of some predefined detids
		void setColor(Color_t color, const std::vector<DetId> &detIds);
					  
		// show superclusters in different colors
		void showSuperClusters(Color_t color);

		// show a specific supercluster in a specific color
		void showSuperCluster(const reco::SuperCluster &cluster, Color_t color);
	
		// fill data
		void fillData(const EcalRecHitCollection *hits, 
						TEveCaloDataVec *data);
	
	
	private:

		const FWEventItem* m_item;		
		float m_eta;					// eta position view centred on
		float m_phi;					// phi position view centred on
		int m_size;						// view half width in number of crystals
		Color_t m_defaultColor;			// default color for crystals
		const fwlite::Event *m_event;	// the event

		// for keeping track of what det id goes in what slice
		std::map<DetId, int> m_detIdsToColor; 
	
		// for keeping track of the colors to use for each slice
		std::vector<Color_t> m_colors;

		// sorting function to sort super clusters by eta.
		static bool superClusterEtaLess(const reco::CaloCluster &lhs, const reco::CaloCluster &rhs)
		{
			return ( lhs.eta() < rhs.eta());
		}
	
};