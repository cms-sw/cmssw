
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
#include "DataFormats/FWLite/interface/Event.h"

#include <map>
#include <vector>
#include <utility>

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
		// ... this uses the above method to set colors...
		void showSuperClusters(Color_t color);

		// fill data
		void fillData(const EcalRecHitCollection *hits, 
						TEveCaloDataVec *data);
	
	
	private:

		const FWEventItem* m_item;
		float m_eta;
		float m_phi;
		int m_size;
		Color_t m_defaultColor;
		const fwlite::Event *m_event;

		std::map<DetId, int> m_detIdsToColor; 
		std::vector<Color_t> m_colors;

		// stuff copied from JM.  Not sure what it all is.
		TCanvas* m_textCanvas;

};