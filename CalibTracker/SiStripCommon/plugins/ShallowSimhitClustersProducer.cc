#include "CalibTracker/SiStripCommon/interface/ShallowSimhitClustersProducer.h"

#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

ShallowSimhitClustersProducer::ShallowSimhitClustersProducer(const edm::ParameterSet& iConfig)
  : clusters_token_( consumes< edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("Clusters"))),
    Prefix( iConfig.getParameter<std::string>("Prefix") ),
		runningmode_( iConfig.getParameter<std::string>("runningMode") )
{
	std::vector<edm::InputTag> simhits_tags = iConfig.getParameter<std::vector<edm::InputTag> >("InputTags");
	for(auto itag : simhits_tags) {
		simhits_tokens_.push_back(
			consumes< std::vector<PSimHit> >(itag)
			);
	}

  produces <std::vector<unsigned> >      ( Prefix + "hits"       );
  produces <std::vector<float> >         ( Prefix + "strip"      );
  produces <std::vector<float> >         ( Prefix + "localtheta" );
  produces <std::vector<float> >         ( Prefix + "localphi"   );
  produces <std::vector<float> >         ( Prefix + "localx"     );
  produces <std::vector<float> >         ( Prefix + "localy"     );
  produces <std::vector<float> >         ( Prefix + "localz"     );
  produces <std::vector<float> >         ( Prefix + "momentum"   );
  produces <std::vector<float> >         ( Prefix + "energyloss" );
  produces <std::vector<float> >         ( Prefix + "time"       );
  produces <std::vector<int> >            ( Prefix + "particle"   );
  produces <std::vector<unsigned short> > ( Prefix + "process"    );
}	      

void ShallowSimhitClustersProducer::
produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  shallow::CLUSTERMAP clustermap = shallow::make_cluster_map(iEvent,clusters_token_);

  int size = clustermap.size();
  auto          hits         = std::make_unique <std::vector<unsigned>>       (size, 0);
  auto          strip        = std::make_unique <std::vector<float>>          (size, -100);
  auto          localtheta   = std::make_unique <std::vector<float>>          (size, -100);
  auto          localphi     = std::make_unique <std::vector<float>>          (size, -100);
  auto          localx       = std::make_unique <std::vector<float>>          (size, -100);
  auto          localy       = std::make_unique <std::vector<float>>          (size, -100);
  auto          localz       = std::make_unique <std::vector<float>>          (size, -100);
  auto          momentum     = std::make_unique <std::vector<float>>          (size, 0);
  auto          energyloss   = std::make_unique <std::vector<float>>          (size, -1);
  auto          time         = std::make_unique <std::vector<float>>          (size, -1);
  auto          particle     = std::make_unique <std::vector<int>>            (size, -500);
  auto          process      = std::make_unique <std::vector<unsigned short>> (size, 0);

  edm::ESHandle<TrackerGeometry> theTrackerGeometry;         iSetup.get<TrackerDigiGeometryRecord>().get( theTrackerGeometry );  
  edm::ESHandle<MagneticField> magfield;		     iSetup.get<IdealMagneticFieldRecord>().get(magfield);		      
  edm::ESHandle<SiStripLorentzAngle> SiStripLorentzAngle;    iSetup.get<SiStripLorentzAngleRcd>().get(runningmode_, SiStripLorentzAngle);      
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;  iEvent.getByLabel("siStripClusters", "", clusters);

	for(auto& simhit_token : simhits_tokens_) {
		edm::Handle< std::vector<PSimHit> > simhits; 
		iEvent.getByToken(simhit_token, simhits);
    for(auto const& hit : *simhits ) {
      
      const uint32_t id = hit.detUnitId();
      const StripGeomDetUnit* theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTrackerGeometry->idToDet( id ) );
      const LocalVector drift = shallow::drift(theStripDet, *magfield, *SiStripLorentzAngle);

      const float driftedstrip_ = theStripDet->specificTopology().strip( hit.localPosition()+0.5*drift );
      const float     hitstrip_ = theStripDet->specificTopology().strip( hit.localPosition()           );

      shallow::CLUSTERMAP::const_iterator cluster = match_cluster( id, driftedstrip_, clustermap, *clusters); 
      if(cluster != clustermap.end()) {
				unsigned i = cluster->second;
				hits->at(i)+=1;
				if(hits->at(i) == 1) {
					strip->at(i) = hitstrip_;
					localtheta->at(i) = hit.thetaAtEntry();
					localphi->at(i) = hit.phiAtEntry();
					localx->at(i) = hit.localPosition().x();
					localy->at(i) = hit.localPosition().y();
					localz->at(i) = hit.localPosition().z();
					momentum->at(i) = hit.pabs();
					energyloss->at(i) = hit.energyLoss();
					time->at(i) = hit.timeOfFlight();
					particle->at(i) = hit.particleType();
					process->at(i) = hit.processType();
				}
      }    
    } 
  }

  iEvent.put(std::move(hits),        Prefix + "hits"      );
  iEvent.put(std::move(strip),       Prefix + "strip"      );
  iEvent.put(std::move(localtheta),  Prefix + "localtheta" );
  iEvent.put(std::move(localphi),    Prefix + "localphi" );
  iEvent.put(std::move(localx),      Prefix + "localx" );
  iEvent.put(std::move(localy),      Prefix + "localy" );
  iEvent.put(std::move(localz),      Prefix + "localz" );
  iEvent.put(std::move(momentum),    Prefix + "momentum" );
  iEvent.put(std::move(energyloss),  Prefix + "energyloss" );
  iEvent.put(std::move(time),        Prefix + "time" );
  iEvent.put(std::move(particle),    Prefix + "particle" );
  iEvent.put(std::move(process),     Prefix + "process" );
}

shallow::CLUSTERMAP::const_iterator ShallowSimhitClustersProducer::
match_cluster( const unsigned& id, const float& strip_, const shallow::CLUSTERMAP& clustermap, const edmNew::DetSetVector<SiStripCluster>& clusters) const {
  shallow::CLUSTERMAP::const_iterator cluster = clustermap.end();
  edmNew::DetSetVector<SiStripCluster>::const_iterator clustersDetSet = clusters.find(id);
  if( clustersDetSet != clusters.end() ) {
    edmNew::DetSet<SiStripCluster>::const_iterator left, right=clustersDetSet->begin();
    while( right != clustersDetSet->end() && strip_ > right->barycenter() ) 
      right++;
    left = right-1;
    if(right!=clustersDetSet->end() && right!=clustersDetSet->begin()) { 
      unsigned firstStrip = (right->barycenter()-strip_) < (strip_-left->barycenter()) ? right->firstStrip() : left->firstStrip();
      cluster = clustermap.find( std::make_pair( id, firstStrip));
    }
    else if(right != clustersDetSet->begin())
      cluster = clustermap.find( std::make_pair( id, left->firstStrip()));
    else 
      cluster = clustermap.find( std::make_pair( id, right->firstStrip()));
  }
  return cluster;
}

