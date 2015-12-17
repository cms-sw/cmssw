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
#include "boost/foreach.hpp"

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
  std::auto_ptr<std::vector<unsigned> >       hits         ( new std::vector<unsigned>    (size,    0)   );
  std::auto_ptr<std::vector<float> >          strip        ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >          localtheta   ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >          localphi     ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >          localx       ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >          localy       ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >          localz       ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >          momentum     ( new std::vector<float>       (size, 0)      );
  std::auto_ptr<std::vector<float> >          energyloss   ( new std::vector<float>       (size, -1)     );
  std::auto_ptr<std::vector<float> >          time         ( new std::vector<float>       (size, -1)     );
  std::auto_ptr<std::vector<int> >            particle     ( new std::vector<int>         (size,-500)    );
  std::auto_ptr<std::vector<unsigned short> > process      ( new std::vector<unsigned short> (size,0)    );

  edm::ESHandle<TrackerGeometry> theTrackerGeometry;         iSetup.get<TrackerDigiGeometryRecord>().get( theTrackerGeometry );  
  edm::ESHandle<MagneticField> magfield;		     iSetup.get<IdealMagneticFieldRecord>().get(magfield);		      
  edm::ESHandle<SiStripLorentzAngle> SiStripLorentzAngle;    iSetup.get<SiStripLorentzAngleRcd>().get(runningmode_, SiStripLorentzAngle);      
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;  iEvent.getByLabel("siStripClusters", "", clusters);

	for(auto& simhit_token : simhits_tokens_) {
		edm::Handle< std::vector<PSimHit> > simhits; 
		iEvent.getByToken(simhit_token, simhits);
    BOOST_FOREACH( const PSimHit hit, *simhits ) {
      
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

  iEvent.put( hits,        Prefix + "hits"      );
  iEvent.put( strip,       Prefix + "strip"      );
  iEvent.put( localtheta,  Prefix + "localtheta" );
  iEvent.put( localphi,    Prefix + "localphi" );
  iEvent.put( localx,      Prefix + "localx" );
  iEvent.put( localy,      Prefix + "localy" );
  iEvent.put( localz,      Prefix + "localz" );
  iEvent.put( momentum,    Prefix + "momentum" );
  iEvent.put( energyloss,  Prefix + "energyloss" );
  iEvent.put( time,        Prefix + "time" );
  iEvent.put( particle,    Prefix + "particle" );
  iEvent.put( process,     Prefix + "process" );
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

