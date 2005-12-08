#ifndef RECOLOCALTRACKER_SISTRIPCLUSTERIZER_SISTRIPCLUSTERMATCH_H
#define RECOLOCALTRACKER_SISTRIPCLUSTERIZER_SISTRIPCLUSTERMATCH_H
#include "PhysicsTools/Candidate/interface/own_vector.h"
#include "DataFormats/TrackingRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

 class SiStripCluster;

class SiStripClusterMatch {
public:
  
  typedef  SiStripClusterCollection::ContainerIterator ClusterIterator;
  SiStripClusterMatch(){};
  template<class T>  
    own_vector<SiStripRecHit2DLocalPos> match(const SiStripCluster *cluster,ClusterIterator &begin, ClusterIterator &end, const DetId &detId, const T &topol,const GeomDetUnit* stripdet,const GeomDetUnit * partnerstripdet){
    own_vector<SiStripRecHit2DLocalPos> collector;
    LocalPoint position;
    const  LocalError dummy;
    // position of the initial and final point of the strip (RPHI cluster)
    MeasurementPoint RPHIpointini=MeasurementPoint(cluster->barycenter(),-0.5);
    MeasurementPoint RPHIpointend=MeasurementPoint(cluster->barycenter(),0.5);
    // position of the initial and final point of the strip in local coordinates (RPHI cluster)
    LocalPoint RPHIpositionini=topol.localPosition(RPHIpointini); 
    LocalPoint RPHIpositionend=topol.localPosition(RPHIpointend); 
    // position of the initial and final point of the strip in global coordinates (RPHI cluster)
    GlobalPoint rphiglobalpointini=(stripdet->surface()).toGlobal(RPHIpositionini);
    GlobalPoint rphiglobalpointend=(stripdet->surface()).toGlobal(RPHIpositionend);
    // position of the initial and final point of the strip in stereo local coordinates (RPHI cluster)
    const T& partnertopol=(T&)partnerstripdet->topology();
    LocalPoint RPHIpositiononStereoini=(partnerstripdet->surface()).toLocal(rphiglobalpointini);
    LocalPoint RPHIpositiononStereoend=(partnerstripdet->surface()).toLocal(rphiglobalpointend);
    SiStripClusterCollection::ContainerIterator seconditer;    
    for(seconditer=begin;seconditer!=end;++seconditer){
      // position of the initial and final point of the strip (STEREO cluster)
      MeasurementPoint STEREOpointini=MeasurementPoint(seconditer->barycenter(),-0.5);
      MeasurementPoint STEREOpointend=MeasurementPoint(seconditer->barycenter(),0.5);
      LocalPoint STEREOpositionini=partnertopol.localPosition(STEREOpointini); 
      LocalPoint STEREOpositionend=partnertopol.localPosition(STEREOpointend); 
      //(x2-x1)(y-y1)=(y2-y1)(x-x1)
      AlgebraicMatrix m(2,2); AlgebraicVector c(2), solution(2);
      m(1,1)=-(RPHIpositiononStereoend.y()-RPHIpositiononStereoini.y()); m(1,2)=(RPHIpositiononStereoend.x()-RPHIpositiononStereoini.x());
      m(2,1)=-(STEREOpositionend.y()-STEREOpositionini.y());m(2,2)=(STEREOpositionend.x()-STEREOpositionini.x());
      c(1)=m(1,2)*RPHIpositiononStereoini.y()+m(1,1)*RPHIpositiononStereoini.x();
      c(2)=m(2,2)*STEREOpositionini.y()+m(2,1)*STEREOpositionini.x();
      solution=solve(m,c);
      if(solution(2)>-(partnertopol.stripLength()/2)&&solution(2)<partnertopol.stripLength()/2){//(to be modified)
	//then we can add it to the Rechit collection 
	position=LocalPoint(solution(1),solution(2));
	collector.push_back(new SiStripRecHit2DLocalPos(position, dummy, stripdet,detId,cluster));
      }
    }
    return collector;
  };
};
#endif
