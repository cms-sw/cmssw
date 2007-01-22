#ifndef TrackClassFilter_H
#define TrackClassFilter_H

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "CondFormats/BTagObjects/interface/TrackProbabilityCategoryData.h"

  /**  filter to define the belonging of a track to a TrackClass
   */ 
class TrackClassFilter : public TrackProbabilityCategoryData
{
 public:

  /**  constructor from the range on p(Gev), eta, and number of
   *   hits and pixel hits
   */ 
 TrackClassFilter() {}

   TrackClassFilter(const TrackProbabilityCategoryData & data ) : TrackProbabilityCategoryData(data) {}
   
   TrackClassFilter(double  pmin,double  pmax, 
		 double  etamin,  double  etamax,
		  int  nhitmin,  int  nhitmax, 
		 int  npixelhitsmin, int  npixelhitsmax,
                  double cmin, double cmax)
  { 
  pMin=pmin;
  pMax=pmax;
  etaMin=etamin;
  etaMax=etamax; 
  nHitsMin=nhitmin;
  nHitsMax=nhitmax; 
  nPixelHitsMin=npixelhitsmin;
  nPixelHitsMax=npixelhitsmax;
  chiMin=cmin;
  chiMax=cmax; 
 } 

  void set(const double & pmin, const double & pmax, 
		 const double & etamin, const double & etamax,
		 int nhitmin, int nhitmax, 
		 int npixelhitsmin,int npixelhitsmax,
		 double cmin, double cmax)
  {
  pMin=pmin;
  pMax=pmax;
  etaMin=etamin;
  etaMax=etamax; 
  nHitsMin=nhitmin;
  nHitsMax=nhitmax; 
  nPixelHitsMin=npixelhitsmin;
  nPixelHitsMax=npixelhitsmax;
  chiMin=cmin;
  chiMax=cmax;
  }

 virtual ~TrackClassFilter(){}

  bool apply(const reco::Track &, const reco::Jet & , const reco::Vertex &) const;

  void dump() const;

};


#endif








