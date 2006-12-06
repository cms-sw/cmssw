#ifndef TrackClassFilter_H
#define TrackClassFilter_H

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/Jet.h"


  /**  filter to define the belonging of a track to a TrackClass
   */ 
class TrackClassFilter {

 public:

  /**  constructor from the range on p(Gev), eta, and number of
   *   hits and pixel hits
   */ 
   TrackClassFilter() {}
   
  TrackClassFilter(const double & pmin, const double & pmax, 
		 const double & etamin, const double & etamax,
		 const int & nhitmin, const int & nhitmax, 
		 const int & npixelhitsmin, const int & npixelhitsmax) :
  thePMin(pmin), thePMax(pmax),
  theEtaMin(etamin), theEtaMax(etamax), 
  nHitsMin(nhitmin), nHitsMax(nhitmax), 
  nPixelHitsMin(npixelhitsmin), nPixelHitsMax(npixelhitsmax) {};

  void set(const double & pmin, const double & pmax, 
		 const double & etamin, const double & etamax,
		 int nhitmin, int nhitmax, 
		 int npixelhitsmin,int npixelhitsmax,
		 double cmin, double cmax)
  {
  thePMin=pmin;
  thePMax=pmax;
  theEtaMin=etamin;
  theEtaMax=etamax; 
  nHitsMin=nhitmin;
  nHitsMax=nhitmax; 
  nPixelHitsMin=npixelhitsmin;
  nPixelHitsMax=npixelhitsmax;
  chimin=cmin;
  chimax=cmax;
  }

 virtual ~TrackClassFilter(){}

  bool apply(const reco::Track &, const reco::Jet & , const reco::Vertex &) const;

  void dump() const;

  double pMin() {return thePMin;}
  double pMax() {return thePMax;}
  double etaMin() {return theEtaMin;}
  double etaMax() {return theEtaMax;}
  int nHitMin() {return nHitsMin;}
  int nHitMax() {return nHitsMax;}
  int nPixelMin() {return nPixelHitsMin;}
  int nPixelMax() {return nPixelHitsMax;}
  double chiMin() {return chimin;}
  double chiMax() {return chimax;}

 private:

  double thePMin, thePMax, theEtaMin, theEtaMax;
  int nHitsMin, nHitsMax, nPixelHitsMin, nPixelHitsMax;
  double chimin,chimax;
};


#endif








