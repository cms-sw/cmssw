/// -*- C++ -*-
///
/// Package:    GeneratorInterface/GenFilters
/// \class      CosmicGenFilterHelix 
///
/// Description: 
///     Event filter for generated particles reaching a certain cylinder surface (around z-axis).
///
/// Implementation:
///     Assumes particles coming from outside of defined cylinder, but might work also otherwise.
///     Uses SteppingHelixPropagator and IdealMagneticFieldRecord.
///
///
/// Original Author:  Gero FLUCKE
///     Created:  Mon Mar  5 16:32:01 CET 2007
///


#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <TObjArray.h>

#include <vector>

class MagneticField;
class Propagator;

class CosmicGenFilterHelix : public edm::EDFilter {
 public:
  explicit CosmicGenFilterHelix(const edm::ParameterSet& config);
  virtual ~CosmicGenFilterHelix();

  virtual void beginJob();
  virtual bool filter(edm::Event &event, const edm::EventSetup &eventSetup);
  virtual void endJob();

 private:
  /// actually propagate to the defined cylinder
  bool propagateToCutCylinder(const GlobalPoint &vertStart, const GlobalVector &momStart,
			      int charge, const MagneticField *field,
                              const Propagator *propagator); //non-const: monitorEnd
  /// true if ID selected, return by value its charge
  bool charge(int id, int &charge) const;
  /// provide magnetic field from Event Setup
  const MagneticField* getMagneticField(const edm::EventSetup &setup) const;
  const Propagator* getPropagator(const edm::EventSetup &setup) const;
// ----------member data ---------------------------

  edm::EDGetTokenT<edm::HepMCProduct> theSrcToken;
  const std::vector<int>  theIds; /// requested Ids
  const std::vector<int>  theCharges; /// charges, parallel to theIds
  const std::string thePropagatorName; // tag to get propagator from ESetup
  const double      theMinP2; /// minimal momentum^2 after propagation to cylinder
  const double      theMinPt2; /// minimal transverse^2 momentum after propagation to cylinder

  Cylinder::ConstCylinderPointer theTargetCylinder; /// target cylinder, around z-axis
  Plane::ConstPlanePointer theTargetPlaneMin; /// plane closing cylinder at 'negative' side
  Plane::ConstPlanePointer theTargetPlaneMax; /// plane closing cylinder at 'positive' side

  unsigned int theNumTotal; /// for final statistics: all seen events
  unsigned int theNumPass; /// for final statistics: events with track reaching target

  // for monitoring:
  void createHistsStart(const char *dirName, TObjArray &hists);
  void createHistsEnd(const char *dirName, TObjArray &hists);
  void monitorStart(const GlobalPoint &vert, const GlobalVector &mom, int charge,TObjArray &hists);
  void monitorEnd(const GlobalPoint &endVert, const GlobalVector &endMom,
		  const GlobalPoint &vert, const GlobalVector &mom, double path, TObjArray &hists);
  bool equidistLogBins(double* bins, int nBins, double first, double last) const;
  const bool theDoMonitor;   /// whether or not to fill monitor hists (needs TFileService)
  TObjArray theHistsBefore; /// hists of properties from generator
  TObjArray theHistsAfter;  /// hists after successfull propagation

};
