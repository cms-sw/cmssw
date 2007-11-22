#ifndef FastSimulation_ParticlePropagator_MagneticFieldMap_H
#define FastSimulation_ParticlePropagator_MagneticFieldMap_H

// Framework Headers
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

// Famos headers
#include "FastSimulation/TrackerSetup/interface/TrackerLayer.h"

#include <vector>
#include <string>

class MagneticField;
class TrackerInteractionGeometry;
class TH1;

class MagneticFieldMap {

public:

  const GlobalVector inTesla( const GlobalPoint& ) const;
  const GlobalVector inKGauss( const GlobalPoint& ) const;
  const GlobalVector inInverseGeV( const GlobalPoint& ) const;
  const GlobalVector inTesla(const TrackerLayer& aLayer, double coord, int success) const;
  double inTeslaZ(const GlobalPoint&) const;
  double inKGaussZ(const GlobalPoint&) const;
  double inInverseGeVZ(const GlobalPoint&) const;
  double inTeslaZ(const TrackerLayer& aLayer, double coord, int success) const;

  const MagneticField& magneticField() const {return *pMF_;}

  static MagneticFieldMap* instance(const MagneticField* pMF,
				    const TrackerInteractionGeometry* myGeo);

  static MagneticFieldMap* instance() ;

private:

  MagneticFieldMap(const MagneticField* pMF,
		   const TrackerInteractionGeometry* myGeo);

  void initialize();

  const std::vector<double>* theFieldEndcapHisto(unsigned layer) const
    { return &(fieldEndcapHistos[layer]); } 

  const std::vector<double>* theFieldBarrelHisto(unsigned layer) const
    { return &(fieldBarrelHistos[layer]); } 

  static MagneticFieldMap* myself;
  const MagneticField* pMF_;
  const TrackerInteractionGeometry* geometry_;
  int bins;
  std::vector<std::vector<double> > fieldBarrelHistos;
  std::vector<std::vector<double> > fieldEndcapHistos;
  std::vector<double> fieldBarrelBinWidth;
  std::vector<double> fieldBarrelZMin;
  std::vector<double> fieldEndcapBinWidth;
  std::vector<double> fieldEndcapRMin;

};

#endif // FastSimulation_ParticlePropagator_MagneticFieldMap_H
