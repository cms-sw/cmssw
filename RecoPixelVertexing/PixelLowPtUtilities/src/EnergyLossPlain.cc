#include "RecoPixelVertexing/PixelLowPtUtilities/interface/EnergyLossPlain.h"

#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
typedef Vector2DBase<float,GlobalTag> Global2DVector;

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShape.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterData.h"

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"

// m_e c^2
#define Tmax 1.

using namespace std;

/*****************************************************************************/
EnergyLossPlain::EnergyLossPlain
  (const TrackerGeometry* theTracker_,
   double pixelToStripMultiplier_,
   double pixelToStripExponent_  ) : theTracker(theTracker_),
      pixelToStripMultiplier(pixelToStripMultiplier_),
      pixelToStripExponent  (pixelToStripExponent_  )
{
}

/*****************************************************************************/
EnergyLossPlain::~EnergyLossPlain()
{
}

/*****************************************************************************/
double EnergyLossPlain::average(vector<double>& values)
{
  int num = values.size();
  double sum[2] = {0.,0.};

  for(int i = 0; i < num; i++)
  {
    sum[1] += values[i];
    sum[0] += 1;
  }

  return sum[1] / sum[0];
}

/*****************************************************************************/
double EnergyLossPlain::logTruncate(vector<double>& values)
{
  sort(values.begin(), values.end());

  int num = values.size();
  double sum[2] = {0.,1.};

  for(int i = 0; i < (num+1)/2; i++)
  { 
    double weight = 1.;

    if(num%2 == 1 && i == (num-1)/2) weight = 1./2;
  
    sum[1] *= pow(values[i], weight); 
    sum[0] += weight;

  }
  
  return pow(sum[1], 1./sum[0]);
} 

/*****************************************************************************/
double EnergyLossPlain::truncate(vector<double>& values)
{
  sort(values.begin(), values.end());

  int num = values.size();
  double sum[2] = {0.,0.};

  for(int i = 0; i < (num+1)/2; i++)
  {
    double weight = 1.;

    if(num%2 == 1 && i == (num-1)/2) weight = 1./2;

    sum[1] += weight * values[i];
    sum[0] += weight;
  }

  return sum[1] / sum[0];
}

/*****************************************************************************/
double EnergyLossPlain::expected(double Delta1, double Delta2)
{
// !!
  return log(Delta2/Delta1) / (1/Delta1 - 1/Delta2);
//    return 1 + (Delta2*log(Delta1) - Delta1*log(Delta2)) / (Delta2 - Delta1);
}

/*****************************************************************************/
void EnergyLossPlain::process
  (LocalVector ldir, const SiPixelRecHit* recHit, vector<double>& values)
{
  DetId id = recHit->geographicalId();
  const PixelGeomDetUnit* pixelDet =
    dynamic_cast<const PixelGeomDetUnit*> (theTracker->idToDet(id));

  // Check if cluster does not touch the edge
  if(recHit->cluster()->minPixelRow() == 0 ||
     recHit->cluster()->maxPixelRow() ==
       pixelDet->specificTopology().nrows() - 1 ||
     recHit->cluster()->minPixelCol() == 0 ||
     recHit->cluster()->maxPixelCol() == 
       pixelDet->specificTopology().ncolumns() - 1)
    return;

  double MeVperElec = 3.61e-6;
  double elecperVcal = 65.;
  double gain     = 2.8; 
  double pedestal = 78.96; // 28.2 * gain

  double p0=0.00382; double p1=0.886; double p2=112.7; double p3=113.0;

  double MeVperVcal = MeVperElec * elecperVcal; 

  // Collect adc
  double Delta = 0;
  for(vector<SiPixelCluster::Pixel>::const_iterator
        pixel = (recHit->cluster()->pixels()).begin();
        pixel!= (recHit->cluster()->pixels()).end(); pixel++)
  {
    double elec = pixel->adc;
    double vcal = elec/elecperVcal;

    int adc  = int((vcal + pedestal)/gain + 0.5);

    double Delta1 = (atanh((adc - p3)/p2) + p1)/p0 * MeVperVcal;
    double Delta2;
    if(adc == 225) Delta2 = Tmax;
              else Delta2 = (atanh(((adc+1) - p3)/p2) + p1)/p0 * MeVperVcal;

    Delta += expected(Delta1, Delta2);
  }

  // Length
  double x = ldir.mag()/fabsf(ldir.z()) *
             pixelDet->surface().bounds().thickness();

  // Extra correction for lost adc
  values.push_back(pixelToStripMultiplier * Delta/x); 
}

/*****************************************************************************/
void EnergyLossPlain::process
  (LocalVector ldir, const SiStripRecHit2D* recHit, vector<double>& values)
{
  DetId id = recHit->geographicalId();
  const StripGeomDetUnit* stripDet =
    dynamic_cast<const StripGeomDetUnit*> (theTracker->idToDet(id)); 

  // Check if cluster does not touch the edge
  if(recHit->cluster()->firstStrip() == 0 ||
     recHit->cluster()->firstStrip() +
     recHit->cluster()->amplitudes().size() ==
       stripDet->specificTopology().nstrips())
    return;

  double MeVperAdc = 313 * 3.61e-6;

  // Collect adc
  double Delta = 0;
  for(vector<uint16_t>::const_iterator
    i = (recHit->cluster()->amplitudes()).begin();
    i!= (recHit->cluster()->amplitudes()).end(); i++)
  {
    // MeV
    double delta = (*i + 0.5) * MeVperAdc;

    if(*i >= 254)
    {
      double delta1, delta2; 
      if(*i == 254) { delta1 = 254 * MeVperAdc; delta2 = 512 * MeVperAdc; }
               else { delta1 = 512 * MeVperAdc; delta2 = Tmax; }

      delta = expected(delta1,delta2);
//cerr << " str " << *i << " " << delta1 << " " << delta2 << " " << delta << endl;
    }
//else
//cerr << " str " << *i << " " << delta << endl;

    Delta += delta;
  }

  // Length
  double x = ldir.mag()/fabsf(ldir.z()) *
             stripDet->surface().bounds().thickness();

  // MeV/cm
  values.push_back(Delta/x);
}

/*****************************************************************************/
int EnergyLossPlain::estimate
  (const Trajectory* trajectory, vector<pair<int,double> >& arithmeticMean,
                                 vector<pair<int,double> >& truncatedMean)
{
  vector<double> vpix,vstr;

  for(vector<TrajectoryMeasurement>::const_iterator
        meas = trajectory->measurements().begin();
        meas!= trajectory->measurements().end(); meas++)
  {
    const TrackingRecHit* recHit = meas->recHit()->hit();
    DetId id = recHit->geographicalId();

    if(recHit->isValid())
    {
      LocalVector ldir = meas->updatedState().localDirection();
  
      if(theTracker->idToDet(id)->subDetector() ==
           GeomDetEnumerators::PixelBarrel ||
         theTracker->idToDet(id)->subDetector() ==
           GeomDetEnumerators::PixelEndcap)
      { 
        // Pixel
        const SiPixelRecHit* pixelRecHit =
          dynamic_cast<const SiPixelRecHit *>(recHit);
  
        if(pixelRecHit != 0)
          process(ldir,pixelRecHit , vpix);
      }
      else
      {
        // Strip
        const SiStripMatchedRecHit2D* stripMatchedRecHit =
          dynamic_cast<const SiStripMatchedRecHit2D *>(recHit);
        const ProjectedSiStripRecHit2D* stripProjectedRecHit =
          dynamic_cast<const ProjectedSiStripRecHit2D *>(recHit);
        const SiStripRecHit2D* stripRecHit =
          dynamic_cast<const SiStripRecHit2D *>(recHit);

        pair<double,double> v;
  
        if(stripMatchedRecHit != 0)
        {
          process(ldir,stripMatchedRecHit->monoHit()  , vstr);
          process(ldir,stripMatchedRecHit->stereoHit(), vstr);
        } 

        if(stripProjectedRecHit != 0)
          process(ldir,&(stripProjectedRecHit->originalHit()), vstr);
  
        if(stripRecHit != 0)
          process(ldir,stripRecHit, vstr);
      }
    }
  }

  vector<double> vall;
  for(int i = 0; i < vpix.size(); i++) vall.push_back(vpix[i]);
  for(int i = 0; i < vstr.size(); i++) vall.push_back(vstr[i]);

  // Arithmetic mean
  arithmeticMean.push_back(pair<int,double>(vpix.size(), average(vpix)));
  arithmeticMean.push_back(pair<int,double>(vstr.size(), average(vstr)));
  arithmeticMean.push_back(pair<int,double>(vall.size(), average(vall)));

  // Truncated mean
  truncatedMean.push_back(pair<int,double>(vpix.size(), truncate(vpix)));
  truncatedMean.push_back(pair<int,double>(vstr.size(), truncate(vstr)));
  truncatedMean.push_back(pair<int,double>(vall.size(), truncate(vall)));

  return vall.size();
}

