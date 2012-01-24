#include "QCDAnalysis/ChargedHadronSpectra/interface/EnergyLossPlain.h"

#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
typedef Vector2DBase<float,GlobalTag> Global2DVector;

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

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

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <fstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// m_e c^2
#define Tmax 1.

using namespace std;

bool  EnergyLossPlain::isFirst = true;
float EnergyLossPlain::optimalWeight[61][61];

/*****************************************************************************/
EnergyLossPlain::EnergyLossPlain
  (const TrackerGeometry* theTracker_,
   double pixelToStripMultiplier_,
   double pixelToStripExponent_  ) : theTracker(theTracker_),
      pixelToStripMultiplier(pixelToStripMultiplier_),
      pixelToStripExponent  (pixelToStripExponent_  )
{
  // Load data
  if(isFirst == true)
  {
    loadOptimalWeights();
    isFirst = false;
  }
}

/*****************************************************************************/
EnergyLossPlain::~EnergyLossPlain()
{
}


/*****************************************************************************/
void EnergyLossPlain::loadOptimalWeights()
{
  edm::FileInPath
    fileInPath("QCDAnalysis/ChargedHadronSpectra/data/energyWeights.dat");
  ifstream inFile(fileInPath.fullPath().c_str());

  while(inFile.eof() == false)
  {
    int i; float w; int n;
    inFile >> i;
    inFile >> w;
    inFile >> n;

    EnergyLossPlain::optimalWeight[n][i] = w;
  }

  inFile.close();

  LogTrace("MinBiasTracking")
    << " [EnergyLossEstimator] optimal weights loaded";
}

/*****************************************************************************/
double EnergyLossPlain::average(std::vector<pair<double,double> >& values)
{
  int num = values.size();
  double sum[2] = {0.,0.};

  for(int i = 0; i < num; i++)
  {
    sum[1] += values[i].first;
    sum[0] += 1;
  }

  return sum[1] / sum[0];
}

/*****************************************************************************/
double EnergyLossPlain::logTruncate(std::vector<pair<double,double> >& values_)
{
  std::vector<double> values;
  for(std::vector<pair<double,double> >::iterator
      v = values_.begin(); v!= values_.end(); v++)
    values.push_back((*v).first);

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
double EnergyLossPlain::truncate(std::vector<pair<double,double> >& values_)
{
  std::vector<double> values;
  for(std::vector<pair<double,double> >::iterator 
      v = values_.begin(); v!= values_.end(); v++)
    values.push_back((*v).first);

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
double EnergyLossPlain::optimal(std::vector<pair<double,double> >& values_)
{
  std::vector<double> values;
  for(std::vector<pair<double,double> >::iterator
      v = values_.begin(); v!= values_.end(); v++)
    values.push_back((*v).first);

  int num = values.size();
  sort(values.begin(), values.end());

  // First guess
  double sum = 0.;
  for(int i = 0; i < num; i++)
  {
    double weight = optimalWeight[num][i];
    sum += weight * values[i];
  }

  // Correct every deposit with log(path)
  for(int i = 0; i < num; i++)
    values_[i].first -= 0.178*log(values_[i].second/0.03) * 0.38 * sum;

  // Sort again 
  values.clear();
  for(std::vector<pair<double,double> >::iterator
      v = values_.begin(); v!= values_.end(); v++)
    values.push_back((*v).first);
  sort(values.begin(), values.end()); 

  // Corrected estimate
  sum = 0.;
  for(int i = 0; i < num; i++)
  {
    double weight = optimalWeight[num][i];
    sum += weight * values[i];
  }

  return sum;
}

/*****************************************************************************/
double EnergyLossPlain::expected(double Delta1, double Delta2)
{
  return log(Delta2/Delta1) / (1/Delta1 - 1/Delta2);
//    return 1 + (Delta2*log(Delta1) - Delta1*log(Delta2)) / (Delta2 - Delta1);
}

/*****************************************************************************/
void EnergyLossPlain::process
  (LocalVector ldir, const SiPixelRecHit* recHit, std::vector<pair<double,double> >& values)
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

  // Collect adc
  double elec = recHit->cluster()->charge();
  double Delta = elec * MeVperElec;

  // Length
  double x = ldir.mag()/fabsf(ldir.z()) *
             pixelDet->surface().bounds().thickness();

  double pix = Delta/x;

  // MeV/cm, only if not low deposit
  if(pix > 1.5 * 0.795)
    values.push_back(std::pair<double,double>(pix, x)); 
}

/*****************************************************************************/
void EnergyLossPlain::process
  (LocalVector ldir, const SiStripRecHit2D* recHit, std::vector<pair<double,double> >& values)
{
  DetId id = recHit->geographicalId();
  const StripGeomDetUnit* stripDet =
    dynamic_cast<const StripGeomDetUnit*> (theTracker->idToDet(id)); 

  // Check if cluster does not touch the edge
  if(recHit->cluster()->firstStrip() == 0 ||
     int(recHit->cluster()->firstStrip() +
         recHit->cluster()->amplitudes().size()) ==
       stripDet->specificTopology().nstrips())
    return;

  double MeVperAdc = 313 * 3.61e-6;

  // Collect adc
  double Delta = 0;
//  for(std::vector<uint16_t>::const_iterator
  for(std::vector<uint8_t>::const_iterator
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
    }

    Delta += delta;
  }

  // Length
  double x = ldir.mag()/fabsf(ldir.z()) *
             stripDet->surface().bounds().thickness();

  double str = Delta/x;

  // MeV/cm, only if not low deposit
  if(str > 1.5)
    values.push_back(std::pair<double,double>(str, x));
}

/*****************************************************************************/
int EnergyLossPlain::estimate
  (const Trajectory* trajectory, std::vector<pair<int,double> >& arithmeticMean,
                                 std::vector<pair<int,double> >& truncatedMean)
{
  // (dE/dx, dx)
  std::vector<pair<double,double> > vpix,vstr; 

  for(std::vector<TrajectoryMeasurement>::const_iterator
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

        std::pair<double,double> v;
  
        if(stripMatchedRecHit != 0)
        {
          auto m = stripMatchedRecHit->monoHit();
          auto s = stripMatchedRecHit->stereoHit();
          process(ldir,&m, vstr);
          process(ldir,&s, vstr);
        } 

        if(stripProjectedRecHit != 0)
          process(ldir,&(stripProjectedRecHit->originalHit()), vstr);
  
        if(stripRecHit != 0)
          process(ldir,stripRecHit, vstr);
      }
    }
  }

  // Transform
  std::vector<pair<double,double> > vall;

  for(unsigned int i = 0; i < vpix.size(); i++)
  {
    float a = 0.795;
    float s = 10.1;

    std::pair<double,double> str(vpix[i]);

    double y = str.first / a / s;
    if(y > 0.9999) y =  0.9999;
    if(y <-0.9999) y = -0.9999;

    str.first = s * atanh(y);

    vall.push_back(str);
  }

  for(unsigned int i = 0; i < vstr.size(); i++) vall.push_back(vstr[i]);

  // Arithmetic mean
  arithmeticMean.push_back(std::pair<int,double>(vpix.size(), average(vpix)));
  arithmeticMean.push_back(std::pair<int,double>(vstr.size(), average(vstr)));
  arithmeticMean.push_back(std::pair<int,double>(vall.size(), average(vall)));

  // Wighted mean
  truncatedMean.push_back(std::pair<int,double>(vpix.size(), optimal(vpix)));
  truncatedMean.push_back(std::pair<int,double>(vstr.size(), optimal(vstr)));
  truncatedMean.push_back(std::pair<int,double>(vall.size(), optimal(vall)));

  return vall.size();
}

