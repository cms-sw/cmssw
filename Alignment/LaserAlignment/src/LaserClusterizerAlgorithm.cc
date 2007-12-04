/** \file LaserClusterizerAlgorithm.cc
 *  
 *
 *  $Date: 2007/03/18 19:00:20 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/LaserClusterizerAlgorithm.h"
#include "Alignment/LaserAlignment/interface/LaserBeamClusterizer.h" 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

LaserClusterizerAlgorithm::LaserClusterizerAlgorithm(const edm::ParameterSet & theConf) :
  theConfiguration(theConf), 
  theClusterMode(theConf.getParameter<std::string>("ClusterMode")),
  theClusterWidth(theConf.getParameter<double>("ClusterWidth")),
  theValidClusterizer(false)
{
  if (theClusterMode == "LaserBeamClusterizer")
    {
      theBeamClusterizer = new LaserBeamClusterizer();
      theValidClusterizer = true;
    }
  else
    {
      throw cms::Exception("LaserClusterizerAlgorithm") << "<LaserClusterizerAlgorithm::LaserClusterizerAlgorithm(const edm::Parameterset&)>: No valid clusterizer selected, possible clusterizer: LaserBeamClusterizer";
      theValidClusterizer = false;
    }
}

LaserClusterizerAlgorithm::~LaserClusterizerAlgorithm() 
{
  if (theBeamClusterizer != 0) { delete theBeamClusterizer; }
}

void LaserClusterizerAlgorithm::run(const edm::DetSetVector<SiStripDigi>& input, const LASBeamProfileFitCollection* beamFit,
				    edm::DetSetVector<SiStripCluster>& output, const edm::ESHandle<TrackerGeometry>& theTrackerGeometry)
{
  if (theValidClusterizer)
    {
      int nDetUnits = 0;
      int nLocalStripRecHits = 0;

      // loop over all detset inside the input collection
      for (edm::DetSetVector<SiStripDigi>::const_iterator DSViter = input.begin(); DSViter != input.end(); DSViter++)
	{
	  unsigned int theDetId = DSViter->id;
	  ++nDetUnits;

	  // get the BeamProfileFit for this detunit
	  const LASBeamProfileFitCollection::Range beamFitRange = beamFit->get(theDetId);
	  LASBeamProfileFitCollection::ContainerIterator beamFitRangeIteratorBegin = beamFitRange.first;
	  LASBeamProfileFitCollection::ContainerIterator beamFitRangeIteratorEnd = beamFitRange.second;

	  if (theClusterMode == "LaserBeamClusterizer")
	    {
	      edm::DetSet<SiStripCluster> theStripCluster(DSViter->id);

	      theBeamClusterizer->clusterizeDetUnit(*DSViter, theStripCluster, 
						    beamFitRangeIteratorBegin, beamFitRangeIteratorEnd, theDetId, theClusterWidth);
	    

	      if (theStripCluster.data.size() > 0)
		{
		  // insert the DetSet<SiStripCluster> in the DetSetVector<SiStripCluster> only if there is at least a digi
		  output.insert(theStripCluster);
		  nLocalStripRecHits += theStripCluster.data.size();
		}
	    }
	}

      edm::LogInfo("LaserClusterizerAlgorithm") << "execution in mode " << theClusterMode << " generating "
					    << nLocalStripRecHits << " SiStripClusters in " << nDetUnits
					    << " DetUnits ";
    }
}
