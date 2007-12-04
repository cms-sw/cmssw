/** \file LaserBeamClusterizer.cc
 *  
 *
 *  $Date: 2007/03/18 19:00:20 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/LaserBeamClusterizer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


void LaserBeamClusterizer::clusterizeDetUnit(const edm::DetSet<SiStripDigi>& input, edm::DetSet<SiStripCluster>& output,
					     BeamFitIterator beginFit, BeamFitIterator endFit, unsigned int detId, double ClusterWidth)
{
  edm::DetSet<SiStripDigi>::const_iterator beginDigi = input.data.begin();
  edm::DetSet<SiStripDigi>::const_iterator endDigi = input.data.end();

  std::vector<SiStripDigi> theDigis;
  theDigis.reserve(10);

  output.data.reserve( (endDigi - beginDigi)/3 + 1 );

  double theMeanStrip = beginFit->mean();
  double theSigma = beginFit->sigma();

  if ( ((theMeanStrip > 0.0) && (theSigma > 0.0)) )
    {
      // loop over the digis
      for (edm::DetSet<SiStripDigi>::const_iterator iDigi = beginDigi; iDigi <= endDigi; iDigi++)
	{
	  if ( (iDigi->strip() > (theMeanStrip - ClusterWidth * theSigma) ) && (iDigi->strip() < (theMeanStrip + ClusterWidth * theSigma) ) )
	    {
	      theDigis.push_back((*iDigi));
	    }
	}
      
      if ( theDigis.size() > 0 ) // check if we have selected some digis
	{
	  output.data.push_back( SiStripCluster(detId, SiStripCluster::SiStripDigiRange(theDigis.begin(), theDigis.end())) );
	}
    }
  else
    {
      edm::LogInfo("LaserBeamClusterizer") << "no fit information available for this DetId " << detId << "; no clusters will be available!\n"; 
    }
}
