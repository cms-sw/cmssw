
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/10/16 09:33:39 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTOccupancyCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TH2F.h"
#include "TMath.h"

#include <iostream>

using namespace std;
using namespace edm;


DTOccupancyCluster::DTOccupancyCluster(const DTOccupancyPoint& firstPoint,
				       const DTOccupancyPoint& secondPoint) : radius(0),
									      theMaxMean(-1.),
									      theMaxRMS(-1.),
									      theMeanSum(0.),
									      theRMSSum(0.) {
  if(!qualityCriterion(firstPoint,secondPoint)) {
    theValidity = false;
  } else {
    // compute the cluster quantities
    thePoints.push_back(firstPoint);
    thePoints.push_back(secondPoint);
    if(firstPoint.mean() > secondPoint.mean()) theMaxMean = firstPoint.mean();
    else theMaxMean = secondPoint.mean();

    if(firstPoint.rms() > secondPoint.rms()) theMaxRMS = firstPoint.rms();
    else theMaxRMS = secondPoint.rms();
    theMeanSum += firstPoint.mean();
    theRMSSum += firstPoint.rms();

    theMeanSum += secondPoint.mean();
    theRMSSum += secondPoint.rms();


    computeRadius();
  }
}



DTOccupancyCluster::DTOccupancyCluster(const DTOccupancyPoint& singlePoint) : radius(0),
									      theMaxMean(singlePoint.mean()),
									      theMaxRMS(singlePoint.rms()),
									      theMeanSum(singlePoint.mean()),
									      theRMSSum(singlePoint.rms()) {
  theValidity = true;

  // compute the cluster quantities
  thePoints.push_back(singlePoint);
}



DTOccupancyCluster::~DTOccupancyCluster(){}



 // Check if the cluster candidate satisfies the quality requirements
bool DTOccupancyCluster::isValid() const {
  return theValidity;
}



// Add a point to the cluster: returns false if the point does not satisfy the
// quality requirement
bool DTOccupancyCluster::addPoint(const DTOccupancyPoint& anotherPoint) {
  LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest|DTOccupancyCluster")
    << "   Add a point to the cluster: mean: " << anotherPoint.mean()
    << " rms: " << anotherPoint.rms() << endl;
  if(qualityCriterion(anotherPoint)) {
    LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest|DTOccupancyCluster") << "   point is valid" << endl;
    thePoints.push_back(anotherPoint);
    // Compute the new cluster size
    computeRadius();
    // compute the max mean and RMS
    if(anotherPoint.mean() > theMaxMean) {
      theMaxMean = anotherPoint.mean();
    }
    if(anotherPoint.rms() > theMaxRMS) {
      theMaxRMS = anotherPoint.rms();
    }
    theMeanSum += anotherPoint.mean();
    theRMSSum += anotherPoint.rms();
    return true;
  } 
  return false;
}



// Compute the distance of a single point from the cluster
// (minimum distance with respect to the cluster points)
double DTOccupancyCluster::distance(const DTOccupancyPoint& point) const {
  double dist = 99999999;
  // compute the minimum distance from a point
  for(vector<DTOccupancyPoint>::const_iterator pt = thePoints.begin();
      pt != thePoints.end(); ++pt) {
    double distance = point.distance(*pt);
    if(distance < dist) {
      dist = distance;
    }
  }
  return dist;
}



double  DTOccupancyCluster::averageMean() const {
  return theMeanSum/(double)thePoints.size();
}



double  DTOccupancyCluster::averageRMS() const {
  return theRMSSum/(double)thePoints.size();
}



double  DTOccupancyCluster::maxMean() const {
  return theMaxMean;
}


  
double  DTOccupancyCluster::maxRMS() const {
  return theMaxRMS;
}



TH2F * DTOccupancyCluster::getHisto(std::string histoName, int nBinsX, double minX, double maxX,
				    int nBinsY, double minY, double maxY, int fillColor) const {
  TH2F *histo = new TH2F(histoName.c_str(),histoName.c_str(),
			 nBinsX, minX, maxX, nBinsY, minY, maxY);
  histo->SetFillColor(fillColor);
  for(vector<DTOccupancyPoint>::const_iterator pt = thePoints.begin();
      pt != thePoints.end(); ++pt) {
    histo->Fill((*pt).mean(), (*pt).rms());
  }
  return histo;
}



bool DTOccupancyCluster::qualityCriterion(const DTOccupancyPoint& firstPoint,
					  const DTOccupancyPoint& secondPoint) {
  //   cout << "   Seed qualityCriterion results: " << endl;
  //   cout << "      delta mean: " << firstPoint.deltaMean(secondPoint) << endl;
  //   cout << "      delta rms: " << firstPoint.deltaRMS(secondPoint) << endl;
  //   cout << "      average rms: " << computeAverageRMS(firstPoint, secondPoint) << endl;
  //   cout << "      min rms: " << computeMinRMS(firstPoint, secondPoint) << endl;
  if(firstPoint.deltaMean(secondPoint) <  computeAverageRMS(firstPoint, secondPoint) &&
     firstPoint.deltaRMS(secondPoint) < computeMinRMS(firstPoint, secondPoint)) {
    theValidity = true;
    //     cout << "    seed is valid!" << endl;
    return true;
  }  
  //   cout << "    seed is not valid!" << endl;
  theValidity = false;
  return false;
}



  
bool  DTOccupancyCluster::qualityCriterion(const DTOccupancyPoint& anotherPoint) {
  //   double minDistance = distance(anotherPoint);

  double minrms = 0;
  if(anotherPoint.rms() < averageRMS()) minrms = anotherPoint.rms();
  else minrms = averageRMS();

  //   cout << "  delta mean: " <<   fabs(averageMean() - anotherPoint.mean()) << endl;
  //   cout << "  average RMS: " << fabs(averageRMS() + anotherPoint.rms())/2. << endl;
  //   cout << "  delta rms: " <<  fabs(averageRMS() - anotherPoint.rms()) << endl;
  //   cout << " min rms: " << minrms << endl;
  //   if(fabs(averageMean() - anotherPoint.mean()) < fabs(averageRMS() + anotherPoint.rms())/2. &&
  //      fabs(averageRMS() - anotherPoint.rms()) < minrms) {

  if(fabs(averageMean() - anotherPoint.mean()) < averageRMS() &&
     fabs(averageRMS() - anotherPoint.rms()) < 2*minrms/3.) {
    theValidity = true;
    //     cout << "    point is valid!" << endl;
    return true;
  }  
  //   cout << "    point is not valid!" << endl;
  theValidity = false;
  return false;

  //   if(thePoints.size() == 1) return false;
  //   // Get the minimum distance from the point
  //   double minDistance = distance(anotherPoint);
  //   cout << "     min distance from cluster is: " << minDistance << endl;
  //   cout << "     radius is: " << radius << endl;
  //   // compare the minimum distance with the radius;
  //   if(minDistance < 2*radius) return true;
  //   else return false;
}




void DTOccupancyCluster::computeRadius() {
  double radius_squared = 0;
  for(vector<DTOccupancyPoint>::const_iterator pt_i = thePoints.begin();
      pt_i != thePoints.end(); ++pt_i) {
    for(vector<DTOccupancyPoint>::const_iterator pt_j = thePoints.begin();
	pt_j != thePoints.end(); ++pt_j) {
      radius_squared += TMath::Power(pt_i->distance(*pt_j),2);
    }
  }
  radius_squared = radius_squared/(2*TMath::Power(thePoints.size()+1,2));
  radius = sqrt(radius_squared);
  //   cout << "    the new cluster radius is: " << radius << endl;
}



int DTOccupancyCluster::nPoints() const {
  return thePoints.size();
}


set<DTLayerId> DTOccupancyCluster::getLayerIDs() const {
  set<DTLayerId> ret;
  for(vector<DTOccupancyPoint>::const_iterator point = thePoints.begin();
      point != thePoints.end(); ++point) {
    ret.insert((*point).layerId());
  }
  return ret;
}


bool clusterIsLessThan(const DTOccupancyCluster& clusterOne, const DTOccupancyCluster& clusterTwo) {
  if(clusterTwo.nPoints() == 1 && clusterOne.nPoints() != 1) {
    return true;
  }
  if(clusterTwo.nPoints() != 1 && clusterOne.nPoints() == 1) {
    return false;
  }

  if(clusterOne.nPoints() > clusterTwo.nPoints()) {
    return true;
  } else if(clusterOne.nPoints() < clusterTwo.nPoints()) {
    return false;
  } else {
    if(fabs(clusterOne.averageRMS() - sqrt(clusterOne.averageMean())) <
       fabs(clusterTwo.averageRMS() - sqrt(clusterTwo.averageMean()))) {
      return true;
    }
  }
  return false;

}


