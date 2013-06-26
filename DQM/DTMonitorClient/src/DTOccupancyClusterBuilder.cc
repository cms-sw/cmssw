
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/07/20 02:58:23 $
 *  $Revision: 1.4 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTOccupancyClusterBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TCanvas.h"
#include "TH2F.h"

#include <algorithm>
#include <sstream>
#include <iostream>

using namespace std;
using namespace edm;

DTOccupancyClusterBuilder::  DTOccupancyClusterBuilder() : maxMean(-1.),
							   maxRMS(-1.) {
}

DTOccupancyClusterBuilder::~DTOccupancyClusterBuilder(){}



void DTOccupancyClusterBuilder::addPoint(const DTOccupancyPoint& point) {
  // loop over points already stored
  for(set<DTOccupancyPoint>::const_iterator pt = thePoints.begin(); pt != thePoints.end(); ++pt) {
    theDistances[(*pt).distance(point)] = make_pair(*pt, point);
  }
  //   cout << "[DTOccupancyClusterBuilder] Add point with mean: " << point.mean()
  //        << " RMS: " << point.rms() << endl;
  thePoints.insert(point);
}


void DTOccupancyClusterBuilder::buildClusters() {
  //   cout << "[DTOccupancyClusterBuilder] buildClusters" << endl;
  while(buildNewCluster()) {
    //     cout << "New cluster builded" << endl;
    //     cout << "# of remaining points: " << thePoints.size() << endl;
    if(thePoints.size() <= 1) break;
  }
    
  // build single point clusters with the remaining points
  for(set<DTOccupancyPoint>::const_iterator pt = thePoints.begin(); pt != thePoints.end();
      ++pt) {
    DTOccupancyCluster clusterCandidate(*pt);
    theClusters.push_back(clusterCandidate);
    // store the range for building the histograms later
    if(clusterCandidate.maxMean() > maxMean) maxMean = clusterCandidate.maxMean();
    if(clusterCandidate.maxRMS() > maxRMS) maxRMS = clusterCandidate.maxRMS();
  }
  LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest|DTOccupancyClusterBuilder")
    << " # of valid clusters: " << theClusters.size() << endl;
  sortClusters();
  
}


void DTOccupancyClusterBuilder::drawClusters(std::string canvasName) {
  int nBinsX = 100;
  int nBinsY = 100;
  int colorMap[12] = {632, 600, 800, 400, 820, 416, 432, 880, 616, 860, 900, 920};

  //   cout << "Draw clusters: " << endl;
  //   cout << "    max mean: " << maxMean << " max rms: " << maxRMS << endl;

  TCanvas *canvas = new TCanvas(canvasName.c_str(),canvasName.c_str()); 
  canvas->cd();
  for(vector<DTOccupancyCluster>::const_iterator cluster = theClusters.begin();
      cluster != theClusters.end(); ++cluster) {
    stringstream stream;
    stream << canvasName << "_" << cluster-theClusters.begin();
    string histoName = stream.str();
    TH2F *histo = (*cluster).getHisto(histoName, nBinsX, 0, maxMean+3*maxMean/100.,
				      nBinsY, 0, maxRMS+3*maxRMS/100., colorMap[cluster-theClusters.begin()]);
    if(cluster == theClusters.begin()) 
      histo->Draw("box");
    else
      histo->Draw("box,same");
  }
}


std::pair<DTOccupancyPoint, DTOccupancyPoint> DTOccupancyClusterBuilder::getInitialPair() {
  return theDistances.begin()->second;
}

void DTOccupancyClusterBuilder::computePointToPointDistances() {
  theDistances.clear();
  for(set<DTOccupancyPoint>::const_iterator pt_i = thePoints.begin(); pt_i != thePoints.end();
      ++pt_i) { // i loopo
    for(set<DTOccupancyPoint>::const_iterator pt_j = thePoints.begin(); pt_j != thePoints.end();
	++pt_j) { // j loop
      if(*pt_i != *pt_j) {
	theDistances[pt_i->distance(*pt_j)] = make_pair(*pt_i, *pt_j);
      }
    }
  }
}



void DTOccupancyClusterBuilder::computeDistancesToCluster(const DTOccupancyCluster& cluster) {
  theDistancesFromTheCluster.clear();
  for(set<DTOccupancyPoint>::const_iterator pt = thePoints.begin(); pt != thePoints.end(); ++pt) {
    theDistancesFromTheCluster[cluster.distance(*pt)] = *pt;
  }
}


bool DTOccupancyClusterBuilder::buildNewCluster() {
  LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest|DTOccupancyClusterBuilder")
    << "--------- New Cluster Candidate ----------------------" << endl;
  pair<DTOccupancyPoint, DTOccupancyPoint> initialPair = getInitialPair();
  LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest|DTOccupancyClusterBuilder")
    << "   Initial Pair: " << endl
    << "           point1: mean " << initialPair.first.mean()
    << " rms " << initialPair.first.rms() << endl
    << "           point2: mean " << initialPair.second.mean()
    << " rms " << initialPair.second.rms() << endl;
  DTOccupancyCluster clusterCandidate(initialPair.first, initialPair.second);
  if(clusterCandidate.isValid()) {
    //     cout <<   " cluster candidate is valid" << endl;
    // remove already used pair
    thePoints.erase(initialPair.first);
    thePoints.erase(initialPair.second);
    if(thePoints.size() != 0) {
      computeDistancesToCluster(clusterCandidate);
      while(clusterCandidate.addPoint(theDistancesFromTheCluster.begin()->second)) {
	thePoints.erase(theDistancesFromTheCluster.begin()->second);
	if(thePoints.size() ==0) break;
	computeDistancesToCluster(clusterCandidate);
      }
    }
  } else {
    return false;
  }
  LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest|DTOccupancyClusterBuilder")
    << "   # of layers: " << clusterCandidate.nPoints()
    << " avrg. mean: " << clusterCandidate.averageMean() << " avrg. rms: " << clusterCandidate.averageRMS() << endl;
  theClusters.push_back(clusterCandidate);
  // store the range for building the histograms later
  if(clusterCandidate.maxMean() > maxMean) maxMean = clusterCandidate.maxMean();
  if(clusterCandidate.maxRMS() > maxRMS) maxRMS = clusterCandidate.maxRMS();
  computePointToPointDistances();
  return true;
}
  


void DTOccupancyClusterBuilder::sortClusters() {
  LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest|DTOccupancyClusterBuilder") << " sorting" << endl;
  sort(theClusters.begin(), theClusters.end(), clusterIsLessThan);
  // we save the detid of the clusters which are not the best one
  for(vector<DTOccupancyCluster>::const_iterator cluster = ++(theClusters.begin());
      cluster != theClusters.end(); ++cluster) { // loop over clusters skipping the first
    set<DTLayerId> clusterLayers = (*cluster).getLayerIDs();
    LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest|DTOccupancyClusterBuilder")
      << "     # layers in the cluster: " << clusterLayers.size() << endl;
    theProblematicLayers.insert(clusterLayers.begin(), clusterLayers.end());
  }
  LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest|DTOccupancyClusterBuilder")
    << " # of problematic layers: " << theProblematicLayers.size() << endl;
}


DTOccupancyCluster DTOccupancyClusterBuilder::getBestCluster() const {
  return theClusters.front();
}

bool DTOccupancyClusterBuilder::isProblematic(DTLayerId layerId) const {
  if(theProblematicLayers.find(layerId) != theProblematicLayers.end()) {
    return true;
  }
  return false;
}
