#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexSeed.h"

using namespace std;
using namespace reco;


PFDisplacedVertexSeed::PFDisplacedVertexSeed() :
  seedPoint_(GlobalPoint(0,0,0)),
  totalWeight_(0)
{}


void PFDisplacedVertexSeed::addElement(TrackBaseRef element) {
  elements_.insert( element ); 
}

    
void PFDisplacedVertexSeed::updateSeedPoint(const GlobalPoint& dcaPoint, TrackBaseRef r1, TrackBaseRef r2, double weight){
    
  
  if ( isEmpty() ) {
    seedPoint_ = dcaPoint;
    totalWeight_ = weight;
  }
  else {
    Basic3DVector<double>vertexSeedVector(seedPoint_);
    Basic3DVector<double>dcaVector(dcaPoint);

 
    dcaVector = (dcaVector*weight + vertexSeedVector*totalWeight_)/(totalWeight_+weight);
    GlobalPoint P(dcaVector.x(), dcaVector.y(), dcaVector.z());
    totalWeight_ += weight;
    seedPoint_ = P;

  }

  addElement(r1); 
  addElement(r2);

}


void PFDisplacedVertexSeed::mergeWith(const PFDisplacedVertexSeed& displacedVertex){

  
  double weight = displacedVertex.totalWeight();
  set<TrackBaseRef, Compare> newElements= displacedVertex.elements();
  GlobalPoint dcaPoint = displacedVertex.seedPoint();

  Basic3DVector<double>vertexSeedVector(seedPoint_);
  Basic3DVector<double>dcaVector(dcaPoint);

  dcaVector = (dcaVector*weight + vertexSeedVector*totalWeight_)/(totalWeight_+weight);
  GlobalPoint P(dcaVector.x(), dcaVector.y(), dcaVector.z());
  totalWeight_ += weight;
  seedPoint_ = P;



  for (  set<TrackBaseRef, Compare>::const_iterator il = newElements.begin(); il != newElements.end(); il++)
    addElement(*il);



}



ostream& operator<<(  ostream& out, 
		      const reco::PFDisplacedVertexSeed& a ) {

  return out;

}
