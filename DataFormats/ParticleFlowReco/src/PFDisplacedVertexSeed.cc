#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexSeed.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/Point3D.h"

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



  for (const auto & newElement : newElements)
    addElement(newElement);



}


void PFDisplacedVertexSeed::Dump( ostream& out ) const {
  if(! out ) return;

  out<<"\t--- DisplacedVertexSeed ---  "<<endl;
  out<<"\tnumber of elements: "<<elements_.size()<<endl;
  
  out<<"\t Seed Point x = " << seedPoint().x() 
     <<"\t Seed Point y = " << seedPoint().y()
     <<"\t Seed Point z = " << seedPoint().z() << endl;

  // Build element label (string) : elid from type, layer and occurence number
  // use stringstream instead of sprintf to concatenate string and integer into string
  for(const auto & element : elements_){

    math::XYZPoint Pi(element.get()->innerPosition());
    math::XYZPoint Po(element.get()->outerPosition());

    float innermost_radius = sqrt(Pi.x()*Pi.x() + Pi.y()*Pi.y() + Pi.z()*Pi.z());
    float outermost_radius = sqrt(Po.x()*Po.x() + Po.y()*Po.y() + Po.z()*Po.z());
    float innermost_rho = sqrt(Pi.x()*Pi.x() + Pi.y()*Pi.y());
    float outermost_rho = sqrt(Po.x()*Po.x() + Po.y()*Po.y());
    
    double pt = element->pt();


    out<<"ie = " << element.key() << " pt = " << pt
       <<" innermost hit radius = " << innermost_radius << " rho = " << innermost_rho
       <<" outermost hit radius = " << outermost_radius << " rho = " << outermost_rho
       <<endl;

    out<<"ie = " << element.key() << " pt = " << pt
      //       <<" inn hit pos x = " << Pi.x() << " y = " << Pi.y() << " z = " << Pi.z() 
       <<" out hit pos x = " << Po.x() << " y = " << Po.y() << " z = " << Po.z() 
       <<endl;

  }
   
  out<<endl;


}
  

