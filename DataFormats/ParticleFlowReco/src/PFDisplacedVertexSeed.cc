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
  if(std::find(elements_.begin(),elements_.end(), element) == elements_.end()) {
    elements_.emplace_back(std::move(element));
  }
}

void PFDisplacedVertexSeed::reserveElements(size_t newSize) {
  elements_.reserve(newSize);
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

  reserveElements(elements_.size()+2);
  addElement(r1); 
  addElement(r2);

}


void PFDisplacedVertexSeed::mergeWith(const PFDisplacedVertexSeed& displacedVertex){

  
  double weight = displacedVertex.totalWeight();
  const GlobalPoint& dcaPoint = displacedVertex.seedPoint();

  Basic3DVector<double>vertexSeedVector(seedPoint_);
  Basic3DVector<double>dcaVector(dcaPoint);

  dcaVector = (dcaVector*weight + vertexSeedVector*totalWeight_)/(totalWeight_+weight);
  GlobalPoint P(dcaVector.x(), dcaVector.y(), dcaVector.z());
  totalWeight_ += weight;
  seedPoint_ = P;

  reserveElements(elements_.size()+displacedVertex.elements().size());
  auto const oldSize=elements_.size();
  //avoid checking elements we just added from displacedVertex.elements()
  for(auto const& e: displacedVertex.elements()) {
    if(std::find(elements_.begin(), elements_.begin()+oldSize,e) == elements_.begin()+oldSize) {
      elements_.emplace_back(e);
    }
  }
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
  for(auto const& ie : elements_) {

    math::XYZPoint Pi(ie.get()->innerPosition());
    math::XYZPoint Po(ie.get()->outerPosition());

    float innermost_radius = sqrt(Pi.x()*Pi.x() + Pi.y()*Pi.y() + Pi.z()*Pi.z());
    float outermost_radius = sqrt(Po.x()*Po.x() + Po.y()*Po.y() + Po.z()*Po.z());
    float innermost_rho = sqrt(Pi.x()*Pi.x() + Pi.y()*Pi.y());
    float outermost_rho = sqrt(Po.x()*Po.x() + Po.y()*Po.y());
    
    double pt = ie->pt();


    out<<"ie = " << ie.key() << " pt = " << pt
       <<" innermost hit radius = " << innermost_radius << " rho = " << innermost_rho
       <<" outermost hit radius = " << outermost_radius << " rho = " << outermost_rho
       <<endl;

    out<<"ie = " << ie.key() << " pt = " << pt
      //       <<" inn hit pos x = " << Pi.x() << " y = " << Pi.y() << " z = " << Pi.z() 
       <<" out hit pos x = " << Po.x() << " y = " << Po.y() << " z = " << Po.z() 
       <<endl;

  }
   
  out<<endl;


}
  

