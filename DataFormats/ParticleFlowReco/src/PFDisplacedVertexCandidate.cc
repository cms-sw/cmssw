#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexCandidate.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/Point3D.h"

using namespace std;
using namespace reco;


PFDisplacedVertexCandidate::PFDisplacedVertexCandidate(){}

void PFDisplacedVertexCandidate::addElement(const TrackBaseRef element) {
  elements_.push_back( element ); 
}


void PFDisplacedVertexCandidate::setLink(const unsigned i1, 
					 const unsigned i2, 
					 const float dist,
					 const GlobalPoint& dcaPoint,
					 const VertexLinkTest test){

  
  assert( test<LINKTEST_ALL );
  
  unsigned index = 0;
  bool ok =  matrix2vector(i1,i2, index);

  if(ok) {
    //ignore the  -1, -1 pair
    if ( dist > -0.5 ) {
      VertexLink & l = vertexLinkData_[index];
      l.distance_ = dist;
      l.dcaPoint_ = dcaPoint;
      l.test_ |= (1 << test);
    }     else  //delete if existing
      {
	VertexLinkData::iterator it = vertexLinkData_.find(index);
	if(it!=vertexLinkData_.end()) vertexLinkData_.erase(it);
      }

  } else {
    assert(0);
  }
  
}


void PFDisplacedVertexCandidate::associatedElements( const unsigned i, 
						     const VertexLinkData& vertexLinkData, 
						     multimap<float, unsigned>& sortedAssociates,
						     const VertexLinkTest test ) const {

  sortedAssociates.clear();
  
  // i is too large
  if( i > elements_.size() ) return;
  // assert(i>=0); // i >= 0, since i is unsigned
  for(unsigned ie=0; ie<elements_.size(); ie++) {
    
    // considered element itself
    if( ie == i ) {
      continue;
    }

    // Order the elements by increasing distance !

    unsigned index = 0;
    if( !matrix2vector(i, ie, index) ) continue;

    float c2=-1;
    VertexLinkData::const_iterator it =  vertexLinkData.find(index);
    if ( it!=vertexLinkData.end() && 
	 ( ( (1 << test ) & it->second.test_) !=0 || (test == LINKTEST_ALL) ) ) 
      c2= it->second.distance_;

    // not associated
    if( c2 < 0 ) { 
      continue;
    }

    sortedAssociates.insert( pair<float,unsigned>(c2, ie) );
  }
} 







// -------- Provide useful information -------- //


PFDisplacedVertexCandidate::DistMap PFDisplacedVertexCandidate::r2Map() const {

  DistMap r2Map;

  for (unsigned ie1 = 0; ie1<elements_.size(); ie1++)
    for (unsigned ie2 = ie1+1; ie2<elements_.size(); ie2++){

      GlobalPoint P = dcaPoint(ie1, ie2);
      if (P.x() > 1e9) continue;

      float r2 = P.x()*P.x()+P.y()*P.y()+P.z()*P.z();

      r2Map.insert(pair<float, pair<int,int> >(r2, pair <int, int>(ie1, ie2)));
    }

  return r2Map;

}



PFDisplacedVertexCandidate::DistVector PFDisplacedVertexCandidate::r2Vector() const {

  DistVector r2Vector;

  for (unsigned ie1 = 0; ie1<elements_.size(); ie1++)
    for (unsigned ie2 = ie1+1; ie2<elements_.size(); ie2++){

      GlobalPoint P = dcaPoint(ie1, ie2);
      if (P.x() > 1e9) continue;

      float r2 = P.x()*P.x()+P.y()*P.y()+P.z()*P.z();

      r2Vector.push_back(r2);
    }

  return r2Vector;

}


PFDisplacedVertexCandidate::DistVector PFDisplacedVertexCandidate::distVector() const {

  DistVector distVector;


  for (unsigned ie1 = 0; ie1<elements_.size(); ie1++)
    for (unsigned ie2 = ie1+1; ie2<elements_.size(); ie2++){

      float d = dist(ie1, ie2);
      if (d < -0.5) continue;

      distVector.push_back(d);

    }

  return distVector;

}

const GlobalPoint PFDisplacedVertexCandidate::dcaPoint( unsigned ie1, unsigned ie2) const {

  GlobalPoint dcaPoint(1e10,1e10,1e10);

  unsigned index = 0;
  if( !matrix2vector(ie1, ie2, index) ) return dcaPoint;
  VertexLinkData::const_iterator it =  vertexLinkData_.find(index);
  if( it!=vertexLinkData_.end() ) dcaPoint = it->second.dcaPoint_;

  return dcaPoint;

}


// -------- Internal tools -------- //

bool PFDisplacedVertexCandidate::testLink(unsigned ie1, unsigned ie2) const {
  float d = dist( ie1, ie2);
  if (d < -0.5) return false;
  return true;
}


const float PFDisplacedVertexCandidate::dist( unsigned ie1, unsigned ie2) const {

  float dist = -1;

  unsigned index = 0;
  if( !matrix2vector(ie1, ie2, index) ) return dist;
  VertexLinkData::const_iterator it =  vertexLinkData_.find(index);
  if( it!=vertexLinkData_.end() ) dist= it->second.distance_;

  return dist;

}









// -------- Storage of the information -------- //


unsigned PFDisplacedVertexCandidate::vertexLinkDataSize() const {
  unsigned n = elements_.size();
  
  // number of possible undirected links between n elements.
  // reflective links impossible.
 
  return n*(n-1)/2; 
}


bool PFDisplacedVertexCandidate::matrix2vector( unsigned iindex, 
						unsigned jindex, 
						unsigned& index ) const {

  unsigned size = elements_.size();
  if( iindex == jindex || 
      iindex >=  size ||
      jindex >=  size ) {
    return false;
  }
  
  if( iindex > jindex ) 
    swap( iindex, jindex);

  
  index = jindex-iindex-1;

  if(iindex>0) {
    index += iindex*size;
    unsigned missing = iindex*(iindex+1)/2;
    index -= missing;
  }
  
  return true;
}

void PFDisplacedVertexCandidate::Dump( ostream& out ) const {
  if(! out ) return;

  const vector < TrackBaseRef >& elements = elements_;
  out<<"\t--- DisplacedVertexCandidate ---  "<<endl;
  out<<"\tnumber of elements: "<<elements.size()<<endl;
  
  // Build element label (string) : elid from type, layer and occurence number
  // use stringstream instead of sprintf to concatenate string and integer into string
  for(unsigned ie=0; ie<elements.size(); ie++) {

    math::XYZPoint Pi(elements[ie].get()->innerPosition());
    math::XYZPoint Po(elements[ie].get()->outerPosition());

    float innermost_radius = sqrt(Pi.x()*Pi.x() + Pi.y()*Pi.y() + Pi.z()*Pi.z());
    float outermost_radius = sqrt(Po.x()*Po.x() + Po.y()*Po.y() + Po.z()*Po.z());
    float innermost_rho = sqrt(Pi.x()*Pi.x() + Pi.y()*Pi.y());
    float outermost_rho = sqrt(Po.x()*Po.x() + Po.y()*Po.y());
    
    double pt = elements[ie]->pt();


    out<<"ie = " << elements[ie].key() << " pt = " << pt
       <<" innermost hit radius = " << innermost_radius << " rho = " << innermost_rho
       <<" outermost hit radius = " << outermost_radius << " rho = " << outermost_rho
       <<endl;
  }
   
  out<<endl;


}
  
