#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

#include <iomanip>


using namespace std;
using namespace reco;


void PFBlock::addElement(  PFBlockElement* element ) {
  element->setIndex( elements_.size() );
  element->lock();
  elements_.push_back( element->clone() ); 

}


void PFBlock::bookLinkData() {

  unsigned dataSize =  linkDataSize();
  linkData_.reserve( dataSize );

  // initialize linkData_ to -1 (no link)
  linkData_.insert( linkData_.begin(), dataSize, -1);
}



void PFBlock::setLink(unsigned i1, unsigned i2, double chi2,
		      std::vector<double>& linkData ) const {
  
  assert( linkData.size() == linkDataSize() );
  

  unsigned index = 0;
  bool ok =  matrix2vector(i1,i2, index);
  if(ok)
    linkData[index] = chi2;
  else 
    assert(0);

}


void PFBlock::lock(unsigned i, std::vector<double>& linkData ) const {
  
  assert( linkData.size() == linkDataSize() );
  
  for(unsigned j=0; j<elements_.size(); j++) {
    
    if(i==j) continue;

    unsigned index = 0;
    bool ok =  matrix2vector(i,j, index);
    if(ok)
      linkData[index] = -1;
    else 
      assert(0);
  }
}



bool PFBlock::matrix2vector( unsigned iindex, unsigned jindex, 
			     unsigned& index ) const {

  unsigned size = elements_.size();
  if( iindex == jindex || 
      iindex >=  size ||
      jindex >=  size ) {
    return false;
  }
  
  if( iindex > jindex ) 
    std::swap( iindex, jindex);

  
  index = jindex-iindex-1;

  if(iindex>0) {
    index += iindex*size;
    unsigned missing = iindex*(iindex+1)/2;
    index -= missing;
  }
  
  return true;
}


double PFBlock::chi2( unsigned ie1, unsigned ie2,
		      const vector<double>& linkData ) const {
  
  
  double chi2 = -1;

  unsigned index = 0;
  if( matrix2vector(ie1, ie2, index) ) {
    assert( index<linkData.size() );
    chi2 = linkData[index]; 
  }
  return chi2;
}




ostream& reco::operator<<(  ostream& out, 
			    const reco::PFBlock& block ) {

  if(! out) return out;
  
  out<<"\t--- PFBlock ---  "<<endl;
  out<<"\tnumber of elements: "<<block.elements_.size()<<endl;
  
  for(PFBlock::IE ie = block.elements_.begin(); 
      ie != block.elements_.end(); ie++) {
    out<<"\t"<<*ie <<endl;
  }
  
  out<<endl;

  if( !block.linkData().empty() ) {
    out<<"\tlink data: "<<endl;
    
    out<<setprecision(1);
    out<<setiosflags(ios::right);
    out<<setiosflags(ios::fixed);
  
  
    for(unsigned i=0; i<block.elements_.size(); i++) {
      out<<"\t";
      for(unsigned j=0; j<block.elements_.size(); j++) {
	out<<setw(10)<<block.chi2(i,j, block.linkData() )<<" ";
      }
      out<<endl;
    }

    out<<setprecision(3);  
    out<<resetiosflags(ios::right|ios::fixed);
  }
  else {
    out<<"\tno links."<<endl;
  }
      
  
  return out;
}
 
 
unsigned PFBlock::linkDataSize() const {
  unsigned n = elements_.size();
  
  // number of possible undirected links between n elements.
  // reflective links impossible.
 
  return n*(n-1)/2; 
}
