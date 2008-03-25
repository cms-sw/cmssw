#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

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



void PFBlock::associatedElements( unsigned i, 
                                  const std::vector<double>& linkData, 
                                  map<double, unsigned>& sortedAssociates,
                                  PFBlockElement::Type type ) 
  const {
  


  sortedAssociates.clear();
  
  // i is too large
  if( i > elements_.size() ) return;
  assert(i>=0);
  
  for(unsigned ie=0; ie<elements_.size(); ie++) {
    
    // considered element itself
    if( ie == i ) {
      continue;
    }
    // not the right type
    if(type !=  PFBlockElement::NONE && 
       elements_[ie].type() != type ) {
      continue;
    }

    double c2 = chi2(i, ie,  linkData );
    
    // not associated
    if( c2 < 0 ) { 
      continue;
    }
    sortedAssociates.insert( make_pair(c2, ie) );
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
  const edm::OwnVector< reco::PFBlockElement >& elements = block.elements();
  out<<"\t--- PFBlock ---  "<<endl;
  out<<"\tnumber of elements: "<<elements.size()<<endl;
  
  // Build element label (string) : elid from type, layer and occurence number
  // use stringstream instead of sprintf to concatenate string and integer into string
 
  vector <string> elid;
  string s;
  stringstream ss;
  int iel = 0;
  int iTK =0;
  int iPS1 = 0;
  int iPS2 = 0;
  int iEE = 0;
  int iEB = 0;
  int iHE = 0;
  int iHB = 0;
  int iHF = 0;
  int iMU = 0;

  // for each element in turn
  
  for(unsigned ie=0; ie<elements.size(); ie++) {
    
    PFBlockElement::Type type = elements[ie].type();
    switch(type){
    case PFBlockElement::TRACK:
      iTK++;
      ss << "TK" << iTK;
      break;
    case PFBlockElement::MUON:
      iMU++;
      ss << "MU" << iMU;
      break;
    default:{
      PFClusterRef clusterref = elements[ie].clusterRef();
      int layer = clusterref->layer();
      switch (layer){
      case PFLayer::PS1:
        iPS1++;
        ss << "PV" << iPS1;
        break;
      case PFLayer::PS2:
        iPS2++;
        ss << "PH" << iPS2;
        break;
      case PFLayer::ECAL_ENDCAP:
        iEE++;
        ss << "EE" << iEE;
        break;
      case PFLayer::ECAL_BARREL:
        iEB++;
        ss << "EB" << iEB;
        break;
      case PFLayer::HCAL_ENDCAP:
        iHE++;
        ss << "HE" << iHE;
        break;
      case PFLayer::HCAL_BARREL1:
        iHB++;
        ss << "HB" << iHB;
        break;
      case PFLayer::VFCAL:
        iHF++;
        ss << "HF" << iHF;
        break;
      default:
        iel++;   
        ss << "??" << iel;
        break;
      }
      break;
    }
    }
    s = ss.str();
    elid.push_back( s );
    // clear stringstream
    ss.str("");

    out<<"\t"<< s <<" "<<elements[ie] <<endl;
  }
  
  out<<endl;
  int width = 6;
  if( !block.linkData().empty() ) {
    out<<"\tlink data: "<<endl;
    out<<setiosflags(ios::right);
    out<<"\t" << setw(width) << " ";
    for(unsigned ie=0; ie<elid.size(); ie++) out <<setw(width)<< elid[ie];
    out<<endl;  
    out<<setiosflags(ios::fixed);
    out<<setprecision(1);      
  
    for(unsigned i=0; i<block.elements_.size(); i++) {
      out<<"\t";
      out <<setw(width) << elid[i];
      for(unsigned j=0; j<block.elements_.size(); j++) {
        double chi2 = block.chi2(i,j, block.linkData() );
        if (chi2 > -0.5) out<<setw(width)<<block.chi2(i,j, block.linkData() );
        else  out <<setw(width)<< " ";
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
