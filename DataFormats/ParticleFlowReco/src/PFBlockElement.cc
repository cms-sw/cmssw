#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include <iostream>

using namespace reco;

int PFBlockElement::instanceCounter_ = 0;

int PFBlockElement::instanceCounter() {
  return instanceCounter_;
}

std::ostream& reco::operator<<( std::ostream& out, 
				const PFBlockElement& element ) {

  if(! out) return out;
  
  out<<"element "<<element.index()<<"- type "<<element.type_<<" ";
  element.Dump();
  
  return out;
}

