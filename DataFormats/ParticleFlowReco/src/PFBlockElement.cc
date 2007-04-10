#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include <iostream>

using namespace reco;

int PFBlockElement::instanceCounter_ = 0;

int PFBlockElement::instanceCounter() {
  return instanceCounter_;
}

void PFBlockElement::Dump(std::ostream& out, 
			  const char* pad) const {
  if(!out) return;
  out<<pad<<"base element";
}

std::ostream& reco::operator<<( std::ostream& out, 
				const PFBlockElement& element ) {

  if(! out) return out;
  
  out<<"element "<<element.index()<<"- type "<<element.type_<<" ";
  element.Dump(out);
  
  return out;
}

