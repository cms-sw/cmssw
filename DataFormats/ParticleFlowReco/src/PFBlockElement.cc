#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"


using namespace reco;

// int PFBlockElement::instanceCounter_ = 0;

// int PFBlockElement::instanceCounter() {
//   return instanceCounter_;
// }

void PFBlockElement::Dump(std::ostream& out, 
			  const char* pad) const {
  if(!out) return;
  out<<pad<<"base element";
}

std::ostream& reco::operator<<( std::ostream& out, 
				const PFBlockElement& element ) {

  if(! out) return out;
  
  out<<"element "<<element.index()<<"- type "<<element.type_<<" ";

  try {
  switch(element.type_) {
  case PFBlockElement::TRACK:
    {
      const reco::PFBlockElementTrack& et =
	dynamic_cast<const reco::PFBlockElementTrack &>( element );
      et.Dump(out);
      break;
    }
  case PFBlockElement::ECAL:
  case PFBlockElement::HCAL:
    {
      const reco::PFBlockElementCluster& ec =
	dynamic_cast<const reco::PFBlockElementCluster &>( element );
      ec.Dump(out);
      break;
    }
  default:
    out<<" unknown type"<<std::endl;
    break;
  }
  }
  catch( std::exception& err) {
    out<<err.what()<<std::endl;
  }

  return out;
}

