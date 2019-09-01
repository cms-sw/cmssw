#include "FastSimDataFormats/NuclearInteractions/interface/FSimVertexType.h"

FSimVertexType::FSimVertexType() : vertexType_(ANY) {}

FSimVertexType::FSimVertexType(VertexType vertexType) : vertexType_(vertexType) {}

std::ostream& operator<<(std::ostream& out, const FSimVertexType& v) {
  out << "vertexType =  " << v.vertexType() << " ";

  switch (v.vertexType()) {
    case FSimVertexType::ANY:
      out << "ANY";
      break;
    case FSimVertexType::PRIMARY_VERTEX:
      out << "PRIMARY";
      break;
    case FSimVertexType::NUCL_VERTEX:
      out << "NUCLEAR";
      break;
    case FSimVertexType::PAIR_VERTEX:
      out << "PAIR";
      break;
    case FSimVertexType::BREM_VERTEX:
      out << "BREM";
      break;
    case FSimVertexType::DECAY_VERTEX:
      out << "DECAY";
      break;
    case FSimVertexType::END_VERTEX:
      out << "END";
      break;
    case FSimVertexType::PILEUP_VERTEX:
      out << "PILEUP";
      break;
    case FSimVertexType::BSM_VERTEX:
      out << "BSM";
      break;
    default:
      out << "CHECK YOUR VERTEX TYPE!!!!";
      break;
  }

  return out;
}
