#include "FastSimDataFormats/NuclearInteractions/interface/FSimDisplacedVertex.h"



FSimDisplacedVertex::FSimDisplacedVertex() :
  id_(-1),
  motherId_(-1),
  nCharged_(0),
  vertexType_(FSimVertexType::ANY),
  isRecoVertex_(false),
  recoVertexId_(-1)

{}
 
FSimDisplacedVertex::FSimDisplacedVertex(const SimVertex& vertex, 
					 unsigned id, int motherId,
					 unsigned nCharged,
					 const std::vector<int>& daughterIds,
					 const FSimVertexType::VertexType vertexType
					 ):
  vertex_(vertex),
  id_(id), 
  motherId_(motherId), 
  nCharged_(nCharged),
  daughterIds_(daughterIds),
  vertexType_(vertexType),
  isRecoVertex_(false),
  recoVertexId_(-1)
{}


FSimDisplacedVertex::FSimDisplacedVertex(const FSimDisplacedVertex& other) :
  vertex_(other.vertex_),
  id_(other.id_), 
  motherId_(other.motherId_), 
  nCharged_(other.nCharged_),
  daughterIds_(other.daughterIds_),
  vertexType_(other.vertexType_),
  isRecoVertex_(other.isRecoVertex()),
  recoVertexId_(other.recoVertexId())
{}


std::ostream& operator<<(std::ostream& out, 
                          const FSimDisplacedVertex& co) {  
  
  return out << "id = " <<  co.id() 
	     << " mother = " <<  co.motherId() 
	     << " N daugh. = " << co.nDaughters()
	     << " N charged " << co.nChargedDaughters()
	     << " Type = " << co.vertexType()
             << " recoVertexId = " << co.recoVertexId() << " "
	     << co.vertex();

}
