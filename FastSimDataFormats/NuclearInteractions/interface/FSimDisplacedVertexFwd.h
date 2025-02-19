#ifndef FastSimDataFormats_NuclearInteractions_FSimDisplacedVertexFwd_h
#define FastSimDataFormats_NuclearInteractions_FSimDisplacedVertexFwd_h 

#include <vector>
#include "DataFormats/Common/interface/Ref.h"


class FSimDisplacedVertex;

/// collection of FSimDisplacedVertex objects
typedef std::vector<FSimDisplacedVertex> FSimDisplacedVertexCollection;  
  
  
/// persistent reference to a FSimDisplacedVertex objects
typedef edm::Ref<FSimDisplacedVertexCollection> FSimDisplacedVertexRef;

/// handle to a FSimDisplacedVertex collection
typedef edm::Handle<FSimDisplacedVertexCollection> FSimDisplacedVertexHandle;


#endif
