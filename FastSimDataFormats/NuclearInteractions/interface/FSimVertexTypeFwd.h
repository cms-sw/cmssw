#ifndef FastSimDataFormats_NuclearInteractions_FSimVertexTypeFwd_h
#define FastSimDataFormats_NuclearInteractions_FSimVertexTypeFwd_h 

#include <vector>
#include "DataFormats/Common/interface/Ref.h"

class FSimVertexType;

/// collection of FSimVertexType objects
typedef std::vector<FSimVertexType> FSimVertexTypeCollection;  
  
  
/// persistent reference to a FSimVertexType objects
typedef edm::Ref<FSimVertexTypeCollection> FSimVertexTypeRef;

/// handle to a FSimVertexType collection
typedef edm::Handle<FSimVertexTypeCollection> FSimVertexTypeHandle;


#endif
