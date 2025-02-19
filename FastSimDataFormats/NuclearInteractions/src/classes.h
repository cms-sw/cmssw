#include "FastSimDataFormats/NuclearInteractions/interface/NUEvent.h"

#include "FastSimDataFormats/NuclearInteractions/interface/FSimVertexType.h"
#include "FastSimDataFormats/NuclearInteractions/interface/FSimVertexTypeFwd.h"

#include "FastSimDataFormats/NuclearInteractions/interface/FSimDisplacedVertex.h"
#include "FastSimDataFormats/NuclearInteractions/interface/FSimDisplacedVertexFwd.h"


#include <vector>
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include <DataFormats/Common/interface/OwnVector.h>
#include <DataFormats/Common/interface/ClonePolicy.h>


namespace { 
  struct dictionary {
    
    FSimVertexType                                        dummy0;
    std::vector<FSimVertexType>                           dummy1;
    edm::Wrapper< std::vector<FSimVertexType> >           dummy2;
    edm::Ref< std::vector<FSimVertexType>, FSimVertexType, edm::refhelper::FindUsingAdvance< std::vector<FSimVertexType>, FSimVertexType> >  dummy3;




    FSimDisplacedVertex                                       dummy4;
    std::vector<FSimDisplacedVertex>                           dummy5;
    edm::Wrapper< std::vector<FSimDisplacedVertex> >           dummy6;
    edm::Ref< std::vector<FSimDisplacedVertex>, FSimDisplacedVertex, edm::refhelper::FindUsingAdvance< std::vector<FSimDisplacedVertex>, FSimDisplacedVertex> >  dummy7;

      
  };
}
