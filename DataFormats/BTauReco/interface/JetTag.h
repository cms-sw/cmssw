#ifndef DataFormats_BTauReco_JetTag_h
#define DataFormats_BTauReco_JetTag_h
// \class JetTag
// 
// \short base class for persistent tagging result 
// JetTag is a simple class with a reference to a jet, it's extended tagging niformations, and a tagging discriminant
// 
//
// \author Marcel Vos, Andrea Rizzi, Andrea Bocci based on ORCA version by Christian Weiser, Andrea Rizzi
// \version first version on January 12, 2006


#include "DataFormats/JetReco/interface/JetFloatAssociation.h"

namespace reco {

typedef JetFloatAssociation::value_type JetTag;
typedef JetFloatAssociation::Container JetTagCollection;

}

#endif // DataFormats_BTauReco_JetTag_h
