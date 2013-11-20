#include <string>
#include "TClassAttributeMap.h"

namespace Reflex {

    class PropertyList : public TClassAttributeMap {
    
      public:
        bool HasProperty (const std::string& key) const { 
            return ( std::string( GetPropertyAsString( key.c_str() ) ) != ""); 
        }

        std::string PropertyAsString(const std::string& key) const { 
            return std::string( GetPropertyAsString( key.c_str() ) ); 
        }
    }; // end class PropertyList

} // end namespace Reflex
