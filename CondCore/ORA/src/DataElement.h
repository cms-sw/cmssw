#ifndef INCLUDE_ORA_DATAELEMENT_H
#define INCLUDE_ORA_DATAELEMENT_H

// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace ora {

    // class describing an elementary part of data to be stored 
  class DataElement {
    public:
    DataElement();
    DataElement( size_t declaringScopeOffset, size_t /*offsetFunction*/ offset );
    virtual ~DataElement();

    DataElement& addChild( size_t declaringScopeOffset, size_t /*offsetFunction*/ offset );

    size_t offset( const void* topLevelAddress ) const;
    void* address( const void* topLevelAddress ) const;
    size_t declaringScopeOffset() const;

    void clear();

    private:

    const DataElement* m_parent;
    std::vector<DataElement*> m_children;
    size_t m_declaringScopeOffset;
    /*Reflex::OffsetFunction m_offsetFunction*/ size_t m_offset;
  };
  
}

#endif

