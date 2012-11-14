#ifndef NPSTAT_READER_FOR_STORABLEMULTIVARIATEFUNCTOR
#define NPSTAT_READER_FOR_STORABLEMULTIVARIATEFUNCTOR

/*!
// \file StorableMultivariateFunctorReader.h
//
// \brief The geners I/O reader factory for classes derived from
//        StorableMultivariateFunctor
//
// Author: I. Volobouev
//
// July 2012
*/

#include "Alignment/Geners/interface/AbsReader.hh"
#include "JetMETCorrections/InterpolationTables/interface/StorableMultivariateFunctor.h"

namespace npstat {
    /**
    // Note that this class does not have any public constructors.
    // All application usage is through the gs::StaticReader wrapper.
    */
    class StorableMultivariateFunctorReader : 
        public gs::DefaultReader<StorableMultivariateFunctor>
    {
        typedef gs::DefaultReader<StorableMultivariateFunctor> Base;
        friend class gs::StaticReader<StorableMultivariateFunctorReader>;
        StorableMultivariateFunctorReader();
    };
        
    /** The reader factory for descendants of StorableMultivariateFunctor */
    typedef gs::StaticReader<StorableMultivariateFunctorReader>
        StaticStorableMultivariateFunctorReader;
}

#endif // NPSTAT_READER_FOR_STORABLEMULTIVARIATEFUNCTOR

