#ifndef NPSTAT_STORABLEMULTIVARIATEFUNCTOR_HH_
#define NPSTAT_STORABLEMULTIVARIATEFUNCTOR_HH_

/*!
// \file StorableMultivariateFunctor.h
//
// \brief Interface definition for storable multivariate functors
//
// Author: I. Volobouev
//
// July 2012
*/

#include <string>
#include <iostream>
#include <typeinfo>

#include "Alignment/Geners/interface/ClassId.hh"
#include "JetMETCorrections/InterpolationTables/interface/AbsMultivariateFunctor.h"

namespace npstat {
    /** Base class for storable multivariate functors */
    class StorableMultivariateFunctor : public AbsMultivariateFunctor
    {
    public:
        inline StorableMultivariateFunctor() {}

        /** Functor description can be an arbitrary string */
        inline StorableMultivariateFunctor(const std::string& descr)
            : AbsMultivariateFunctor(), description_(descr) {}

        inline virtual ~StorableMultivariateFunctor() {}

        /** Retrieve the functor description */
        inline const std::string& description() const {return description_;}

        /** Change the functor description */
        inline void setDescription(const std::string& newDescription)
            {description_ = newDescription;}

        /**
        // This method will throw npstat::NpstatRuntimeError in case
        // functor description is different from the provided argument
        */
        void validateDescription(const std::string& description) const;

        //@{
        /**
        // Do not override comparison operators in the derived classes,
        // override the method "isEqual" instead.
        */
        inline bool operator==(const StorableMultivariateFunctor& r) const
            {return (typeid(*this) == typeid(r)) && this->isEqual(r);}
        inline bool operator!=(const StorableMultivariateFunctor& r) const
            {return !(*this == r);}
        //@}

        //@{
        /** Method related to "geners" I/O */
        virtual gs::ClassId classId() const = 0;
        virtual bool write(std::ostream& of) const = 0;
        //@}

        // I/O methods needed for reading
        static inline const char* classname()
            {return "npstat::StorableMultivariateFunctor";}
        static inline unsigned version() {return 1;}
        static StorableMultivariateFunctor* read(
            const gs::ClassId& id, std::istream& in);

    protected:
        /**
        // Method needed to compare objects for equality.
        // Must be overriden by derived classes. It is left
        // up to the derived classes to decide whether they
        // should compare description strings in order to
        // establish equality.
        */
        virtual bool isEqual(const StorableMultivariateFunctor&) const = 0;

    private:
        std::string description_;
    };
}

#endif // NPSTAT_STORABLEMULTIVARIATEFUNCTOR_HH_

