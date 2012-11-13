//=========================================================================
// AbsReader.hh
//
// Template implementation of a factory pattern for reading objects from
// C++ streams. Derive a reader for your concrete inheritance hierarchy
// from the "DefaultReader" template (as illustrated by the examples
// provided with the package). Wrap your reader using the "StaticReader"
// template when the actual reading is performed to ensure that the reader
// factory is unique.
//
// The classes which can be read must implement the static "read" function
// which creates objects on the heap. The signature of this function is
//
// static ClassName* read(const ClassId& id, std::istream& is);
//
// In this function, the class id argument can be used to implement object
// versioning. The "read" method must always succeed. If it is unable to
// build the object, it must throw an exception which inherits from
// std::exception.
//
// I. Volobouev
// September 2010
//=========================================================================

#ifndef GENERS_ABSREADER_HH_
#define GENERS_ABSREADER_HH_

#include <map>
#include <string>
#include <sstream>
#include "Alignment/Geners/interface/IOException.hh"

#include "Alignment/Geners/interface/ClassId.hh"

namespace gs {
    template<class Base>
    struct AbsReader
    {
        virtual ~AbsReader() {}

        virtual Base* read(const ClassId& id, std::istream& in) const = 0;
    };

    template<class Base, class Derived>
    struct ConcreteReader : public AbsReader<Base>
    {
        virtual ~ConcreteReader() {}

        inline Derived* read(const ClassId& id, std::istream& in) const
        {
            // Assume that Derived::read(id, in) returns a new object
            // of type "Derived" allocated on the heap
            return Derived::read(id, in);
        }
    };

    template<class Base>
    class DefaultReader : public std::map<std::string, AbsReader<Base>*>
    {
    public:
        typedef Base value_type;

        inline DefaultReader() : std::map<std::string, AbsReader<Base>*>() {}

        virtual ~DefaultReader()
        {
            for (typename std::map<std::string, AbsReader<Base>*>::
                     iterator it = this->begin(); it != this->end(); ++it)
                delete it->second;
        }

        inline Base* read(const ClassId& id, std::istream& in) const
        {
            typename std::map<std::string, AbsReader<Base>*>::
                const_iterator it = this->find(id.name());
            if (it == this->end()) 
            {
                std::ostringstream os;
                os << "In gs::DefaultReader::read: class \""
                   << id.name() << "\" is not mapped to a concrete reader";
                throw gs::IOInvalidArgument(os.str());
            }
            return it->second->read(id, in);
        }

    private:
        DefaultReader(const DefaultReader&);
        DefaultReader& operator=(const DefaultReader&);
    };

    // A trivial implementation of the Meyers singleton for use with reader
    // factories. Naturally, this assumes that all factories are independent
    // from each other (otherwise we are getting into trouble with undefined
    // singleton destruction order). Also, this particular code is not
    // thread-safe (but should become thread-safe in C++11 if I understand
    // static local initialization guarantees correctly).
    //
    // Assume that "Reader" is derived from "DefaultReader" and that it
    // publishes its base class as "Base".
    //
    template <class Reader>
    class StaticReader
    {
    public:
        typedef typename Reader::Base::value_type InheritanceBase;

        static const Reader& instance()
        {
            static Reader obj;
            return obj;
        }

        template <class Derived>
        static void registerClass()
        {
            Reader& rd = const_cast<Reader&>(instance());
            const ClassId& id(ClassId::makeId<Derived>());
            delete rd[id.name()];
            rd[id.name()] = new ConcreteReader<InheritanceBase,Derived>();
        }

    private:
        // Disable the constructor
        StaticReader();
    };
}

#endif // GENERS_ABSREADER_HH_

