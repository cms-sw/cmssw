//=========================================================================
// ClassId.hh
//
// Class identifier for I/O operations. Contains class name and version
// number. For templates, it should also contain version numbers of all
// template parameter classes.
//
// I. Volobouev
// September 2010
//=========================================================================

#ifndef GENERS_CLASSID_HH_
#define GENERS_CLASSID_HH_

#include <vector>
#include <string>
#include <iostream>

namespace gs {
    class ClassId
    {
    public:
        // Generic constructor using a prefix (which is usually
        // a class name) and a version number
        inline ClassId(const char* prefix, const unsigned version,
                       const bool isPtr=false)
            {initialize(prefix, version, isPtr);}

        // Generic constructor using a prefix (which is usually
        // a class name) and a version number
        inline ClassId(const std::string& prefix, const unsigned version,
                       const bool isPtr=false)
            {initialize(prefix.c_str(), version, isPtr);}

        // Use the following constructor in the "classId()" methods
        // of user-developed classes.
        //
        // Implementation note: it is possible to "specialize"
        // this constructor by delegating the actual job to the
        // "ClassIdSpecialization". Then we would be able to create
        // class ids for built-in and user types in a unified
        // way. This, however, would incur a performance hit
        // due to the necessity of making another ClassId and
        // copying the result into the internals of the new object.
        // This performance hit was deemed significant. If you
        // need a universal way to create class ids at some
        // point in your code, use the "itemId" method instead
        // (this may or may not incur a performance hit, depending
        // on what exactly the compiler does).
        template<class T>
        inline ClassId(const T&)
            {initialize(T::classname(), T::version(), false);}

        // Constructor from the class id represented by a string
        explicit ClassId(const std::string& id);

        // Use the following constructor in "read" functions.
        // Dummy argument "reading" is needed in order to generate
        // a distinct function signature (otherwise the templated
        // constructor can win).
        ClassId(std::istream& in, int reading);

        // Use the following pseudo-constructor in static "read"
        // methods in case a type check is desired. It has to be
        // made static because constructor without any arguments
        // can not be a template. Also, this is the way to construct
        // class ids for built-in types (there is no way to specialize
        // member methods).
        template<class T>
        static ClassId makeId();

        // "Universal" item id which also works for built-in types
        template<class T>
        static ClassId itemId(const T&);

        // Inspectors for the class name and version number
        inline const std::string& name() const {return name_;}
        inline unsigned version() const {return version_;}

        // Is this class a pointer for I/O purposes?
        inline bool isPointer() const {return isPtr_;}

        // The following function should return a unique class id string
        // which takes version number into account
        inline const std::string& id() const {return id_;}

        // The following checks if the class name corresponds to
        // a template (using the standard manner of class name forming)
        bool isTemplate() const;

        // The following function fills the vector with class template
        // parameters (if the class is not a template, the vector is
        // cleared). Due to the manner in which things are used in this
        // package, the result is actually a vector of (vectors of size 1).
        void templateParameters(std::vector<std::vector<ClassId> >* p) const;

        // Function to write this object out. Returns "true" on success.
        bool write(std::ostream& of) const;        

        // Comparison operators
        inline bool operator==(const ClassId& r) const
            {return id_ == r.id_;}
        inline bool operator!=(const ClassId& r) const
            {return !(*this == r);}
        inline bool operator<(const ClassId& r) const
            {return id_ < r.id_;}
        inline bool operator>(const ClassId& r) const
            {return id_ > r.id_;}

        // Modify the version number
        void setVersion(unsigned newVersion);

        // The following methods verify that the id/classname/version
        // of this object are equal to those of the argument and throw
        // "gs::IOInvalidArgument" exception if this is not so
        void ensureSameId(const ClassId& id) const;
        void ensureSameName(const ClassId& id) const;
        void ensureSameVersion(const ClassId& id) const;

        // The following method ensures that the version number of this
        // class id is within certain range [min, max], with both limits
        // allowed. "gs::IOInvalidArgument" exception is thrown if this
        // is not so.
        void ensureVersionInRange(unsigned min, unsigned max) const;

        // Sometimes one really needs to make a placeholder class id...
        // This is a dangerous function: the code using ClassId class
        // will normally assume that a ClassId object is always in a valid
        // state. Invalid class ids can be distinguished by their empty
        // class names (i.e., name().empty() returns "true").
        static ClassId invalidId();

    private:
        ClassId();

        void initialize(const char* prefix, unsigned version, bool isPtr);
        bool makeName();
        bool makeVersion();

        std::string name_;
        std::string id_;
        unsigned version_;
        bool isPtr_;

        // Return "true" if the prefix is valid
        static bool validatePrefix(const char* prefix);
    };


    // Simple class id compatibility checkers for use as policy classes
    // in templated code
    struct SameClassId
    {
        inline static bool compatible(const ClassId& id1, const ClassId& id2)
            {return id1.name() == id2.name();}
    };

    struct SameClassName
    {
        inline static bool compatible(const ClassId& id1, const ClassId& id2)
            {return id1 == id2;}
    };


    // Specialize the following template in order to be able to construct
    // ClassId for classes which do not implement static functions
    // "classname()" and "version()".
    template<class T>
    struct ClassIdSpecialization
    {
        inline static ClassId classId(const bool isPtr=false)
        {
            return ClassId(T::classname(), T::version(), isPtr);
        }
    };


    // Utility functions for naming template classes. The "nInclude"
    // argument tells us how many template parameters to include into
    // the generated template name. For example, use of
    //
    // template_class_name<X,Y>("myTemplate",1)
    //
    // will generate a class name which looks like myTemplate<X>, with
    // second template parameter omitted. While the result is equivalent
    // to invoking "template_class_name<X>("myTemplate")", having an
    // explicit limit is convenient for use from certain higher-level
    // functions. Note, however, that in the call with two template
    // parameters the class id specialization for Y must be available,
    // even though it is not used.
    //
    // This feature is sometimes helpful when certain template parameters
    // specify aspects of template behavior which have nothing to do
    // with object data contents and I/O. Typical example of such
    // a parameter is std::allocator of STL -- changing this to a custom
    // allocator will not affect serialized representation of an STL
    // container.
    //
    template<class T>
    std::string template_class_name(const char* templateName,
                                    unsigned nInclude=1);
    template<class T>
    std::string template_class_name(const std::string& templateName,
                                    unsigned nInclude=1);
    template<class T1, class T2>
    std::string template_class_name(const char* templateName,
                                    unsigned nInclude=2);
    template<class T1, class T2>
    std::string template_class_name(const std::string& templateName,
                                    unsigned nInclude=2);
    template<class T1, class T2, class T3>
    std::string template_class_name(const char* templateName,
                                    unsigned nInclude=3);
    template<class T1, class T2, class T3>
    std::string template_class_name(const std::string& templateName,
                                    unsigned nInclude=3);
    template<class T1, class T2, class T3, class T4>
    std::string template_class_name(const char* templateName,
                                    unsigned nInclude=4);
    template<class T1, class T2, class T3, class T4>
    std::string template_class_name(const std::string& templateName,
                                    unsigned nInclude=4);
    template<class T1, class T2, class T3, class T4, class T5>
    std::string template_class_name(const char* templateName,
                                    unsigned nInclude=5);
    template<class T1, class T2, class T3, class T4, class T5>
    std::string template_class_name(const std::string& templateName,
                                    unsigned nInclude=5);
    template<class T1, class T2, class T3, class T4, class T5, class T6>
    std::string template_class_name(const char* templateName,
                                    unsigned nInclude=6);
    template<class T1, class T2, class T3, class T4, class T5, class T6>
    std::string template_class_name(const std::string& templateName,
                                    unsigned nInclude=6);

    // Utility functions for naming stack-based containers such as std::array
    template<class T, std::size_t N>
    std::string stack_container_name(const char* templateName);

    template<class T, std::size_t N>
    std::string stack_container_name(const std::string& templateName);
}

#include <cassert>
#include <utility>
#include <vector>
#include <sstream>

#include "Alignment/Geners/interface/IOIsClassType.hh"
#include "Alignment/Geners/interface/IOIsAnyPtr.hh"


#ifdef GENERS_EMPTY_TYPE_QUALIFYER_
#undef GENERS_EMPTY_TYPE_QUALIFYER_
#endif

#define GENERS_EMPTY_TYPE_QUALIFYER_

// Specializations of "ClassIdSpecialization" for built-in classes.
// They all look the same, so we want to use a macro
#define gs_specialize_class_helper(qualifyer, name, version) /**/       \
    template<> struct ClassIdSpecialization<qualifyer name>             \
    {inline static ClassId classId(const bool isPtr=false)              \
    {return ClassId(#name, version, isPtr);}};

#define gs_specialize_class_id(name, version) /**/                           \
namespace gs {                                                               \
    gs_specialize_class_helper(GENERS_EMPTY_TYPE_QUALIFYER_, name, version)\
    gs_specialize_class_helper(const, name, version)                         \
    gs_specialize_class_helper(volatile, name, version)                      \
    gs_specialize_class_helper(const volatile, name, version)                \
}

// Specializations of "ClassIdSpecialization" for single-argument templates
#define gs_specialize_template_help_T(qualifyer, name, version, MAX) /**/   \
    template<class T> struct ClassIdSpecialization<qualifyer name <T> >     \
    {inline static ClassId classId(const bool isPtr=false)                  \
    {return ClassId(template_class_name<T>(#name,MAX), version, isPtr);}};

#define gs_specialize_template_id_T(name, version, MAX) /**/                         \
namespace gs {                                                                       \
    gs_specialize_template_help_T(GENERS_EMPTY_TYPE_QUALIFYER_, name, version, MAX)\
    gs_specialize_template_help_T(const, name, version, MAX)                         \
    gs_specialize_template_help_T(volatile, name, version, MAX)                      \
    gs_specialize_template_help_T(const volatile, name, version, MAX)                \
}

// Specializations of "ClassIdSpecialization" for two-argument templates
#define gs_specialize_template_help_TT(qualifyer, name, version, MAX) /**/  \
    template<class T,class U>                                               \
    struct ClassIdSpecialization<qualifyer name <T,U> >                     \
    {inline static ClassId classId(const bool isPtr=false)                  \
    {return ClassId(template_class_name<T,U>(#name,MAX), version, isPtr);}};

#define gs_specialize_template_id_TT(name, version, MAX) /**/                         \
namespace gs {                                                                        \
    gs_specialize_template_help_TT(GENERS_EMPTY_TYPE_QUALIFYER_, name, version, MAX)\
    gs_specialize_template_help_TT(const, name, version, MAX)                         \
    gs_specialize_template_help_TT(volatile, name, version, MAX)                      \
    gs_specialize_template_help_TT(const volatile, name, version, MAX)                \
}

// Specializations of "ClassIdSpecialization" for three-argument templates
#define gs_specialize_template_help_TTT(qualifyer, name, version, MAX) /**/ \
    template<class T,class U,class V>                                       \
    struct ClassIdSpecialization<qualifyer name <T,U,V> >                   \
    {inline static ClassId classId(const bool isPtr=false)                  \
    {return ClassId(template_class_name<T,U,V>(#name,MAX), version, isPtr);}};

#define gs_specialize_template_id_TTT(name, version, MAX) /**/                         \
namespace gs {                                                                         \
    gs_specialize_template_help_TTT(GENERS_EMPTY_TYPE_QUALIFYER_, name, version, MAX)\
    gs_specialize_template_help_TTT(const, name, version, MAX)                         \
    gs_specialize_template_help_TTT(volatile, name, version, MAX)                      \
    gs_specialize_template_help_TTT(const volatile, name, version, MAX)                \
}

// Specializations of "ClassIdSpecialization" for four-argument templates
#define gs_specialize_template_help_TTTT(qualifyer, name, version, MAX) /**/   \
    template<class T,class U,class V,class X>                                  \
    struct ClassIdSpecialization<qualifyer name <T,U,V,X> >                    \
    {inline static ClassId classId(const bool isPtr=false)                     \
    {return ClassId(template_class_name<T,U,V,X>(#name,MAX), version, isPtr);}};

#define gs_specialize_template_id_TTTT(name, version, MAX) /**/                         \
namespace gs {                                                                          \
    gs_specialize_template_help_TTTT(GENERS_EMPTY_TYPE_QUALIFYER_, name, version, MAX)\
    gs_specialize_template_help_TTTT(const, name, version, MAX)                         \
    gs_specialize_template_help_TTTT(volatile, name, version, MAX)                      \
    gs_specialize_template_help_TTTT(const volatile, name, version, MAX)                \
}

// Specializations of "ClassIdSpecialization" for five-argument templates
#define gs_specialize_template_hlp_TTTTT(qualifyer, name, version, MAX) /**/     \
    template<class T,class U,class V,class X,class Y>                            \
    struct ClassIdSpecialization<qualifyer name <T,U,V,X,Y> >                    \
    {inline static ClassId classId(const bool isPtr=false)                       \
    {return ClassId(template_class_name<T,U,V,X,Y>(#name,MAX), version, isPtr);}};

#define gs_specialize_template_id_TTTTT(name, version, MAX) /**/                        \
namespace gs {                                                                          \
    gs_specialize_template_hlp_TTTTT(GENERS_EMPTY_TYPE_QUALIFYER_, name, version, MAX)\
    gs_specialize_template_hlp_TTTTT(const, name, version, MAX)                         \
    gs_specialize_template_hlp_TTTTT(volatile, name, version, MAX)                      \
    gs_specialize_template_hlp_TTTTT(const volatile, name, version, MAX)                \
}

// Specializations of "ClassIdSpecialization" for six-argument templates
#define gs_specialize_template_h_TTTTTT(qualifyer, name, version, MAX) /**/       \
    template<class T,class U,class V,class X,class Y,class Z>                     \
    struct ClassIdSpecialization<qualifyer name <T,U,V,X,Y,Z> >                   \
    {inline static ClassId classId(const bool isPtr=false)                        \
    {return ClassId(template_class_name<T,U,V,X,Y,Z>(#name,MAX), version, isPtr);}};

#define gs_specialize_template_id_TTTTTT(name, version, MAX) /**/                      \
namespace gs {                                                                         \
    gs_specialize_template_h_TTTTTT(GENERS_EMPTY_TYPE_QUALIFYER_, name, version, MAX)\
    gs_specialize_template_h_TTTTTT(const, name, version, MAX)                         \
    gs_specialize_template_h_TTTTTT(volatile, name, version, MAX)                      \
    gs_specialize_template_h_TTTTTT(const volatile, name, version, MAX)                \
}

// Specializations of "ClassIdSpecialization" for two-argument templates
// which include an integer as a second argument (like std::array)
#define gs_specialize_template_help_TN(qualifyer, name, version, MAX) /**/  \
    template<class T,std::size_t N>                                         \
    struct ClassIdSpecialization<qualifyer name <T,N> >                     \
    {inline static ClassId classId(const bool isPtr=false)                  \
    {return ClassId(stack_container_name<T,N>(#name,MAX), version, isPtr);}};

#define gs_specialize_template_id_TN(name, version, MAX) /**/                         \
namespace gs {                                                                        \
    gs_specialize_template_help_TN(GENERS_EMPTY_TYPE_QUALIFYER_, name, version, MAX)\
    gs_specialize_template_help_TN(const, name, version, MAX)                         \
    gs_specialize_template_help_TN(volatile, name, version, MAX)                      \
    gs_specialize_template_help_TN(const volatile, name, version, MAX)                \
}

namespace gs {
    // "template_class_name" implementations
    template<class T>
    std::string template_class_name(const char* templateName,
                                    const unsigned nInclude)
    {
        assert(templateName);
        std::string name(templateName);
        if (nInclude)
        {
            name += '<';
            const ClassId& id(ClassIdSpecialization<T>::classId());
            name += id.id();
            name += '>';
        }
        return name;
    }

    template<class T>
    inline std::string template_class_name(const std::string& templateName,
                                           const unsigned nInclude)
    {
        return template_class_name<T>(templateName.c_str(), nInclude);
    }

    template<class T1, class T2>
    std::string template_class_name(const char* templateName,
                                    const unsigned nInclude)
    {
        assert(templateName);
        std::string name(templateName);
        if (nInclude)
        {
            name += '<';
            const ClassId& id1(ClassIdSpecialization<T1>::classId());
            name += id1.id();
            if (nInclude > 1)
            {
                name += ',';
                const ClassId& id2(ClassIdSpecialization<T2>::classId());
                name += id2.id();
            }
            name += '>';
        }
        return name;
    }

    template<class T1, class T2>
    inline std::string template_class_name(const std::string& templateName,
                                           const unsigned nInclude)
    {
        return template_class_name<T1,T2>(templateName.c_str(), nInclude);
    }

    template<class T1, class T2, class T3>
    std::string template_class_name(const char* templateName,
                                    const unsigned nInclude)
    {
        assert(templateName);
        std::string name(templateName);
        if (nInclude)
        {
            name += '<';
            const ClassId& id1(ClassIdSpecialization<T1>::classId());
            name += id1.id();
            if (nInclude > 1)
            {
                name += ',';
                const ClassId& id2(ClassIdSpecialization<T2>::classId());
                name += id2.id();
            }
            if (nInclude > 2)
            {
                name += ',';
                const ClassId& id3(ClassIdSpecialization<T3>::classId());
                name += id3.id();
            }
            name += '>';
        }
        return name;
    }

    template<class T1, class T2, class T3>
    inline std::string template_class_name(const std::string& templateName,
                                           const unsigned nInclude)
    {
        return template_class_name<T1,T2,T3>(templateName.c_str(), nInclude);
    }

    template<class T1, class T2, class T3, class T4>
    std::string template_class_name(const char* templateName,
                                    const unsigned nInclude)
    {
        assert(templateName);
        std::string name(templateName);
        if (nInclude)
        {
            name += '<';
            const ClassId& id1(ClassIdSpecialization<T1>::classId());
            name += id1.id();
            if (nInclude > 1)
            {
                name += ',';
                const ClassId& id2(ClassIdSpecialization<T2>::classId());
                name += id2.id();
            }    
            if (nInclude > 2)
            {
                name += ',';
                const ClassId& id3(ClassIdSpecialization<T3>::classId());
                name += id3.id();
            }
            if (nInclude > 3)
            {
                name += ',';
                const ClassId& id4(ClassIdSpecialization<T4>::classId());
                name += id4.id();
            }
            name += '>';
        }
        return name;
    }

    template<class T1, class T2, class T3, class T4>
    inline std::string template_class_name(const std::string& templateName,
                                           const unsigned n)
    {
        return template_class_name<T1,T2,T3,T4>(templateName.c_str(), n);
    }

    template<class T1, class T2, class T3, class T4, class T5>
    std::string template_class_name(const char* templateName,
                                    const unsigned nInclude)
    {
        assert(templateName);
        std::string name(templateName);
        if (nInclude)
        {
            name += '<';
            const ClassId& id1(ClassIdSpecialization<T1>::classId());
            name += id1.id();
            if (nInclude > 1)
            {
                name += ',';
                const ClassId& id2(ClassIdSpecialization<T2>::classId());
                name += id2.id();
            }
            if (nInclude > 2)
            {
                name += ',';
                const ClassId& id3(ClassIdSpecialization<T3>::classId());
                name += id3.id();
            }
            if (nInclude > 3)
            {
                name += ',';
                const ClassId& id4(ClassIdSpecialization<T4>::classId());
                name += id4.id();
            }
            if (nInclude > 4)
            {
                name += ',';
                const ClassId& id5(ClassIdSpecialization<T5>::classId());
                name += id5.id();
            }
            name += '>';
        }
        return name;
    }

    template<class T1, class T2, class T3, class T4, class T5>
    inline std::string template_class_name(const std::string& templateName,
                                           const unsigned n)
    {
        return template_class_name<T1,T2,T3,T4,T5>(templateName.c_str(), n);
    }

    template<class T1, class T2, class T3, class T4, class T5, class T6>
    std::string template_class_name(const char* templateName,
                                    const unsigned nInclude)
    {
        assert(templateName);
        std::string name(templateName);
        if (nInclude)
        {
            name += '<';
            const ClassId& id1(ClassIdSpecialization<T1>::classId());
            name += id1.id();
            if (nInclude > 1)
            {
                name += ',';
                const ClassId& id2(ClassIdSpecialization<T2>::classId());
                name += id2.id();
            }
            if (nInclude > 2)
            {
                name += ',';
                const ClassId& id3(ClassIdSpecialization<T3>::classId());
                name += id3.id();
            }
            if (nInclude > 3)
            {
                name += ',';
                const ClassId& id4(ClassIdSpecialization<T4>::classId());
                name += id4.id();
            }
            if (nInclude > 4)
            {
                name += ',';
                const ClassId& id5(ClassIdSpecialization<T5>::classId());
                name += id5.id();
            }
            if (nInclude > 5)
            {
                name += ',';
                const ClassId& id6(ClassIdSpecialization<T6>::classId());
                name += id6.id();
            }
            name += '>';
        }
        return name;
    }

    template<class T1, class T2, class T3, class T4, class T5, class T6>
    inline std::string template_class_name(const std::string& templateName,
                                           const unsigned n)
    {
        return template_class_name<T1,T2,T3,T4,T5,T6>(templateName.c_str(), n);
    }

    template<class T, std::size_t N>
    std::string stack_container_name(const char* templateName)
    {
        assert(templateName);
        const ClassId& id1(ClassIdSpecialization<T>::classId());
        std::ostringstream os;
        os << templateName << '<' << id1.id() << ',' << N << "(0)>";
        return os.str();
    }

    template<class T, std::size_t N>
    std::string stack_container_name(const std::string& templateName)
    {
        return stack_container_name<T,N>(templateName.c_str());
    }

    // Skip references in class ids
    template<class T>
    struct ClassIdSpecialization<T&>
    {
        inline static ClassId classId(const bool isPtr=false)
            {return ClassIdSpecialization<T>::classId(isPtr);}
    };

    // Skip pointers in class ids
    template<class T>
    struct ClassIdSpecialization<T*>
    {
        inline static ClassId classId(const bool /* isPtr */=false)
            {return ClassIdSpecialization<T>::classId(true);}
    };

    template<class T>
    struct ClassIdSpecialization<T* const>
    {
        inline static ClassId classId(const bool /* isPtr */=false)
        {return ClassIdSpecialization<T>::classId(true);}
    };

    template<class T>
    struct ClassIdSpecialization<T* volatile>
    {
        inline static ClassId classId(const bool /* isPtr */=false)
        {return ClassIdSpecialization<T>::classId(true);}
    };

    template<class T>
    struct ClassIdSpecialization<T* const volatile>
    {
        inline static ClassId classId(const bool /* isPtr */=false)
        {return ClassIdSpecialization<T>::classId(true);}
    };

    // Skip shared pointers in class ids
    template<class T>
    struct ClassIdSpecialization<CPP11_shared_ptr<T> >
    {
        inline static ClassId classId(const bool /* isPtr */=false)
        {return ClassIdSpecialization<T>::classId(true);}
    };

    template<class T>
    struct ClassIdSpecialization<const CPP11_shared_ptr<T> >
    {
        inline static ClassId classId(const bool /* isPtr */=false)
        {return ClassIdSpecialization<T>::classId(true);}
    };

    template<class T>
    struct ClassIdSpecialization<volatile CPP11_shared_ptr<T> >
    {
        inline static ClassId classId(const bool /* isPtr */=false)
        {return ClassIdSpecialization<T>::classId(true);}
    };

    template<class T>
    struct ClassIdSpecialization<const volatile CPP11_shared_ptr<T> >
    {
        inline static ClassId classId(const bool /* isPtr */=false)
        {return ClassIdSpecialization<T>::classId(true);}
    };

    // Skip IOPtr in class ids and do not turn on the pointer flag
    template<class T>
    struct ClassIdSpecialization<IOPtr<T> >
    {
        inline static ClassId classId(const bool isPtr=false)
        {return ClassIdSpecialization<T>::classId(isPtr);}
    };

    template<class T>
    struct ClassIdSpecialization<const IOPtr<T> >
    {
        inline static ClassId classId(const bool isPtr=false)
        {return ClassIdSpecialization<T>::classId(isPtr);}
    };

    template<class T>
    struct ClassIdSpecialization<volatile IOPtr<T> >
    {
        inline static ClassId classId(const bool isPtr=false)
        {return ClassIdSpecialization<T>::classId(isPtr);}
    };

    template<class T>
    struct ClassIdSpecialization<const volatile IOPtr<T> >
    {
        inline static ClassId classId(const bool isPtr=false)
        {return ClassIdSpecialization<T>::classId(isPtr);}
    };

    // Same thing for IOProxy
    template<class T>
    struct ClassIdSpecialization<IOProxy<T> >
    {
        inline static ClassId classId(const bool isPtr=false)
        {return ClassIdSpecialization<T>::classId(isPtr);}
    };

    template<class T>
    struct ClassIdSpecialization<const IOProxy<T> >
    {
        inline static ClassId classId(const bool isPtr=false)
        {return ClassIdSpecialization<T>::classId(isPtr);}
    };

    template<class T>
    struct ClassIdSpecialization<volatile IOProxy<T> >
    {
        inline static ClassId classId(const bool isPtr=false)
        {return ClassIdSpecialization<T>::classId(isPtr);}
    };

    template<class T>
    struct ClassIdSpecialization<const volatile IOProxy<T> >
    {
        inline static ClassId classId(const bool isPtr=false)
        {return ClassIdSpecialization<T>::classId(isPtr);}
    };

    // The remaining ClassId static functions
    template<class T>
    inline ClassId ClassId::makeId()
    {
        return ClassIdSpecialization<T>::classId();
    }

    namespace Private {
        template<bool, class T>
        struct CallClassId
        {
            static inline ClassId get(const T&)
                {return ClassIdSpecialization<T>::classId();}
        };

        template<class T>
        struct CallClassId<true,T>
        {
            static inline ClassId get(const T& obj) {return obj.classId();}
        };

        // The following class will check for the existence of two
        // possible signatures of the "classId" method:
        // "const ClassId& classId() const" and "ClassId classId() const".
        template<class Tp>
        class TypeHasClassIdHelper
        {
            template <typename T, T> struct TypeCheck;
            template <typename T> struct FcnType1
            {typedef ClassId (T::*fptr)() const;};
            template <typename T> struct FcnType2
            {typedef const ClassId& (T::*fptr)() const;};

            typedef char Yes;
            typedef struct {char a[2];} No;

            template <typename T> static Yes Has1(
                TypeCheck<typename FcnType1<T>::fptr, &T::classId>*);
            template <typename T> static No  Has1(...);

            template <typename T> static Yes Has2(
                TypeCheck<typename FcnType2<T>::fptr, &T::classId>*);
            template <typename T> static No  Has2(...);

        public:
            static const bool value = ((sizeof(Has1<Tp>(0)) == sizeof(Yes)) ||
                                       (sizeof(Has2<Tp>(0)) == sizeof(Yes)));
        };

        template<class T, bool b=IOIsClassType<T>::value>
        struct TypeHasClassId
        {
            static const bool value = 0;
        };

        template <typename T>
        struct TypeHasClassId<T, true>
        {
            static const bool value = TypeHasClassIdHelper<T>::value;
        };
    }

    template<class T>
    inline ClassId ClassId::itemId(const T& item)
    {
        // Make sure that item is not a pointer.
        static_assert((!IOIsAnyPtr<T>::value), "can not use pointers with this method");

        // If the classId() function is avalable for this item, call it
        // (it could be virtual). Otherwise, call the generic method.
        return Private::CallClassId<
            Private::TypeHasClassId<T>::value,T>::get(item);
    }
}

// Class ids for standard types
gs_specialize_class_id(float, 0)
gs_specialize_class_id(double, 0)
gs_specialize_class_id(long double, 0)
gs_specialize_class_id(int, 0)
gs_specialize_class_id(unsigned, 0)
gs_specialize_class_id(long, 0)
gs_specialize_class_id(long long, 0)
gs_specialize_class_id(unsigned long, 0)
gs_specialize_class_id(unsigned long long, 0)
gs_specialize_class_id(short, 0)
gs_specialize_class_id(unsigned short, 0)
gs_specialize_class_id(bool, 0)
gs_specialize_class_id(char, 0)
gs_specialize_class_id(unsigned char, 0)
gs_specialize_class_id(signed char, 0)
gs_specialize_class_id(void, 0)
gs_specialize_class_id(std::string, 0)

// Class ids for some standard library templates
// used by this I/O package
gs_specialize_template_id_T(std::less, 0, 1)
gs_specialize_template_id_T(std::equal_to, 0, 1)
gs_specialize_template_id_T(std::allocator, 0, 1)
gs_specialize_template_id_T(std::char_traits, 0, 1)
gs_specialize_template_id_TT(std::vector, 0, 1)
gs_specialize_template_id_TT(std::pair, 0, 2)
gs_specialize_template_id_TTT(std::basic_string, 0, 2)


#endif // GENERS_CLASSID_HH_

