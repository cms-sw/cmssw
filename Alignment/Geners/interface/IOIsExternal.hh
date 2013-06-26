#ifndef GENERS_IOISEXTERNAL_HH_
#define GENERS_IOISEXTERNAL_HH_

namespace gs {
    template <class T>
    struct IOIsExternal
    {
        enum {value = 0};
    };
}

// Use the following macro (outside of any namespace)
// to declare some type as external for I/O purposes
//
#define gs_declare_type_external(T) /**/                                   \
namespace gs {                                                             \
    template <> struct IOIsExternal<T> {enum {value = 1};};                \
    template <> struct IOIsExternal<T const> {enum {value = 1};};          \
    template <> struct IOIsExternal<T volatile> {enum {value = 1};};       \
    template <> struct IOIsExternal<T const volatile> {enum {value = 1};}; \
}

// Use the following macro (outside of any namespace)
// to declare some template parameterized by one argument
// as external for I/O purposes
//
#define gs_declare_template_external_T(name) /**/                          \
namespace gs {                                                             \
    template <class T> struct IOIsExternal< name <T> >                     \
       {enum {value = 1};};                                                \
    template <class T> struct IOIsExternal<const name <T> >                \
       {enum {value = 1};};                                                \
    template <class T> struct IOIsExternal<volatile name <T> >             \
       {enum {value = 1};};                                                \
    template <class T> struct IOIsExternal<const volatile name <T> >       \
       {enum {value = 1};};                                                \
}

// Use the following macro (outside of any namespace)
// to declare some template parameterized by two arguments
// as external for I/O purposes
//
#define gs_declare_template_external_TT(name) /**/                         \
namespace gs {                                                             \
    template <class T,class U> struct IOIsExternal< name <T,U> >           \
       {enum {value = 1};};                                                \
    template <class T,class U> struct IOIsExternal<const name <T,U> >      \
       {enum {value = 1};};                                                \
    template <class T,class U> struct IOIsExternal<volatile name <T,U> >   \
       {enum {value = 1};};                                                \
    template <class T,class U> struct IOIsExternal<const volatile name <T,U> >\
       {enum {value = 1};};                                                \
}

#endif // GENERS_IOISEXTERNAL_HH_

