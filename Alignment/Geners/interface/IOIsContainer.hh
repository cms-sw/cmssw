#ifndef GENERS_IOISCONTAINER_HH_
#define GENERS_IOISCONTAINER_HH_

#include <string>

namespace gs {
    // In the following template, enum "IsContainer" is evaluated to 1
    // at compile time if T has T::value_type typedef
    template <typename T>
    class IOIsContainer
    {
    private:
        typedef char One;
        typedef struct {char a[2];} Two;
        template<typename C> static One test(typename C::value_type const*);
        template<typename C> static Two test(...);

    public:
        enum {value = sizeof(IOIsContainer<T>::template test<T>(0)) == 1};
    };

    // Char strings get a special treatment
    template <>
    class IOIsContainer<std::string>
    {
    public:
        enum {value = 0};
    };

    template <>
    class IOIsContainer<const std::string>
    {
    public:
        enum {value = 0};
    };

    template <>
    class IOIsContainer<volatile std::string>
    {
    public:
        enum {value = 0};
    };

    template <>
    class IOIsContainer<const volatile std::string>
    {
    public:
        enum {value = 0};
    };
}

#endif // GENERS_IOISCONTAINER_HH_

