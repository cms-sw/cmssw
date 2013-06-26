#ifndef GENERS_CLEARIFPOINTER_HH_
#define GENERS_CLEARIFPOINTER_HH_

namespace gs {
    template <class T>
    struct ClearIfPointer
    {
        static void clear(T&) {}
    };

    template <class T>
    struct ClearIfPointer<T*>
    {
        static void clear(T*& ptr) {ptr = 0;}
    };

    // The following will set object value to 0 if object is a pointer
    template <class T>
    void clearIfPointer(T& obj) {ClearIfPointer<T>::clear(obj);}
}

#endif // GENERS_CLEARIFPOINTER_HH_

