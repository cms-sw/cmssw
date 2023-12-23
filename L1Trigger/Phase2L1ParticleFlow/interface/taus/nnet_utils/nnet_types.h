#ifndef NNET_TYPES_H_
#define NNET_TYPES_H_

#include <assert.h>
#include <cstddef>
#include <cstdio>

namespace nnet {

// Fixed-size array
template <typename T, unsigned N> struct array {
    typedef T value_type;
    int size = N;

    T data[N];

    T &operator[](size_t pos) { return data[pos]; }

    const T &operator[](size_t pos) const { return data[pos]; }

    array &operator=(const array &other) {
        if (&other == this)
            return *this;

        assert(N == other.size && "Array sizes must match.");

        for (unsigned i = 0; i < N; i++) {
            data[i] = other[i];
        }
        return *this;
    }
};

} // namespace nnet

#endif
