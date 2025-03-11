#ifndef HeterogeneousCore_MPICore_plugins_macros_h
#define HeterogeneousCore_MPICore_plugins_macros_h

#include <boost/preprocessor.hpp>

#include <mpi.h>

namespace mpi_traits {
  template <typename T>
  constexpr inline size_t mpi_length = 1;

  template <typename T, size_t N>
  constexpr inline size_t mpi_length<T[N]> = N;

  template <typename T>
  struct mpi_type {
    inline static const MPI_Datatype value = MPI_DATATYPE_NULL;
  };

  template <typename T, size_t N>
  struct mpi_type<T[N]> {
    inline static const MPI_Datatype value = mpi_type<T>::value;
  };

  template <>
  struct mpi_type<char> {
    inline static const MPI_Datatype value = MPI_CHAR;
  };

  template <>
  struct mpi_type<unsigned char> {
    inline static const MPI_Datatype value = MPI_UNSIGNED_CHAR;
  };

  template <>
  struct mpi_type<wchar_t> {
    inline static const MPI_Datatype value = MPI_WCHAR;
  };

  template <>
  struct mpi_type<short int> {
    inline static const MPI_Datatype value = MPI_SHORT;
  };

  template <>
  struct mpi_type<unsigned short int> {
    inline static const MPI_Datatype value = MPI_UNSIGNED_SHORT;
  };

  template <>
  struct mpi_type<int> {
    inline static const MPI_Datatype value = MPI_INT;
  };

  template <>
  struct mpi_type<unsigned int> {
    inline static const MPI_Datatype value = MPI_UNSIGNED;
  };

  template <>
  struct mpi_type<long int> {
    inline static const MPI_Datatype value = MPI_LONG;
  };

  template <>
  struct mpi_type<unsigned long int> {
    inline static const MPI_Datatype value = MPI_UNSIGNED_LONG;
  };

  template <>
  struct mpi_type<long long int> {
    inline static const MPI_Datatype value = MPI_LONG_LONG;
  };

  template <>
  struct mpi_type<unsigned long long int> {
    inline static const MPI_Datatype value = MPI_UNSIGNED_LONG_LONG;
  };

  template <>
  struct mpi_type<float> {
    inline static const MPI_Datatype value = MPI_FLOAT;
  };

  template <>
  struct mpi_type<double> {
    inline static const MPI_Datatype value = MPI_DOUBLE;
  };

  template <>
  struct mpi_type<long double> {
    inline static const MPI_Datatype value = MPI_LONG_DOUBLE;
  };

  template <>
  struct mpi_type<std::byte> {
    inline static const MPI_Datatype value = MPI_BYTE;
  };
}  // namespace mpi_traits

// clang-format off

#define _GET_MPI_TYPE_LENGTH_IMPL(STRUCT, FIELD)                                                    \
  mpi_traits::mpi_length<decltype(STRUCT ::FIELD)>

#define _GET_MPI_TYPE_LENGTH(R, STRUCT, FIELD)                                                      \
  _GET_MPI_TYPE_LENGTH_IMPL(STRUCT, FIELD),

#define _GET_MPI_TYPE_LENGTHS(STRUCT, ...)                                                          \
  BOOST_PP_SEQ_FOR_EACH(_GET_MPI_TYPE_LENGTH, STRUCT, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define _GET_MPI_TYPE_OFFSET_IMPL(STRUCT, FIELD)                                                    \
  offsetof(STRUCT, FIELD)

#define _GET_MPI_TYPE_OFFSET(R, STRUCT, FIELD)                                                      \
  _GET_MPI_TYPE_OFFSET_IMPL(STRUCT, FIELD),

#define _GET_MPI_TYPE_OFFSETS(STRUCT, ...)                                                          \
  BOOST_PP_SEQ_FOR_EACH(_GET_MPI_TYPE_OFFSET, STRUCT, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define _GET_MPI_TYPE_TYPEID_IMPL(STRUCT, FIELD)                                                    \
  mpi_traits::mpi_type<decltype(STRUCT :: FIELD)>::value

#define _GET_MPI_TYPE_TYPEID(R, STRUCT, FIELD)                                                      \
  _GET_MPI_TYPE_TYPEID_IMPL(STRUCT, FIELD),

#define _GET_MPI_TYPE_TYPEIDS(STRUCT, ...)                                                          \
  BOOST_PP_SEQ_FOR_EACH(_GET_MPI_TYPE_TYPEID, STRUCT, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define DECLARE_MPI_TYPE(TYPE, STRUCT, ...)                                                         \
  _Pragma("GCC diagnostic push");                                                                   \
  _Pragma("GCC diagnostic ignored \"-Winvalid-offsetof\"");                                         \
  {                                                                                                 \
    constexpr int lenghts[] = {_GET_MPI_TYPE_LENGTHS(STRUCT, __VA_ARGS__)};                         \
    constexpr MPI_Aint displacements[] = {_GET_MPI_TYPE_OFFSETS(STRUCT, __VA_ARGS__)};              \
    const MPI_Datatype types[] = {_GET_MPI_TYPE_TYPEIDS(STRUCT, __VA_ARGS__)};                      \
    MPI_Type_create_struct(std::size(lenghts), lenghts, displacements, types, &TYPE);               \
    MPI_Type_commit(&TYPE);                                                                         \
  }                                                                                                 \
  _Pragma("GCC diagnostic pop")

// clang-format on

#endif  // HeterogeneousCore_MPICore_plugins_macros_h
