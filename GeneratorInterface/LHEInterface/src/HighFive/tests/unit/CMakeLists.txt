if(NOT HIGHFIVE_UNIT_TESTS OR NOT USE_BOOST)
  return()
endif()

include_directories(${PROJECT_SOURCE_DIR}/include ${HDF5_INCLUDE_DIRS})

if(NOT MSVC)
  # silent boost warnings
  add_definitions(-Wno-unused-parameter -Wno-unused-local-typedef)
endif()

if(NOT Boost_USE_STATIC_LIBS)
  add_definitions(-DBOOST_TEST_DYN_LINK=TRUE)
endif()

## base tests
file(GLOB tests_high_five_src "*base.cpp")

add_executable(tests_high_five_cpp11_bin ${tests_high_five_src})
target_link_libraries(tests_high_five_cpp11_bin
${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} ${HDF5_C_LIBRARIES})

if(HIGHFIVE_PARALLEL_HDF5)
    target_link_libraries(tests_high_five_cpp11_bin ${MPI_C_LIBRARIES})
endif()

target_compile_options(tests_high_five_cpp11_bin PRIVATE "${HIGHFIVE_CPP_STD_FLAG}")
target_compile_definitions(tests_high_five_cpp11_bin PRIVATE "-DHIGHFIVE_CPP11_ENABLE=1")

add_test(NAME tests_high_five_cpp11 COMMAND tests_high_five_cpp11_bin)



if(HIGHFIVE_PARALLEL_HDF5)

  include(TestHelpers)

  ## parallel MPI tests
  file(GLOB tests_parallel_src "*parallel.cpp")

  add_executable(tests_parallel_bin ${tests_parallel_src})
  target_link_libraries(tests_parallel_bin
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} ${HDF5_C_LIBRARIES}
    ${MPI_C_LIBRARIES})
  add_test(NAME tests_parallel COMMAND ${TEST_MPI_EXEC_PREFIX_DEFAULT} -n 8
    ${CMAKE_CURRENT_BINARY_DIR}/tests_parallel_bin)

endif()

