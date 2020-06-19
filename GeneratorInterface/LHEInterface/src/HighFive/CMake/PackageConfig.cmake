add_library(HighFive INTERFACE)
target_link_libraries(HighFive INTERFACE ${HDF5_LIBRARIES})
target_include_directories(HighFive INTERFACE
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>"
  "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_include_directories(HighFive SYSTEM INTERFACE ${HDF5_INCLUDE_DIRS})
if(USE_BOOST)
  target_include_directories(HighFive SYSTEM INTERFACE ${Boost_INCLUDE_DIR})
  target_compile_definitions(HighFive INTERFACE -DH5_USE_BOOST)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/highfive
		DESTINATION ${INCLUDE_INSTALL_DIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/HighFiveConfig.cmake.in
  ${PROJECT_BINARY_DIR}/HighFiveConfig.cmake
  INSTALL_DESTINATION share/${PROJECT_NAME}/CMake
  )
install(FILES ${PROJECT_BINARY_DIR}/HighFiveConfig.cmake
  DESTINATION share/${PROJECT_NAME}/CMake)

# Generate ${PROJECT_NAME}Targets.cmake; is written after the CMake run
# succeeds. Provides IMPORTED targets when using this project from the install
# tree.
install(EXPORT HighFiveTargets FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION share/${PROJECT_NAME}/CMake)

install(TARGETS HighFive EXPORT ${PROJECT_NAME}Targets
  INCLUDES DESTINATION include)

export(EXPORT HighFiveTargets
  FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")
