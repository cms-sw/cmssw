cmake_minimum_required(VERSION 3.7)

project(optics_generator)

#set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})

find_package(ROOT REQUIRED COMPONENTS RIO Net)
find_package(XercesC REQUIRED)
include(${ROOT_USE_FILE})

#ROOT_GENERATE_DICTIONARY(FitCint TMultiDimFet.h LHCOpticsApproximator.h TNtupleDcorr.h LINKDEF stub/FitCintLinkDef.h)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories("interface" ${XercesC_INCLUDE_DIRS} ${ROOT_INCLUDE_DIRS})

set(SOURCE_FILES
        interface/ApertureAnalyser.h
        interface/CurrentMemoryUsage.h
        interface/DPEDataGenerator.h
        interface/FitConfig.h
        interface/FitData.h
        interface/FitEngine.h
        interface/h101.h
        interface/LHCOpticsApproximator.h
        interface/LHCOpticsApproximator_gcc.h
        interface/LHCOpticsApproximator_rootcint.h
        interface/MADParamGenerator.h
        interface/RPXMLConfig.h
        interface/TMultiDimFet.h
        interface/TNtupleDcorr.h
        src/ApertureAnalyser.cc
        src/CurrentMemoryUsage.cc
        src/DPEDataGenerator.cc
        src/FindApproximation.cc
        src/FitConfig.cc
        src/FitData.cc
        src/FitEngine.cc
        src/h101.cc
        src/LHCOpticsApproximator.cc
        src/MADParamGenerator.cc
        src/RPXMLConfig.cc
        src/TMultiDimFet.cc
        src/TNtupleDcorr.cc
        stub/FitCint.cc
        )

#execute_process(/usr/bin/rootcint -f .//stub/FitCint.cc -c -I/usr/include/root -I/usr/include/root -I.//interface -I.//ConfigXML/interface -I./ .//interface/TMultiDimFet.h .//interface/LHCOpticsApproximator.h .//interface/TNtupleDcorr.h .//stub/FitCintLinkDef.h)


add_library(Fit SHARED ${SOURCE_FILES})
target_include_directories(Fit PUBLIC "interface" ${XercesC_INCLUDE_DIRS} ${ROOT_INCLUDE_DIRS})
target_link_libraries(Fit ${XercesC_LIBRARIES} ${ROOT_LIBRARIES})

add_executable(optics_generator src/FindApproximation.cc)
target_include_directories(optics_generator PUBLIC "interface" ${XercesC_INCLUDE_DIRS} ${ROOT_INCLUDE_DIRS})
target_link_libraries(optics_generator PUBLIC Fit ${XercesC_LIBRARIES} ${ROOT_LIBRARIES} CLHEP)
