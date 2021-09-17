# Local setup, change as desired
SRC_DIR:=$(PWD)

CXXOTHERFLAGS :=
UNAME:=$(shell uname -s)
ifeq ($(UNAME), Linux)
CXXOTHERFLAGS := -D__USE_XOPEN2K8
endif

# Compiler stuff
SRC_INCLUDES=-I$(SRC_DIR) -I$(SRC_DIR)/include
CXXFLAGS=-O3 -g -std=c++11 -fPIC $(CXXOTHERFLAGS)
LDFLAGS= -std=c++11
CXX:=g++
LINKER:=g++

# CPP unit stuff
CPPUNIT_INCLUDES := -I/opt/local/include
CPPUNIT_LIB_PATH := -L/opt/local/lib
CPPUNIT_LIB := $(CPPUNIT_LIB_PATH) -lcppunit

# Boost stuff
BOOST_INC=/opt/local/include/boost
BOOST_INCLUDES := -I$(BOOST_INC)
