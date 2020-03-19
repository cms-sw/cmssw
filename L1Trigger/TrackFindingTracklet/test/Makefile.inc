#common definitions go in here
CXX           = g++
CXXFLAGS      = -O -Wall  -fPIC -g -ansi -Wextra -Wno-deprecated -std=c++11
LD            = g++ 
LDFLAGS       = -O
LIBDIR       = $(WORKDIR)
AR            = ar
ARFLAGS       = cr

OS := $(shell uname -s)
ifeq ($(OS),Darwin)
	SOFLAGS       = -dynamiclib
else
	SOFLAGS       = -shared
endif	

ifndef ROOTSYS
$(error *** Please set up Root)
endif


#find out about the installed ROOT
ROOTCONFIG   := $(ROOTSYS)/bin/root-config
ROOTCFLAGS   := $(shell $(ROOTCONFIG) --cflags)
ROOTLDFLAGS  := $(shell $(ROOTCONFIG) --ldflags) $(shell $(ROOTCONFIG) --libs --nonew --glibs)

#append to options
CXXFLAGS     += $(ROOTCFLAGS)
LDFLAGS      += $(ROOTLDFLAGS)
# Needed for ROOT-Tree
#LDFLAGS      += -L./ FPGAEvent_cxx.so


LIBS = -lHtml


ifneq ($(CMSSW_BASE),)
	ROOTFIT_INCLUDE := $(shell cd $(CMSSW_BASE); scram tool info roofitcore | grep INCLUDE= | sed 's|INCLUDE=||')
	ROOTFIT_LIBDIR := $(shell cd $(CMSSW_BASE); scram tool info roofitcore | grep LIBDIR= | sed 's|LIBDIR=||')
	
	CINTINCLUDES := -I$(ROOTFIT_INCLUDE)
	CXXFLAGS     += -I$(ROOTFIT_INCLUDE)
	LDFLAGS      += -L$(ROOTFIT_LIBDIR)
endif
