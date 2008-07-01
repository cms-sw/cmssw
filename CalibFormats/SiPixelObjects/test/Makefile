
# XDAQ_ROOT should be defined 
# define XDAQ_OS amd XDAQ_PLATRFORM
include $(XDAQ_ROOT)/config/mfAutoconf.rules
include $(XDAQ_ROOT)/config/mfDefs.$(XDAQ_OS)

PACKAGE=pixel

CC_OPTIM=-fPIC -DLINUX 
#ADDED_CFLAGS=-g -DDEBUG
ADDED_CFLAGS=-DNDEBUG 

IncludeDirs = \
        -I$(BUILD_HOME)/$(PACKAGE) \
        -I$(XDAQ_ROOT)/include


LibraryDirs = \
        -L$(BUILD_HOME)/pixel/lib \
        -L$(XDAQ_ROOT)/lib 


Libraries= -l SiPixelObjects -l PixelTestStandUtilities


HAL = -L$(XDAQ_ROOT)/daq/hal/lib/$(XDAQ_OS)/$(XDAQ_PLATFORM)  -lGenericHAL -lHALUtilities -lCAENVME -lconfig


LIBS = $(LibraryDirs) $(Libraries)  

PIXEL_INCLUDE = $(XDAQ_ROOT)/pixel/ 
INC = -I.  $(IncludeDirs) -I$(PIXEL_INCLUDE) 

CFLAGS=$(CC_OPTIM) $(ADDED_CFLAGS) -Wall -Wstrict-prototypes $(INC)


testPixelCalib:	testPixelCalib.cc
	g++ -o testPixelCalib -I$(XDAQ_ROOT)/pixel $(IncludeDirs) testPixelCalib.cc $(LIBS) 


