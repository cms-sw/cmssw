# $Id: Makefile,v 1.9 2009/10/20 10:13:22 aryd Exp $

#########################################################################
# XDAQ Components for Distributed Data Acquisition                      #
# Copyright (C) 2007, Cornell U.  			                #
# All rights reserved.                                                  #
# Authors: A. Ryd                					#
#                                                                       #
#########################################################################

#
# This makefile is for the online build of this package
#


include $(XDAQ_ROOT)/config/mfAutoconf.rules
include $(XDAQ_ROOT)/config/mfDefs.$(XDAQ_OS)

Project=pixel
Package=CalibFormats/SiPixelObjects
PackageName=SiPixelObjects


Sources = $(wildcard src/*.cc)

IncludeDirs = \
        $(BUILD_HOME)/$(Project) \

LibraryDirs = 

UserSourcePath = \
	src

UserCFlags = -O
UserCCFlags = -g -O -Wno-long-long
UserDynamicLinkFlags =
UserStaticLinkFlags =
UserExecutableLinkFlags =

# These libraries can be platform specific and
# potentially need conditional processing
#

Libraries =
ExternalObjects = 

#
# Compile the source files and create a shared library
#
#ifdef Library
DynamicLibrary= SiPixelObjects
#DynamicLibrary= $(Library)
#endif



ifdef Executable
Libraries=toolbox xdata xcept xoap xerces-c log4cplus mimetic
Executables= $(Executable).cc
endif

include $(XDAQ_ROOT)/config/Makefile.rules
include $(BUILD_HOME)/$(Project)/RPM.version
include $(XDAQ_ROOT)/config/mfRPM.rules

