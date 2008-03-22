# $Id: Makefile,v 1.4 2007/12/10 13:42:51 aryd Exp $

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

Sources = PixelCalibConfiguration.cc PixelROCDACSettings.cc PixelMaskAllPixels.cc PixelTrimAllPixels.cc PixelMaskBase.cc PixelConfigBase.cc PixelCalibBase.cc PixelROCMaskBits.cc PixelTrimBase.cc PixelROCTrimBits.cc PixelDACSettings.cc PixelROCName.cc PixelNameTranslation.cc PixelHdwAddress.cc PixelDetectorConfig.cc PixelFEDCard.cc PixelFECConfig.cc PixelFEDConfig.cc PixelTBMSettings.cc PixelModuleName.cc PixelTrimCommon.cc PixelPortCardConfig.cc PixelDelay25Calib.cc PixelPortcardMap.cc PixelDACScanRange.cc PixelTTCciConfig.cc PixelFECParameters.cc PixelFEDParameters.cc PixelLTCConfig.cc PixelFEDTestDAC.cc PixelTKFECConfig.cc PixelTKFECParameters.cc PixelChannel.cc PixelTBMChannel.cc PixelROCStatus.cc PixelConfigurationVerifier.cc

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
