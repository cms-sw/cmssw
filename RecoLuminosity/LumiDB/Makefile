# Packages to be built
Project=RecoLuminosity
Package=LumiDB

IncludeDirs = \
	$(BUILD_HOME)/$(Project)/$(Package)/include \
	$(BUILD_HOME)/$(Project)/HLXReadOut/CoreUtils/include \
	/opt/oracle/current/sdk/include

Sources = \
	DBWriter.cc

# some XDAQ-required stuff
include $(XDAQ_ROOT)/config/mfAutoconf.rules
include $(XDAQ_ROOT)/config/mfDefs.$(XDAQ_OS)

LibraryDirs = \
	/opt/oracle/current/lib/

# compiler flags
UserCFlags =
UserCCFlags = -D_REENTRANT
UserDynamicLinkFlags =
UserStaticLinkFlags =
UserExecutableLinkFlags =

ExternalObjects = 

DynamicLibrary= LumiDB
StaticLibrary=
Executables= 
Libraries= clntsh occi10 nnz10 n10
TestExecutables= 
TestLibraries= 

# targets
all : _all 

#standard XDAQ C++ compilation
include $(XDAQ_ROOT)/config/Makefile.rules
