# Packages to be built
Project=RecoLuminosity
Package=LumiDB

IncludeDirs = \
	$(XDAQ_ROOT)/$(Project)/$(Package)/include \
	$(XDAQ_ROOT)/$(Project)/HLXReadOut/CoreUtils/include \
	$(ORACLE_HOME)/rdbms/demo \
	$(ORACLE_HOME)/rdbms/public \
	$(ORACLE_HOME)/plsql/public \
	$(ORACLE_HOME)/network/public

Sources = \
	DBWriter.cc

# some XDAQ-required stuff
include $(XDAQ_ROOT)/config/mfAutoconf.rules
include $(XDAQ_ROOT)/config/mfDefs.$(XDAQ_OS)

LibraryDirs = \
	$(ORACLE_HOME)/lib/ \
	$(ORACLE_HOME)/rdbms/

# compiler flags
UserCFlags =
UserCCFlags = 
UserDynamicLinkFlags =
UserStaticLinkFlags =
UserExecutableLinkFlags =

ExternalObjects = 

DynamicLibrary= LumiDB
StaticLibrary=
Executables= clntsh occi10 nnz10 n10
Libraries= 
TestExecutables= 
TestLibraries= 

# targets
all : _all 

#standard XDAQ C++ compilation
include $(XDAQ_ROOT)/config/Makefile.rules
