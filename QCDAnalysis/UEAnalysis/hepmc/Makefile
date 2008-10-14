################################################################################
# Makefile for building examples for the MC School
#
#    make example_CascadeToHepMC.exe -- compile cascade example
#    make hepmc_analysis.exe         -- compile HepMC analysis example
#    make ReadHepMC.exe	  	     -- compile example creating human readable event list
#    make clean                      -- remove .o files
#    make fresh                      -- complete restart/garbage collection
#
############################## Directory paths #################################

  ROOTdir       = /afs/cern.ch/sw/lcg/external/root/5.18.00/slc4_amd64_gcc34/root
#
  SYSNAME       = slc4_amd64_gcc34
  PREFIX        = /afs/cern.ch/sw/lcg/external
  FastJetdir    = $(PREFIX)/fastjet/2.3.0/$(SYSNAME)
  HepMCdir      = $(PREFIX)/HepMC/2.03.06/$(SYSNAME)
  HepPDTdir     = $(PREFIX)/HepPDT/3.02.02/$(SYSNAME)
  CLHEPdir      = $(PREFIX)/clhep/2.0.3.2/$(SYSNAME)

############################### Libraries ######################################

  INCLUDEC      = -I$(PREFIX)/MCGenerators/cascade/2.0.1/$(SYSNAME)/include
  INCLUDEH      = -I$(FastJetdir)/include \
                  -I$(HepMCdir)/include -I$(CLHEPdir)/include \
                  -I$(HepPDTdir)/include -I$(ROOTdir)/include
  INCLUDES      = $(INCLUDEH) $(INCLUDEC)
  ROOTGLIBS     = -L$(ROOTdir)/lib -lCore -lCint -lRIO -lNet -lHist \
                  -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript \
                  -lMatrix -lPhysics -lGui -pthread -lm -ldl -rdynamic
  CLHEP_LIB     = -L$(CLHEPdir)/lib -lCLHEP

  FastJetlib    = -L$(FastJetdir)/lib -lfastjet -lsiscone -l SISConePlugin
  HepPDTlib     = -L$(HepPDTdir)/lib -lHepPDT -lHepPID
  HepMClib      = $(HepMCdir)/lib/libHepMC.so $(HepMCdir)/lib/libHepMCfio.a
  LINK_LIBS     = -lfrtbegin -lg2c
  HDRS          = $(HepMCdir)/include/HepMC/*.h *.h

  Pythia_LIB	= -L$(PREFIX)/lib -lpythia6 -lpythia6_dummy -lpythia6_pdfdummy
  Cascade_LIB	= -L$(PREFIX)/lib -lbases -lcascade2
  CERN_LIB	= -L/opt/products/cernlib/2005/lib -lmathlib -lkernlib -lpacklib

############################## Compiler options ################################

  CXX           = g++
  F77           = g77
  CXXFLAGS      = -O2 -Wall -fPIC -pthread   $(INCLUDES) 
  FFLAGS        = -O2 -fno-second-underscore $(INCLUDES)
  LD            = g++

################################ Object files ##################################

  pythia6_OBJ	= initpydata.o initPythia.o
  cascade_OBJ	= example_CascadeToHepMC.o upevnt.o upinit.o 
  ReadHepMC_OBJ	= ReadHepMC.o 
             
  EXAMPLES	= example_MyCascadeToHepMC.exe \
		  hepmc_analysis.exe \
		  ReadHepMC.exe 
              
################################################################################ 

.SUFFIXES:      .o .cxx .f .exe
all:	$(EXAMPLES)


example_CascadeToHepMC.exe: $(cascade_OBJ) $(pythia6_OBJ)
	@echo "Building $@ ..."
	$(CXX) $(CXXFLAGS) $(cascade_OBJ) $(pythia_OBJ) \
	$(Cascade_LIB) $(Pythia_LIB) $(Cascade_LIB) $(HepMClib) \
	$(CERN_LIB) $(LINK_LIBS) -o $@


hepmc_analysis.exe: hepmc_analysis.o
	@echo "Building $@ ..."
	$(CXX) $(CXXFLAGS) hepmc_analysis.o \
	$(HepMClib) $(HepPDTlib) $(CLHEP_LIB) \
	$(ROOTGLIBS) $(LINK_LIBS) -o $@

PhotonJetAnalysis: PhotonJetAnalysis.o
	@echo "Building $@ ..."
	$(CXX) $(CXXFLAGS) PhotonJetAnalysis.o \
	$(FastJetlib) \
	$(HepMClib) $(HepPDTlib) $(CLHEP_LIB) \
	$(ROOTGLIBS) $(LINK_LIBS) -o $@

UEAnalysisRootupleProducer: UEAnalysisRootupleProducer.o
	@echo "Building $@ ..."
	$(CXX) $(CXXFLAGS) UEAnalysisRootupleProducer.o \
	$(FastJetlib) \
	$(HepMClib) $(HepPDTlib) $(CLHEP_LIB) \
	$(ROOTGLIBS) $(LINK_LIBS) -o $@

ReadHepMC.exe: $(ReadHepMC_OBJ) 
	echo "Building $@ ..."
	$(CXX) $(CXXFLAGS) $(ReadHepMC_OBJ) $(HepMClib) \
	$(CERN_LIB) $(LINK_LIBS) -o $@

##############################################################################
# instructions for building a .o file from a .cxx file
#
.cc.o:         $(HDRS) $<
	@echo "Compiling $< with $(CXX) $(CXXFLAGS) "
	@$(CXX) $(CXXFLAGS) -c $< -o $@

###############################################################################
# instructions for building a .o file from a .f file
#
.f.o:           $<
	@echo "Compiling $< with $(F77) ..."
	@$(F77) $(FFLAGS) -c $< -o $@

###############################################################################
# gmake clean       removes all garbage from HepMC directories.
#
clean:
	rm -f *.o

###############################################################################
# gmake fresh       removes all compiled libraries, executables, +garbage
#                   to restart from scratch
fresh: 
	$(MAKE) clean --no-print-directory
	rm -f *.exe
	rm -f *.dat
