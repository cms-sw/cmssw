ifeq ($(shell uname), Linux) 
   BoostLibs :=  boost_thread-gcc34-mt-1_38  boost_signals-gcc34-mt-1_38  boost_filesystem-gcc34-mt-1_38  boost_program_options-gcc34-mt-1_38  boost_regex-gcc34-mt-1_38
   RootLibs := Cint Core  Net Tree Hist Matrix MathCore MathMore Reflex EG Eve FTGL Ged Geom GeomPainter Gpad Graf Graf3d Gui Physics RGL Rint TreePlayer
   ProjectLibs := $(BoostLibs) $(RootLibs) CLHEP-1.9.4.2 gsl gslcblas HepMC pcre  sigc-2.0 nsl crypt dl z

   ProjectIncludes :=  external/inc/boost/1.34.1-cms/include external/inc/clhep/1.9.3.2-cms/include
   ProjectIncludes += external/inc/gsl/1.10-cms/include 
   ProjectIncludes += external/inc/hepmc/2.03.06/include
   ProjectIncludes +=  external/inc/sigcpp/2.0.17-cms/include/sigc++-2.0
else 
   ifeq ($(shell uname), Darwin)
      BoostLibs := boost_system-xgcc40-mt-1_38 boost_thread-xgcc40-mt-1_38  boost_signals-xgcc40-mt-1_38  boost_filesystem-xgcc40-mt-1_38 boost_program_options-xgcc40-mt-1_38 boost_regex-xgcc40-mt-1_38 
      RootLibs :=  Core RIO TreePlayer Gpad Graf3d Graf Hist Matrix Physics Postscript Cintex Cint Core Net Tree MathCore MathMore GenVector Reflex
      ProjectLibs := $(BoostLibs) $(RootLibs) CLHEP-1.9.4.2 gsl gslcblas Core RIO HepMC sigc-2.0.0 dl

      ProjectIncludes :=  external/inc/boost/1.38.0/include
      ProjectIncludes +=  external/inc/clhep/1.9.4.2/include
      ProjectIncludes += external/inc/gsl/1.10/include external/inc/hepmc/2.03.06/include
      ProjectIncludes += external/inc/sigcpp/2.0.17/include/sigc++-2.0
      ProjectIncludes += /usr/X11R6/include
   endif     
endif
