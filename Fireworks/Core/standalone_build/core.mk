ifeq ($(shell uname), Linux)
   BoostLibs :=  boost_thread-gcc34-mt-1_38  boost_signals-gcc34-mt-1_38  boost_filesystem-gcc34-mt-1_38  boost_program_options-gcc34-mt-1_38  boost_regex-gcc34-mt-1_38  
   RootLibs := Cintex Cint Core  Net Tree TreePlayer Gpad Graf3d Graf Hist Matrix Physics Postscript Hist Matrix MathCore MathMore GenVector Minuit Minuit2 Physics Reflex
   CoreLibs :=   $(BoostLibs) $(RootLibs) CLHEP-1.9.4.2  gsl gslcblas HepMC pcre sigc-2.0 nsl crypt dl uuid
   
   CoreIncludes := external/inc/boost/1.38.0/include external/inc/clhep/1.9.4.2-cms2/include 
   CoreIncludes += external/inc/gsl/1.10-cms/include external/inc/hepmc/2.03.06-cms3/include
   CoreIncludes += external/inc/pcre/4.4-cms/include external/inc/sigcpp/2.0.17-cms/include/sigc++-2.0 
   CoreIncludes += external/inc/uuid/1.38-cms/include
else 
   ifeq ($(shell uname), Darwin)
      BoostLibs := boost_system-xgcc40-mt-1_38 boost_thread-xgcc40-mt-1_38  boost_signals-xgcc40-mt-1_38  boost_filesystem-xgcc40-mt-1_38  boost_program_options-xgcc40-mt-1_38 boost_regex-xgcc40-mt-1_38 
      RootLibs :=  RIO TreePlayer Gpad Graf3d Graf Hist Matrix Physics Postscript Cintex Cint Core Net Tree MathCore MathMore GenVector Reflex 
      CoreLibs :=  $(BoostLibs) $(RootLibs) CLHEP-1.9.4.2 gsl gslcblas HepMC sigc-2.0.0 dl

      CoreIncludes :=  external/inc/boost/1.38.0/include external/inc/clhep/1.9.4.2/include 
      CoreIncludes +=  external/inc/gsl/1.10/include external/inc/hepmc/2.03.06/include
      CoreIncludes += external/inc/sigcpp/2.0.17/include/sigc++-2.0
   endif     
endif
