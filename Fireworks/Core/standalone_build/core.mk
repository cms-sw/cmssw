ifeq ($(shell uname), Linux)
   BoostLibs := boost_thread  boost_signals  boost_filesystem boost_program_options boost_regex
   RootLibs  := Cintex Cint Core  Net Tree TreePlayer Gpad Graf3d Graf Hist Matrix Physics Postscript Hist Matrix MathCore MathMore GenVector Minuit Minuit2 Physics Reflex
   CoreLibs  := $(BoostLibs) $(RootLibs) CLHEP-2.0.4.2 gsl gslcblas HepMC pcre sigc-2.0 nsl crypt dl uuid

   CoreIncludes := external/inc/boost/current/include
   CoreIncludes += external/inc/clhep/2.0.4.2-cms/include
   CoreIncludes += external/inc/gsl/current/include
   CoreIncludes += external/inc/hepmc/current/include
   CoreIncludes += external/inc/pcre/4.4/include
   CoreIncludes += external/inc/sigcpp/2.2.3/include/sigc++-2.0
   CoreIncludes += external/inc/uuid/1.38/include
else
   ifeq ($(shell uname), Darwin)
      BoostLibs := boost_system boost_thread boost_signals  boost_filesystem  boost_program_options boost_regex
      RootLibs  := RIO Cintex Cint Core  Net Tree TreePlayer Gpad Graf3d Graf Hist Matrix Physics Postscript Hist Matrix MathCore MathMore GenVector Minuit Minuit2 Physics Reflex
      CoreLibs :=  $(BoostLibs) $(RootLibs) CLHEP-2.0.4.2 gsl gslcblas HepMC sigc-2.0.0 dl

      CoreIncludes := external/inc/boost/1.40.0/include
      CoreIncludes += external/inc/clhep/2.0.4.2/include
      CoreIncludes += external/inc/gsl/1.10/include
      CoreIncludes += external/inc/hepmc/2.03.06/include
      CoreIncludes += external/inc/sigcpp/2.2.3/include/sigc++-2.0
   endif
endif
