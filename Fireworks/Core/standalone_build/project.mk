ifeq ($(shell uname), Linux)
   BoostLibs := boost_thread  boost_signals  boost_filesystem boost_program_options boost_regex
   RootLibs  := Cint Core  Net Tree Hist Matrix MathCore MathMore Reflex EG Eve FTGL Ged Geom GeomPainter Gpad Graf Graf3d Gui Physics RGL Rint TreePlayer RIO GuiHtml
   ProjectLibs  := $(BoostLibs) $(RootLibs) CLHEP-2.0.4.2 gsl gslcblas HepMC pcre sigc-2.0 nsl crypt dl uuid

   ProjectIncludes := external/inc/boost/current/include
   ProjectIncludes += external/inc/clhep/2.0.4.2-cms/include
   ProjectIncludes += external/inc/gsl/current/include
   ProjectIncludes += external/inc/hepmc/current/include
   ProjectIncludes += external/inc/pcre/4.4/include
   ProjectIncludes += external/inc/sigcpp/2.2.3/include/sigc++-2.0
   ProjectIncludes += external/inc/uuid/1.38/include
 else
   ifeq ($(shell uname), Darwin)
      BoostLibs := boost_thread  boost_iostreams boost_signals  boost_filesystem boost_system boost_program_options boost_regex
      RootLibs  := Cint Core  Net Tree Hist Matrix MathCore MathMore Reflex EG Eve FTGL Ged Geom GeomPainter Gpad Graf Graf3d Gui Physics RGL Rint TreePlayer RIO GuiHtml
      ProjectLibs :=  $(BoostLibs) $(RootLibs) CLHEP-2.0.4.2 gsl gslcblas HepMC sigc-2.0.0 dl

      ProjectIncludes := external/inc/boost/1.40.0/include
      ProjectIncludes += external/inc/clhep/2.0.4.2/include
      ProjectIncludes += external/inc/gsl/1.10/include
      ProjectIncludes += external/inc/hepmc/2.03.06/include
      ProjectIncludes += external/inc/sigcpp/2.2.3/include/sigc++-2.0
   endif
endif
