ifeq ($(shell uname), Linux)
   CoreLibs :=  boost_thread-gcc-mt boost_signals-gcc-mt boost_filesystem-gcc-mt boost_program_options-gcc-mt boost_regex-gcc-mt CLHEP-1.9.4.2  gsl gslcblas HepMCfio HepMC pcre Cintex Cint Core  Net Tree TreePlayer Gpad Graf3d Graf Hist Matrix Physics Postscript Hist Matrix MathCore MathMore GenVector Minuit Minuit2 Physics Reflex sigc-2.0 nsl crypt dl uuid
   CoreIncludes :=  external/inc/boost/1.38.0/include external/inc/clhep/1.9.4.2-cms2/include external/inc/gsl/1.10-cms/include external/inc/hepmc/2.03.06-cms3/include external/inc/pcre/4.4-cms/include external/inc/sigcpp/2.0.17-cms/include/sigc++-2.0 external/inc/uuid/1.38-cms/include external/inc/zlib/1.2.3-cms/include
else 
   ifeq ($(shell uname), Darwin)
      CoreLibs :=  boost_thread-gcc-mt boost_regex-gcc-mt boost_filesystem-gcc-mt CLHEP-1.9.3.2 gsl gslcblas HepMCfio HepMC TreePlayer Gpad Graf3d Graf Hist Matrix Physics Postscript Cintex Cint Core Net Tree MathCore MathMore GenVector Reflex sigc-2.0 dl
      CoreIncludes :=  external/inc/boost/1.38.0/inComclude external/inc/clhep/1.9.4.2-cms2/include external/inc/gsl/1.10/include external/inc/hepmc/2.03.06-cms3/include external/inc/pcre/4.4-cms/include external/inc/sigcpp/2.0.17-cms/include/sigc++-2.0 external/inc/uuid/1.38-cms/include external/inc/zlib/1.2.3-cms/include
   endif     
endif
