#ifndef GeneratorInterface_AlpgenInterface_AlpgenHeader_h
#define GeneratorInterface_AlpgenInterface_AlpgenHeader_h

#include <string>
#include <vector>
#include <map>

/// A structure that can parse and represent the contents 
/// of an ALPGEN _unw.par run information file.
struct AlpgenHeader {
  enum Parameter {
    ih2		= 2,
    ebeam	= 3,
    ndns	= 4,
    iqopt	= 5,
    qfac	= 6,
    ickkw	= 7,
    ktfac	= 8,
    njets	= 10,
    ihvy	= 11,
    ihvy2	= 12,
    nw		= 13,
    nz		= 14,
    nh		= 15,
    nph		= 16,
    ptjmin	= 30,
    ptbmin	= 31,
    ptcmin	= 32,
    ptlmin	= 33,
    metmin	= 34,
    ptphmin	= 35,
    etajmax	= 40,
    etabmax	= 41,
    etacmax	= 42,
    etalmax	= 43,
    etaphmax	= 44,
    drjmin	= 50,
    drbmin	= 51,
    drcmin      = 52,
    drlmin	= 55,
    drphjmin	= 56,
    drphlmin	= 57,
    drphmin	= 58,
    mllmin	= 61,
    mllmax	= 62,
    iseed1	= 90,
    iseed2	= 91,
    itopprc	= 102,
    cluopt	= 160,
    iseed3	= 190,
    iseed4	= 191
  };
  
  /// Function to return the human-readable 
  /// names of ALPGEN parameters.
  static std::string parameterName(Parameter index);

  enum Masses {
    mc = 0, mb, mt, mw, mz, mh, MASS_MAX
  };

  /// A function to parse a std::<vector<std::string> containing
  /// a _unw.par ALPGEN file, and store it in the internal structure.
  bool parse(const std::vector<std::string>::const_iterator &begin,
	     const std::vector<std::string>::const_iterator &end);

  std::map<Parameter, double>	params;
  unsigned int			ihrd;
  double			xsec;
  double			xsecErr;
  double			nEvents;
  double			lumi;
  double			masses[MASS_MAX];
};

#include "GeneratorInterface/AlpgenInterface/interface/AlpgenCommonBlocks.h"

#endif // GeneratorInterface_AlpgenInterface_AlpgenHeader_h
