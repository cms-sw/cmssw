#ifndef FASTSIM_CONSTANTS_H
#define FASTSIM_CONSTANTS_H

///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////

namespace fastsim {
  //! Definition of constants needed for the SimplifiedGeometryPropagator package.
  namespace Constants {
    static double constexpr speedOfLight = 29.9792458;  //!< Speed of light [cm / ns]
    static double constexpr eMass = 0.0005109990615;    //!< Electron mass[GeV]
    static double constexpr muMass = 0.1056583745;      //!< Muon mass [GeV]
    static double constexpr epsilonDistance_ = 1.0e-7;  //!< some epsilon for numerical comparisons
    static double constexpr NA = 6.022e+23;             //!< Avogadro's number
  };                                                    // namespace Constants
}  // namespace fastsim

#endif
