#ifndef PhysicsTools_PatUtils_ParticleCode_h
#define PhysicsTools_PatUtils_ParticleCode_h

/**
   \file ParticleCode.h
   \brief Defines the enumerations of particle type and status.

   Inspired from the MrParticle ID flags.

   \author F.J. Ronga (ETH Zurich)
   \version $Id: ParticleCode.h,v 1.2 2008/02/14 10:50:43 fronga Exp $
**/


namespace pat {

  //! Definition of particle types
  enum ParticleType { 
    UNKNOWN = 0,  //!< 0: Unidentified isolated particle
    ELECTRON,     //!< 1:
    MUON,         //!< 2:
    TAU,          //!< 3:
    PHOTON,       //!< 4:
    JET,          //!< 5:
    BJET,         //!< 6:
    TOP,          //!< 7:
    INVISIBLE     //!< 8: Invisible particle (Monte Carlo only)
  };
                      

  //! Definition of particle status after selection
  enum ParticleStatus { 
    GOOD = 0,    //!< 0: Passed selection
    BAD,         //!< 1: Failed selection (without additional info)
    HOVERE,      //!< 2: Bad H/E ratio
    SHOWER,      //!< 3: Bad ECAL shower shape
    MATCHING     //!< 4: Bad matching to track
  };


}

#endif
