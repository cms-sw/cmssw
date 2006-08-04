#ifndef METReco_GenMET_h
#define METReco_GenMET_h

/** \class GenMET
 *
 * \short MET made from Generator level HEPMC particles
 *
 * GenMET represents MET made from HEPMC particles
 * Provide energy contributions from different particles
 * in addition to generic MET parameters
 *
 * \author    R. Cavanaugh, UFL (inspiration taken from F. Ratnikov)
 *
 ************************************************************/

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/SpecificGenMETData.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/GenMETFwd.h"

namespace reco
{
  class GenMET : public MET
    {
    public:
      /* Constructors*/
      GenMET() {}
      GenMET( SpecificGenMETData gen_data_, double sumet_, 
	       const LorentzVector& fP4, const Point& fVertex ) 
	: MET( sumet_, fP4, fVertex ), gen_data( gen_data_ ) {}
      /* Default destructor*/
      virtual ~GenMET() {}
      /** Returns energy of electromagnetic particles*/
      double emEnergy() const {return gen_data.m_EmEnergy;};
      /** Returns energy of hadronic particles*/
      double hadEnergy() const {return gen_data.m_HadEnergy;};
      /** Returns invisible energy*/
      double invisibleEnergy() const {return gen_data.m_InvisibleEnergy;};
      /** Returns other energy (undecayed Sigmas etc.)*/
      double auxiliaryEnergy() const {return gen_data.m_AuxiliaryEnergy;};
      // block accessors
      //const Specific& getSpecific () const {return gen_data;}
    private:
      virtual bool overlap( const Candidate & ) const;
      // Data members
      //Variables specific to to the GenMET class
      SpecificGenMETData gen_data;
    };
}
#endif
