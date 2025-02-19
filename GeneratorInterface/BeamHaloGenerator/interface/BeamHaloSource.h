#ifndef BeamHalo_Source_h
#define BeamHalo_Source_h

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include <string>
#include "HepMC/GenEvent.h"

#include "CLHEP/Random/RandFlat.h"

/*
namespace CLHEP
{
  class RandFlat ;
  class HepRandomEngine;
}
*/

namespace edm
{
  class BeamHaloSource : public GeneratedInputSource {
  public:

    /// Constructor
    BeamHaloSource(const ParameterSet &, const InputSourceDescription &);
    /// Destructor
    virtual ~BeamHaloSource();

  private:
	bool call_ki_bhg_init(long& seed);
        bool call_bh_set_parameters(int* ival, float* fval,const std::string cval_string);
	bool call_ki_bhg_fill(int& iret, float& weight);
	bool call_ki_bhg_stat(int& iret);

  private:

    virtual bool produce(Event & e);
    void clear();

    HepMC::GenEvent  *evt;

        int GENMOD_;
	int LHC_B1_;
	int LHC_B2_;
	int IW_MUO_;
	int IW_HAD_;
	float EG_MIN_;
	float EG_MAX_;
	std::string G3FNAME_;
  };

}

#endif



