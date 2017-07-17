#ifndef BeamHalo_Producer_h
#define BeamHalo_Producer_h

#include <map>
#include <string>

#include "HepMC/GenEvent.h"

#include "CLHEP/Random/RandFlat.h"

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm
{
  class BeamHaloProducer : public one::EDProducer<EndRunProducer, one::WatchLuminosityBlocks, one::SharedResources> {
  public:

    /// Constructor
    BeamHaloProducer(const ParameterSet &);
    /// Destructor
    virtual ~BeamHaloProducer();

    void setRandomEngine(CLHEP::HepRandomEngine* v);

  private:
	bool call_ki_bhg_init(long& seed);
        bool call_bh_set_parameters(int* ival, float* fval,const std::string cval_string);
	bool call_ki_bhg_fill(int& iret, float& weight);
	bool call_ki_bhg_stat(int& iret);

  private:

    virtual void produce(Event & e, const EventSetup & es) override;
    virtual void endRunProduce(Run & r, const EventSetup & es) override;
    virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&) override;
    virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&) override { }

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

    bool isInitialized_;
  };

}

#endif
