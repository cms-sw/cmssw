#ifndef CosMuoGenProducer_h
#define CosMuoGenProducer_h
//
// CosmicMuonProducer by droll (01/FEB/2006)
//
#include "GeneratorInterface/CosmicMuonGenerator/interface/CosmicMuonProducer.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "HepMC/GenEvent.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
//#include "FWCore/Framework/interface/GeneratedInputSource.h"
//#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"


namespace edm
{
  class CosMuoGenProducer : public EDProducer{
  public:
    CosMuoGenProducer(const ParameterSet& );
    virtual ~CosMuoGenProducer();

  private: 
    virtual void produce(Event & e, const EventSetup& es);
    
    void clear();
    // define the configurable generator parameters
    int32_t      RanS; // seed of random number generator (from Framework)
    double       MinP; // min. P     [GeV]
    double       MinP_CMS; // min. P at CMS surface    [GeV]; default is MinP_CMS=MinP, thus no bias from access-shaft
    double       MaxP; // max. P     [GeV]
    double       MinT; // min. theta [deg]
    double       MaxT; // max. theta [deg]
    double       MinPh; // min. phi   [deg]
    double       MaxPh; // max. phi   [deg]
    double       MinS; // min. t0    [ns]
    double       MaxS; // max. t0    [ns]
    double       ELSF; // scale factor for energy loss
    double       RTarget; // Radius of target-cylinder which cosmics HAVE to hit [mm], default is CMS-dimensions
    double       ZTarget; // z-length of target-cylinder which cosmics HAVE to hit [mm], default is CMS-dimensions
    bool         TrackerOnly; //if set to "true" detector with tracker-only setup is used, so no material or B-field outside is considerd
    bool         TIFOnly_constant; //if set to "true" cosmics can also be generated below 2GeV with unphysical constant energy dependence
    bool         TIFOnly_linear; //if set to "true" cosmics can also be generated below 2GeV with unphysical linear energy dependence
    bool         MTCCHalf; //if set to "true" muons are sure to hit half of CMS important for MTCC, 
                           //still material and B-field of whole CMS is considered

    CosmicMuonProducer* CosMuoGen;
    // the event format itself
    HepMC::GenEvent* fEvt;
    bool cmVerbosity_;
  };
} 

#endif
