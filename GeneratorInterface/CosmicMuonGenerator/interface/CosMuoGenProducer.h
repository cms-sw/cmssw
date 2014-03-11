#ifndef CosMuoGenProducer_h
#define CosMuoGenProducer_h
//
// CosmicMuonProducer by droll (01/FEB/2006)
//
#include "HepMC/GenEvent.h"

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "GeneratorInterface/CosmicMuonGenerator/interface/CosmicMuonGenerator.h"

namespace edm
{
  class CosMuoGenProducer : public one::EDProducer<EndRunProducer> {
  public:
    CosMuoGenProducer(const ParameterSet& );
    virtual ~CosMuoGenProducer();

  private: 
    virtual void produce(Event & e, const EventSetup& es);
    
    virtual void endRunProduce(Run & r, const EventSetup & es);
    
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
    double       ZCTarget; // z-position of centre of target-cylinder which cosmics HAVE to hit [mm], default is Nominal Interaction Point
    bool         TrackerOnly; //if set to "true" detector with tracker-only setup is used, so no material or B-field outside is considerd
    bool         MultiMuon; //read in multi-muon events from file instead of generating single muon events
    std::string  MultiMuonFileName; //file containing multi muon events, to be read in
    int32_t      MultiMuonFileFirstEvent;
    int32_t      MultiMuonNmin;
    bool         TIFOnly_constant; //if set to "true" cosmics can also be generated below 2GeV with unphysical constant energy dependence
    bool         TIFOnly_linear; //if set to "true" cosmics can also be generated below 2GeV with unphysical linear energy dependence
    bool         MTCCHalf; //if set to "true" muons are sure to hit half of CMS important for MTCC, 
                           //still material and B-field of whole CMS is considered

    //Plug position (default = on shaft)
    double PlugVtx;
    double PlugVtz;

    //material densities in g/cm^3
    double VarRhoAir;
    double VarRhoWall;
    double VarRhoRock;
    double VarRhoClay;
    double VarRhoPlug;
    double ClayLayerWidth; //[mm]


    //For upgoing muon generation: Neutrino energy limits
    double MinEn;
    double MaxEn;
    double NuPrdAlt;

    bool AllMu; //Accepting All Muons regardeless of direction

    // external cross section and filter efficiency
    double       extCrossSect;
    double       extFilterEff;

    CosmicMuonGenerator* CosMuoGen;
    // the event format itself
    HepMC::GenEvent* fEvt;
    bool cmVerbosity_;
  };
} 

#endif
