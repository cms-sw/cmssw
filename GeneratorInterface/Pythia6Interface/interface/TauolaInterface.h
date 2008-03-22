#ifndef GeneratorInterface_Pythia6Interface_TauolaInterface_h
#define GeneratorInterface_Pythia6Interface_TauolaInterface_h

/** \class TauolaInterface
 *
 * Interface to TAUOLA tau decay library
 *
 * Serge Slabospitsky (original Fortran implementation)
 *
 * Christian Veelken (port to C++)
 *  04/17/07
 *
 ***************************************/

namespace edm
{
  class TauolaInterface 
  {  
   public:

    // Constructor
    TauolaInterface();

    // Destructor
    ~TauolaInterface();

    // enable/disable polarization effects in tauola
    void enablePolarizationEffects() { keypol_ = 1; }
    void disablePolarizationEffects() { keypol_ = 0; }

    // initialization of TAUOLA package;
    // to be called **before** the event loop
    void initialize();

    // decay of tau leptons;
    // to be called **within** the event loop
    // (after PYTHIA has been called 
    //  to put the tau lepton in the event)
    void processEvent();

    // print information about TAUOLA package
    // and statistics about branching ratios
    void print();

  private:
    // set production vertex for tau decay products,
    // taking tau lifetime of c tau = 87 um into account
    // (per default, the production vertex of the particles produced in the tau decay
    //  is set to the tau production vertex by TAUOLA)
    void setDecayVertex(int numGenParticles_beforeTAUOLA, int numGenParticles_afterTAUOLA);

    // further decay of unstable hadrons produced in tau decay;
    // return value is the last index of particle resulting from tau decay
    // (including direct and indirect decay products)
    int decayUnstableHadrons(int numGenParticles_beforeTAUOLA, int numGenParticles_afterTAUOLA);

    // flag to switch between standard TAUOLA package/and TAUOLA package customized for CMS by Serge Slabospitsky 
    int version_;

    // flag to enable/disable polarization effects in TAUOLA
    // (1 = angular distribution of decay products takes polarization of tau lepton into account
    //  0 = assume all tau leptons are unpolarization)
    int keypol_;
    
    // int switch_photos_ ;

    // maximum number of entries in PYJETS common block
    // (current CMS default = 4000,
    //  but might be increased to 10000 in the future)
    static const int maxNumberParticles = 4000;

    // flag to enable/disable debug output
    static int debug_;
  };
} 

#endif


