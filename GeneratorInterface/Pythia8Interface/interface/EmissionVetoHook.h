#include <Pythia.h>

class EmissionVetoHook : public Pythia8::UserHooks {

public:

   // Constructor and destructor.
   EmissionVetoHook() : firstNoRad(true) { }
  ~EmissionVetoHook() { }

  // Use VetoMIStep to analyse the incoming LHEF event and
  // extract the veto scale
  bool canVetoMIStep()    { return true; }
  int  numberVetoMIStep() { return 1; }
  bool doVetoMIStep(int, const Pythia8::Event &e);

  // For subsequent ISR/FSR emissions, find the pT of the shower
  // emission and veto as necessary
  bool canVetoISREmission() { return true; }
  bool doVetoISREmission(int, const Pythia8::Event &e);

  bool canVetoFSREmission() { return true; }
  bool doVetoFSREmission(int, const Pythia8::Event &e);

  // Functions to return information
  double getPTpowheg() { return pTpowheg; }
  double getPTshower() { return pTshower; }
  int    getNISRveto() { return nISRveto; }
  int    getNFSRveto() { return nFSRveto; }
  bool   getNoRad()    { return noRad;    }

private:
      
  double pTveto, pTpowheg, pTshower;
  int    nISRveto, nFSRveto;
  bool   noRad, firstNoRad;
};

