#include <Pythia.h>

class EmissionVetoHook : public Pythia8::UserHooks {

public:

  // Constructor and destructor.
  EmissionVetoHook(int argVerbosity, bool argCheckHard) : last(-1), nISRveto(0), nFSRveto(0),
    Verbosity(argVerbosity), CheckHard(argCheckHard), firstNoRad(true) { }
 ~EmissionVetoHook() {
    cout << "Number of ISR vetoed = " << nISRveto << endl;
    cout << "Number of FSR vetoed = " << nFSRveto << endl;
  }

  // Use VetoMPIStep to analyse the incoming LHEF event and
  // extract the veto scale
  bool canVetoMPIStep()    { return true; }
  int  numberVetoMPIStep() { return 1; }
  bool doVetoMPIStep(int, const Pythia8::Event &e);

  // For subsequent ISR/FSR emissions, find the pT of the shower
  // emission and veto as necessary
  bool canVetoISREmission() { return true; }
  bool doVetoISREmission(int, const Pythia8::Event &e, int);

  bool canVetoFSREmission() { return true; }
  bool doVetoFSREmission(int, const Pythia8::Event &e, int, bool=false);

  void fatalEmissionVeto(string message);

  // Functions to return information
  double getPTpowheg() { return pTpowheg; }
  double getPTshower() { return pTshower; }
  int    getNISRveto() { return nISRveto; }
  int    getNFSRveto() { return nFSRveto; }
  bool   getNoRad()    { return noRad;    }

private:
      
  double pTveto, pTpowheg, pTshower;
  int    last, nISRveto, nFSRveto, Verbosity;
  bool   CheckHard, noRad, firstNoRad;
};
