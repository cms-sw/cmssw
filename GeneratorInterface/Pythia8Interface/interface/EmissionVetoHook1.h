#include "Pythia.h"

class EmissionVetoHook1 : public Pythia8::UserHooks {

public:  

  // Constructor and destructor.
  EmissionVetoHook1(int nFinalIn, int vetoModeIn, int vetoCountIn,
                    int pThardModeIn, int pTemtModeIn, int emittedModeIn,
                    int pTdefModeIn, int MPIvetoModeIn) :
                    nFinal(nFinalIn),
                    vetoMode(vetoModeIn), vetoCount(vetoCountIn),
                    pThardMode(pThardModeIn), pTemtMode(pTemtModeIn),
                    emittedMode(emittedModeIn), pTdefMode(pTdefModeIn),
                    MPIvetoMode(MPIvetoModeIn), nISRveto(0), nFSRveto(0) {}
 ~EmissionVetoHook1() {
    cout << "Number of ISR vetoed = " << nISRveto << endl;
    cout << "Number of FSR vetoed = " << nFSRveto << endl;
  }

//--------------------------------------------------------------------------

  bool canVetoMPIStep()    { return true; }
  int  numberVetoMPIStep() { return 1; }
  bool doVetoMPIStep(int nMPI, const Pythia8::Event &e);

  bool canVetoISREmission() { return (vetoMode == 0) ? false : true; }
  bool doVetoISREmission(int, const Pythia8::Event &e, int iSys);

  bool canVetoFSREmission() { return (vetoMode == 0) ? false : true; }
  bool doVetoFSREmission(int, const Pythia8::Event &e, int iSys, bool);

  bool canVetoMPIEmission() { return (MPIvetoMode == 0) ? false : true; }
  bool doVetoMPIEmission(int, const Pythia8::Event &e);

  void fatalEmissionVeto(string message);  

  double pTpythia(const Pythia8::Event &e, int RadAfterBranch,
                  int EmtAfterBranch, int RecAfterBranch, bool FSR);

  double pTpowheg(const Pythia8::Event &e, int i, int j, bool FSR);

  double pTcalc(const Pythia8::Event &e, int i, int j, int k, int r, int xSRin);

//--------------------------------------------------------------------------

private:
  int    nFinal, vetoMode, vetoCount, pThardMode, pTemtMode,
         emittedMode, pTdefMode, MPIvetoMode;
  double pThard, pTMPI;
  bool   accepted;
  // The number of accepted emissions (in a row)
  int nAcceptSeq;
  // Statistics on vetos
  unsigned long int nISRveto, nFSRveto;

};
