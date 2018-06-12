#include "GeneratorInterface/Pythia8Interface/plugins/PowhegHooksBB4L.h"

class EmissionVetoHook1 : public Pythia8::UserHooks {

public:  

  // Constructor and destructor.
  EmissionVetoHook1(int nFinalIn, bool vetoOnIn, int vetoCountIn,
                    int pThardModeIn, int pTemtModeIn, int emittedModeIn,
                    int pTdefModeIn, bool MPIvetoOnIn, int QEDvetoModeIn,
                    int nFinalModeIn, int VerbosityIn) :
                    nFinalExt(nFinalIn),
                    vetoOn(vetoOnIn), vetoCount(vetoCountIn),
                    pThardMode(pThardModeIn), pTemtMode(pTemtModeIn),
                    emittedMode(emittedModeIn), pTdefMode(pTdefModeIn),
		    MPIvetoOn(MPIvetoOnIn), QEDvetoMode(QEDvetoModeIn),
		    nFinalMode(nFinalModeIn), nISRveto(0), nFSRveto(0),
                    Verbosity(VerbosityIn) {}
 ~EmissionVetoHook1() override {
    std::cout << "Number of ISR vetoed = " << nISRveto << std::endl;
    std::cout << "Number of FSR vetoed = " << nFSRveto << std::endl;
  }

//--------------------------------------------------------------------------

  bool canVetoMPIStep() override    { return true; }
  int  numberVetoMPIStep() override { return 1; }
  bool doVetoMPIStep(int nMPI, const Pythia8::Event &e) override;

  bool canVetoISREmission() override { return vetoOn; }
  bool doVetoISREmission(int, const Pythia8::Event &e, int iSys) override;

  bool canVetoFSREmission() override { return vetoOn; }
  bool doVetoFSREmission(int, const Pythia8::Event &e, int iSys, bool) override;

  bool canVetoMPIEmission() override { return MPIvetoOn; }
  bool doVetoMPIEmission(int, const Pythia8::Event &e) override;

  void fatalEmissionVeto(std::string message);

  double pTpythia(const Pythia8::Event &e, int RadAfterBranch,
                  int EmtAfterBranch, int RecAfterBranch, bool FSR);

  double pTpowheg(const Pythia8::Event &e, int i, int j, bool FSR);

  double pTcalc(const Pythia8::Event &e, int i, int j, int k, int r, int xSRin);

//--------------------------------------------------------------------------

private:
  int    nFinalExt, vetoOn, vetoCount, pThardMode, pTemtMode,
         emittedMode, pTdefMode, MPIvetoOn, QEDvetoMode, nFinalMode;      
  int    nFinal;
  double pThard, pTMPI;
  bool   accepted, isEmt;
  // The number of accepted emissions (in a row)
  int nAcceptSeq;
  // Statistics on vetos
  unsigned long int nISRveto, nFSRveto;
  int Verbosity;
};
