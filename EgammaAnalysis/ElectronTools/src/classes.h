#include "EgammaAnalysis/ElectronTools/interface/SimpleElectron.h"
#include "EgammaAnalysis/ElectronTools/interface/EpCombinationTool.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyCalibratorRun2.h"
#include "EgammaAnalysis/ElectronTools/interface/PhotonEnergyCalibratorRun2.h"

namespace EgammaAnalysis_ElectronTools {
  struct dictionaryfuffa {
    SimpleElectron fuffaElectron;
    EpCombinationTool fuffaElectronCombinator;
    ElectronEnergyCalibratorRun2 fuffaElectronCalibrator;
    PhotonEnergyCalibratorRun2 fuffaPhotonCalibrator;
  };
}  // namespace EgammaAnalysis_ElectronTools
