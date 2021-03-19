#include "CondFormats/EcalObjects/interface/EcalTPGTPMode.h"
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalTPGTPMode::EcalTPGTPMode() {}

EcalTPGTPMode::~EcalTPGTPMode() {}

void EcalTPGTPMode::Print() const {
  LogDebug("EcalTPGTPMode") << ">>> Trigger primitive mode:  ";
  LogDebug("EcalTPGTPMode") << "    enable EE odd filter  " << EnableEEOddFilter;
  LogDebug("EcalTPGTPMode") << "    enable EB odd filter  " << EnableEBOddFilter;
  LogDebug("EcalTPGTPMode") << "    enable EE odd peak finder  " << EnableEEOddPeakFinder;
  LogDebug("EcalTPGTPMode") << "    enable EB odd peak finder  " << EnableEBOddPeakFinder;
  LogDebug("EcalTPGTPMode") << "    disable EE even peak finder  " << DisableEEEvenPeakFinder;
  LogDebug("EcalTPGTPMode") << "    disable EB even peak finder  " << DisableEBEvenPeakFinder;
  if (FenixEEStripOutput == 0)
    LogDebug("EcalTPGTPMode") << "    EE strip formatter output: even filter ";
  if (FenixEEStripOutput == 1)
    LogDebug("EcalTPGTPMode") << "    EE strip formatter output: odd filter ";
  if (FenixEEStripOutput == 2)
    LogDebug("EcalTPGTPMode") << "    EE strip formatter output: larger of odd and even ";
  if (FenixEEStripOutput == 3)
    LogDebug("EcalTPGTPMode") << "    EE strip formatter output: odd + even ";
  if (FenixEBStripOutput == 0)
    LogDebug("EcalTPGTPMode") << "    EB strip formatter output: even filter ";
  if (FenixEBStripOutput == 1)
    LogDebug("EcalTPGTPMode") << "    EB strip formatter output: odd filter ";
  if (FenixEBStripOutput == 2)
    LogDebug("EcalTPGTPMode") << "    EB strip formatter output: larger of odd and even ";
  if (FenixEBStripOutput == 3)
    LogDebug("EcalTPGTPMode") << "    EB strip formatter output: odd + even ";
  LogDebug("EcalTPGTPMode") << "    Flag EE odd>even strip  " << FenixEEStripInfobit2;
  LogDebug("EcalTPGTPMode") << "    Flag EB odd>even strip  " << FenixEBStripInfobit2;
  if (EBFenixTcpOutput == 0)
    LogDebug("EcalTPGTPMode") << "    EB tcp formatter output: even filter ";
  if (EBFenixTcpOutput == 1)
    LogDebug("EcalTPGTPMode") << "    EB tcp formatter output: larger of odd and even ";
  if (EBFenixTcpOutput == 2)
    LogDebug("EcalTPGTPMode") << "    EB tcp formatter output: even + odd ";
  LogDebug("EcalTPGTPMode") << "    Flag EB odd>even TCP  " << EBFenixTcpInfobit1;
}