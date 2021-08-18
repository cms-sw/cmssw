#include "CondFormats/EcalObjects/interface/EcalTPGTPMode.h"
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalTPGTPMode::EcalTPGTPMode() {}

EcalTPGTPMode::~EcalTPGTPMode() {}

void EcalTPGTPMode::print(std::ostream& out) const {
  out << ">>> Trigger primitive mode:  " << std::endl;
  out << "    enable EE odd filter  " << EnableEEOddFilter << std::endl;
  out << "    enable EB odd filter  " << EnableEBOddFilter << std::endl;
  out << "    enable EE odd peak finder  " << EnableEEOddPeakFinder << std::endl;
  out << "    enable EB odd peak finder  " << EnableEBOddPeakFinder << std::endl;
  out << "    disable EE even peak finder  " << DisableEEEvenPeakFinder << std::endl;
  out << "    disable EB even peak finder  " << DisableEBEvenPeakFinder << std::endl;
  if (FenixEEStripOutput == 0)
    out << "    EE strip formatter output: even filter " << std::endl;
  if (FenixEEStripOutput == 1)
    out << "    EE strip formatter output: odd filter " << std::endl;
  if (FenixEEStripOutput == 2)
    out << "    EE strip formatter output: larger of odd and even " << std::endl;
  if (FenixEEStripOutput == 3)
    out << "    EE strip formatter output: odd + even " << std::endl;
  if (FenixEBStripOutput == 0)
    out << "    EB strip formatter output: even filter " << std::endl;
  if (FenixEBStripOutput == 1)
    out << "    EB strip formatter output: odd filter " << std::endl;
  if (FenixEBStripOutput == 2)
    out << "    EB strip formatter output: larger of odd and even " << std::endl;
  if (FenixEBStripOutput == 3)
    out << "    EB strip formatter output: odd + even " << std::endl;
  out << "    Flag EE odd>even strip  " << FenixEEStripInfobit2 << std::endl;
  out << "    Flag EB odd>even strip  " << FenixEBStripInfobit2 << std::endl;
  if (EBFenixTcpOutput == 0)
    out << "    EB tcp formatter output: even filter " << std::endl;
  if (EBFenixTcpOutput == 1)
    out << "    EB tcp formatter output: larger of odd and even " << std::endl;
  if (EBFenixTcpOutput == 2)
    out << "    EB tcp formatter output: even + odd " << std::endl;
  out << "    Flag EB odd>even TCP  " << EBFenixTcpInfobit1 << std::endl;
}