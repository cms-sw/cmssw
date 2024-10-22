#ifndef EDM_ML_DEBUG
#define EDM_ML_DEBUG
#endif
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edmtest {

  void sampleStandAlone() {
    LogDebug("cat_A");  //test stand alone declaration
    LogDebug("cat_A").log([](auto&& iLog) { iLog << " LogDebug.log was called"; });
    LogDebug("cat_A") << "LogDebug    was used to send cat_A";
    LogDebug("cat_B") << "LogDebug    was used to send cat_B";
    LogTrace("cat_A") << "LogTrace    was used to send cat_A";
    LogTrace("cat_B") << "LogTrace    was used to send cat_B";
    edm::LogInfo("cat_A") << "LogInfo     was used to send cat_A";
    edm::LogInfo("cat_B") << "LogInfo     was used to send cat_B";
    edm::LogVerbatim("cat_A") << "LogVerbatim was used to send cat_A";
    edm::LogVerbatim("cat_B") << "LogVerbatim was used to send cat_B";
    edm::LogFwkInfo("cat_A") << "LogFwkInfo  was used to send cat_A";
    edm::LogFwkInfo("cat_B") << "LogFwkInfo  was used to send cat_B";
    edm::LogFwkVerbatim("cat_A") << "LogFwkVerbatim was used to send cat_A";
    edm::LogFwkVerbatim("cat_B") << "LogFwkVerbatim was used to send cat_B";
    edm::LogWarning("cat_A") << "LogWarning  was used to send cat_A";
    edm::LogWarning("cat_B") << "LogWarning  was used to send cat_B";
    edm::LogPrint("cat_A") << "LogPrint    was used to send cat_A";
    edm::LogPrint("cat_B") << "LogPrint    was used to send cat_B";
    edm::LogError("cat_A") << "LogError    was used to send cat_A";
    edm::LogError("cat_B") << "LogError    was used to send cat_B";
    edm::LogProblem("cat_A") << "LogProblem  was used to send cat_A";
    edm::LogProblem("cat_B") << "LogProblem  was used to send cat_B";
  }

}  // namespace edmtest

int main() {
  edm::LogImportant("note") << "Default settings";
  edmtest::sampleStandAlone();

  edm::LogImportant("note") << "threshold DEBUG";
  edm::setStandAloneMessageThreshold(edm::messagelogger::ELdebug);
  edmtest::sampleStandAlone();

  edm::LogImportant("note") << "threshold INFO";
  edm::setStandAloneMessageThreshold(edm::messagelogger::ELinfo);
  edmtest::sampleStandAlone();

  edm::LogImportant("note") << "threshold FWKINFO";
  edm::setStandAloneMessageThreshold(edm::messagelogger::ELfwkInfo);
  edmtest::sampleStandAlone();

  edm::LogImportant("note") << "threshold WARNING";
  edm::setStandAloneMessageThreshold(edm::messagelogger::ELwarning);
  edmtest::sampleStandAlone();

  edm::LogImportant("note") << "threshold ELerror";  // ERROR would confuse
                                                     // the grep checking that
                                                     // the runtests worked
  edm::setStandAloneMessageThreshold(edm::messagelogger::ELerror);
  edmtest::sampleStandAlone();

  edm::LogImportant("note") << "squelch cat_A";
  edm::squelchStandAloneMessageCategory("cat_A");
  edmtest::sampleStandAlone();

  edm::LogImportant("note") << "squelch cat_B";
  edm::squelchStandAloneMessageCategory("cat_B");
  edmtest::sampleStandAlone();

  return 0;
}
