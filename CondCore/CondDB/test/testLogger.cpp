
#include "CondCore/CondDB/interface/Logger.h"
//
#include <iostream>

using namespace cond::persistency;

int main(int argc, char** argv) {
  Logger logger("TestO2O_gg_code");
  logger.start();
  std::string s("XYZ");
  logger.logInfo() << "Step #" << 1 << " and string is [" << s << "]";
  ::sleep(2);
  logger.logError() << "Step #" << 2;
  logger.logWarning() << "Step #" << 3;
  logger.log("SPECIAL") << "Blabla val=" << 77 << " and more stuff: " << std::string("ciao");
  logger.end(-1);
  logger.saveOnFile();
  return 0;
}
