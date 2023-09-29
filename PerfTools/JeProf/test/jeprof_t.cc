#include "PerfTools/JeProf/interface/jeprof.h"
#include <string>

int main() {
  std::string name("heap.dump");
  const char *fileName = name.c_str();
  cms::jeprof::makeHeapDump(fileName);
}
