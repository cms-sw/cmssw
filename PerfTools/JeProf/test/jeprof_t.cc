#include "PerfTools/JeProf/interface/jeprof.h"
#include <string>

int main() {
  setenv("MALLOC_CONF","prof_leak:true,lg_prof_sample:10,prof_final:true",1);
  std::string name("heap.dump");
  const char *fileName = name.c_str();
  cms::jeprof::makeHeapDump(fileName);
}
