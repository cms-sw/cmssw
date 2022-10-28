#include <cstdio>

namespace Matriplex {
  void align_check(const char *pref, void *adr) {
    printf("%s 0x%llx  -  modulo 64 = %lld\n", pref, (long long unsigned)adr, (long long)adr % 64);
  }
}  // namespace Matriplex
