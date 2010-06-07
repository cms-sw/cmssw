#include "stack_zGolden.C"
#include "stack_zmTrk.C"
#include "stack_zmTrkMu.C"
#include "stack_zmSta.C"
#include "stack_zmm1hlt.C"
#include "stack_zmm2hlt.C"
#include "stack_zmmNotIso.C"
#include "stack_zmm0neLess2p4.C"

void stack_all() {
  stack_zGolden();
  stack_zmTrk();
  stack_zmTrkMu();
  stack_zmSta();
  stack_zmm1hlt();
  stack_zmm2hlt();
  stack_zmmNotIso();
  stack_zmm0neLess2p4();
}
