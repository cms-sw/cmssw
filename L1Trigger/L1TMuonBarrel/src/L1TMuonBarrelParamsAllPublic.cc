#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelParamsAllPublic.h"

/*const L1TMuonBarrelParamsAllPublic& cast_to_L1TMuonBarrelParamsAllPublic(const L1TMuonBarrelParams& a)
{
	assert(sizeof(L1TMuonBarrelParamsAllPublic) == sizeof(L1TMuonBarrelParams));
	const void * pa = &a;
	const L1TMuonBarrelParamsAllPublic * py = static_cast<const L1TMuonBarrelParamsAllPublic *>(pa);
	return *py;
}
*/
const L1TMuonBarrelParams& cast_to_L1TMuonBarrelParams(const L1TMuonBarrelParamsAllPublic& a) {
  assert(sizeof(L1TMuonBarrelParamsAllPublic) == sizeof(L1TMuonBarrelParams));
  const void* pa = &a;
  const L1TMuonBarrelParams* py = static_cast<const L1TMuonBarrelParams*>(pa);
  return *py;
}
