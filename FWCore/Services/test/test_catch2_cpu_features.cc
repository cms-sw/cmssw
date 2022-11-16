#include <iostream>

#include "catch.hpp"

#include "cpu_features/cpu_features_macros.h"

#if defined(CPU_FEATURES_ARCH_X86)
#include "cpu_features/cpuinfo_x86.h"
#elif defined(CPU_FEATURES_ARCH_ARM)
#include "cpu_features/cpuinfo_arm.h"
#elif defined(CPU_FEATURES_ARCH_AARCH64)
#include "cpu_features/cpuinfo_aarch64.h"
#elif defined(CPU_FEATURES_ARCH_PPC)
#include "cpu_features/cpuinfo_ppc.h"
#endif

TEST_CASE("Test cpu_features library", "[cpu_features]") {
  using namespace cpu_features;

#if defined(CPU_FEATURES_ARCH_X86)
  const X86Info info = GetX86Info();
  std::cout << "arch:     "
            << "x86"
            << "\nbrand:    " << info.brand_string << "\nfamily:   " << info.family << "\nmodel:    " << info.model
            << "\nstepping: " << info.stepping
            << "\nuarch:    " << GetX86MicroarchitectureName(GetX86Microarchitecture(&info)) << std::endl;
#elif defined(CPU_FEATURES_ARCH_ARM)
  const ArmInfo info = GetArmInfo();
  std::cout << "arch          "
            << "ARM"
            << "\nimplementer:  " << info.implementer << "\narchitecture: " << info.architecture
            << "\nvariant:      " << info.variant << "\npart:         " << info.part
            << "\nrevision:     " << info.revision << std::endl;
#elif defined(CPU_FEATURES_ARCH_AARCH64)
  const Aarch64Info info = GetAarch64Info();
  std::cout << "arch:        "
            << "aarch64"
            << "\nimplementer: " << info.implementer << "\nvariant:     " << info.variant
            << "\npart:        " << info.part << "\nrevision:    " << info.revision << std::endl;
#elif defined(CPU_FEATURES_ARCH_PPC)
  const PPCPlatformStrings strings = GetPPCPlatformStrings();
  std::cout << "arch:              "
            << "ppc"
            << "\nplatform:          " << strings.platform << "\nmodel:             " << strings.model
            << "\nmachine:           " << strings.machine << "\ncpu:               " << strings.cpu
            << "\ninstruction:       " << strings.type.platform << "\nmicroarchitecture: " << strings.type.base_platform
            << std::endl;
#endif
}
