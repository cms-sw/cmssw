#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

#include <iostream>
#include <cassert>

using namespace cms::perftools;

namespace {
  class TestMonitor : public AllocMonitorBase {
  public:
    TestMonitor(size_t& iRequested, size_t& iTotal) : requested_(iRequested), total_(iTotal) {}

    void allocCalled(size_t iRequested, size_t iActual) {
      requested_ = iRequested;
      total_ += iActual;
    }

    void deallocCalled(size_t iActual) { total_ -= iActual; }

  private:
    size_t& requested_;
    size_t& total_;
  };
}  // namespace

int main() {
  size_t requested = 0;
  size_t total = 0;

  {
    auto monitor = AllocMonitorRegistry::instance().createAndRegisterMonitor<TestMonitor>(requested, total);
    if (requested != 0) {
      std::cout << "Memory requested during monitor creation";
      exit(1);
    }
    {
      [[maybe_unused]] auto i = std::make_unique<int>(1);
      if (requested != sizeof(int)) {
        std::cout << "int request size wrong, got " << requested << " expected " << sizeof(int);
        exit(1);
      }
    }
    if (total != 0) {
      std::cout << "int request not cleaned up";
      exit(1);
    }

    {
      [[maybe_unused]] auto i = new int[5];
      if (requested != sizeof(int) * 5) {
        std::cout << "int request size wrong, got " << requested << " expected " << sizeof(int) * 5;
        exit(1);
      }
      delete[] i;
    }
    if (total != 0) {
      std::cout << "int request not cleaned up";
      exit(1);
    }

    {
      [[maybe_unused]] auto i = new (std::align_val_t{512}) int;
      if (requested != sizeof(int)) {
        std::cout << "int request size wrong, got " << requested << " expected " << sizeof(int);
        exit(1);
      }
      delete i;
    }
    if (total != 0) {
      std::cout << "int request not cleaned up";
      exit(1);
    }

    {
      [[maybe_unused]] auto i = new (std::align_val_t{512}) int[5];
      if (requested != sizeof(int) * 5) {
        std::cout << "int request size wrong, got " << requested << " expected " << sizeof(int) * 5;
        exit(1);
      }
      delete[] i;
    }
    if (total != 0) {
      std::cout << "int request not cleaned up";
      exit(1);
    }

    {
      [[maybe_unused]] auto i = new (std::align_val_t{512}, std::nothrow) int;
      if (requested != sizeof(int)) {
        std::cout << "int request size wrong, got " << requested << " expected " << sizeof(int);
        exit(1);
      }
      delete i;
    }
    if (total != 0) {
      std::cout << "int request not cleaned up";
      exit(1);
    }

    {
      [[maybe_unused]] auto i = new (std::align_val_t{512}, std::nothrow) int[5];
      if (requested != sizeof(int) * 5) {
        std::cout << "int request size wrong, got " << requested << " expected " << sizeof(int) * 5;
        exit(1);
      }
      delete[] i;
    }
    if (total != 0) {
      std::cout << "int request not cleaned up";
      exit(1);
    }

    {
      [[maybe_unused]] auto i = new (std::nothrow) int;
      if (requested != sizeof(int)) {
        std::cout << "int request size wrong, got " << requested << " expected " << sizeof(int);
        exit(1);
      }
      delete i;
    }
    if (total != 0) {
      std::cout << "int request not cleaned up";
      exit(1);
    }

    {
      [[maybe_unused]] auto i = new (std::nothrow) int[5];
      if (requested != sizeof(int) * 5) {
        std::cout << "int request size wrong, got " << requested << " expected " << sizeof(int) * 5;
        exit(1);
      }
      delete[] i;
    }
    if (total != 0) {
      std::cout << "int request not cleaned up";
      exit(1);
    }

    {
      auto p = calloc(12, 1);
      assert(p != nullptr);
      {
        auto r = requested;
        if (r != 12) {
          std::cout << "calloc request size wrong, got " << r << " expected " << 12;
          exit(1);
        }
      }
      free(p);
      if (total != 0) {
        std::cout << "calloc request not cleaned up";
        exit(1);
      }

      p = malloc(50);
      assert(p != nullptr);
      {
        auto r = requested;
        if (r != 50) {
          std::cout << "malloc request size wrong, got " << r << " expected " << 50;
          exit(1);
        }
      }
      p = realloc(p, 100);
      assert(p != nullptr);
      auto r = requested;
      if (r != 100) {
        std::cout << "realloc request size wrong, got " << r << " expected " << 100;
        exit(1);
      }
      if (total < 100) {
        auto t = total;
        std::cout << "realloc request total too small " << t;
        exit(1);
      }

      free(p);
      if (total != 0) {
        auto t = total;
        std::cout << "free after realloc request not cleaned up, still have " << t;
        exit(1);
      }

      p = aligned_alloc(128, 128 * 3);
      if (requested != 128 * 3) {
        auto r = requested;
        std::cout << "aligned_alloc request size wrong, got " << r << " expected " << 128 * 3;
        exit(1);
      }
      free(p);
      if (total != 0) {
        std::cout << "aligned_alloc request not cleaned up";
        exit(1);
      }

      p = memalign(256, 24);
      if (requested != 24) {
        auto r = requested;
        std::cout << "memalign request size wrong, got " << r << " expected " << 24;
        exit(1);
      }
      free(p);
      if (total != 0) {
        std::cout << "memalign request not cleaned up";
        exit(1);
      }

      p = nullptr;
      auto ret = posix_memalign(&p, 128, 64);
      if (p == nullptr) {
        std::cout << "posix_memalign failed to allocate ";
        exit(1);
      }
      if (ret != 0) {
        std::cout << "posix_memalign returned failed valued " << ret;
        exit(1);
      }
      if (requested != 64) {
        auto r = requested;
        std::cout << "posix_memalign request size wrong, got " << r << " expected " << 64;
        exit(1);
      }
      free(p);
      if (total != 0) {
        std::cout << "posix_memalign request not cleaned up";
        exit(1);
      }
    }
    AllocMonitorRegistry::instance().deregisterMonitor(monitor);
  }
}
