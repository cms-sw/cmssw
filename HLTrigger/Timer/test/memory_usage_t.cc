#include <array>
#include <iostream>
#include <memory>

#include "HLTrigger/Timer/interface/memory_usage.h"

void print_thread_header() {
  std::cout << "+allocated / -deallocated / < peak memory for the current thread:" << std::endl;
}

void print_thread_stat() {
  std::cout << "\t+" << (memory_usage::allocated() / 1024) << " kB / "  // allocated memory, in kB
            << "-" << (memory_usage::deallocated() / 1024) << " kB / "  // deallocated memory, in kB
            << "< " << (memory_usage::peak() / 1024) << " kB"           // peak used memory, in kB
            << std::endl;
}

int main(void) {
  constexpr int size = 10;
  std::array<std::unique_ptr<std::byte[]>, size> buffers;

  print_thread_header();
  print_thread_stat();

  for (auto& buffer : buffers) {
    buffer = std::make_unique<std::byte[]>(16 * 1024);
    print_thread_stat();
  }

  for (auto& buffer : buffers) {
    buffer.reset();
    print_thread_stat();
  }

  std::cout << std::endl;
  memory_usage::reset_peak();
  print_thread_header();
  print_thread_stat();

  for (auto& buffer : buffers) {
    buffer = std::make_unique<std::byte[]>(16 * 1024);
    print_thread_stat();
    buffer.reset();
    print_thread_stat();
  }

  return 0;
}
