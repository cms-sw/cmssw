#include "esTaskArenas.h"

namespace {
  bool& switchForDifferentArenas() {
    static bool s = false;
    return s;
  }

  tbb::task_arena makeESTaskArena() {
    if (switchForDifferentArenas()) {
      return tbb::task_arena{tbb::this_task_arena::max_concurrency()};
    }
    return tbb::task_arena{tbb::task_arena::attach()};
  }
}  // namespace

void edm::implementation_detail::useDifferentArenas() { switchForDifferentArenas() = true; }

tbb::task_arena& edm::mainTaskArena() {
  static tbb::task_arena s_arena{tbb::task_arena::attach()};
  return s_arena;
}

tbb::task_arena& edm::esTaskArena() {
  static tbb::task_arena s_arena{makeESTaskArena()};
  return s_arena;
}
