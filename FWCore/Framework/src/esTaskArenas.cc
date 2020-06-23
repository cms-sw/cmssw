#include "esTaskArenas.h"

tbb::task_arena& edm::mainTaskArena() {
  static tbb::task_arena s_arena{tbb::task_arena::attach()};
  return s_arena;
}

tbb::task_arena& edm::esTaskArena() {
  static tbb::task_arena s_arena{tbb::this_task_arena::max_concurrency()};
  return s_arena;
}
