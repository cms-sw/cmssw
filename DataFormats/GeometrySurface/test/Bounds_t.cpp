#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"

#include <benchmark/benchmark.h>
#include <utility>

namespace {
  const RectangularPlaneBounds bound(1., 1., 1.);

  const Local3DPoint a(10., 10., 10.);

  const Local3DPoint in(0., 0., 0.);
  const Local3DPoint in2(0.5, 0., 0.);

  const Local3DPoint outY(0., 3., 0.);
  const Local3DPoint out2(0., 1.5, 0.);

  const Local3DPoint outZ(0., 3., 3.);

  const LocalError err(0.1, 0.0, 0.1);
  const LocalError errB(1.0, 0.0, 1.0);

  void BM_InsideFar(benchmark::State& state) {
    bool ok{};
    for (auto _ : state) {
      ok = bound.inside(a);
      benchmark::DoNotOptimize(ok);
    }
    if (ok != false)
      state.SkipWithError("unexpected inside result");
  }
  BENCHMARK(BM_InsideFar);

  void BM_InsideOutZ(benchmark::State& state) {
    bool ok{};
    for (auto _ : state) {
      ok = bound.inside(outZ);
      benchmark::DoNotOptimize(ok);
    }
    if (ok != false)
      state.SkipWithError("unexpected inside result");
  }
  BENCHMARK(BM_InsideOutZ);

  void BM_InsideIn(benchmark::State& state) {
    bool ok{};
    for (auto _ : state) {
      ok = bound.inside(in);
      benchmark::DoNotOptimize(ok);
    }
    if (ok != true)
      state.SkipWithError("unexpected inside result");
  }
  BENCHMARK(BM_InsideIn);

  void BM_InsideInErr(benchmark::State& state) {
    bool ok{};
    for (auto _ : state) {
      ok = bound.inside(in, err);
      benchmark::DoNotOptimize(ok);
    }
    if (ok != true)
      state.SkipWithError("unexpected inside result");
  }
  BENCHMARK(BM_InsideInErr);

  void BM_InOutInErr(benchmark::State& state) {
    std::pair<bool, bool> io{};
    for (auto _ : state) {
      io = bound.inout(in, err);
      benchmark::DoNotOptimize(io);
    }
    if (io.first != true || io.second != false)
      state.SkipWithError("unexpected inout result");
  }
  BENCHMARK(BM_InOutInErr);

  void BM_InOutIn2ErrB(benchmark::State& state) {
    std::pair<bool, bool> io{};
    for (auto _ : state) {
      io = bound.inout(in2, errB);
      benchmark::DoNotOptimize(io);
    }
    if (io.first != true || io.second != true)
      state.SkipWithError("unexpected inout result");
  }
  BENCHMARK(BM_InOutIn2ErrB);

  void BM_InsideOutYErr(benchmark::State& state) {
    bool ok{};
    for (auto _ : state) {
      ok = bound.inside(outY, err);
      benchmark::DoNotOptimize(ok);
    }
    if (ok != false)
      state.SkipWithError("unexpected inside result");
  }
  BENCHMARK(BM_InsideOutYErr);

  void BM_InOutOutYErr(benchmark::State& state) {
    std::pair<bool, bool> io{};
    for (auto _ : state) {
      io = bound.inout(outY, err);
      benchmark::DoNotOptimize(io);
    }
    if (io.first != false || io.second != true)
      state.SkipWithError("unexpected inout result");
  }
  BENCHMARK(BM_InOutOutYErr);

  void BM_InOutOut2ErrB(benchmark::State& state) {
    std::pair<bool, bool> io{};
    for (auto _ : state) {
      io = bound.inout(out2, errB);
      benchmark::DoNotOptimize(io);
    }
    if (io.first != true || io.second != true)
      state.SkipWithError("unexpected inout result");
  }
  BENCHMARK(BM_InOutOut2ErrB);

  void BM_Inside2DIn(benchmark::State& state) {
    bool ok{};
    for (auto _ : state) {
      ok = bound.inside(Local2DPoint(in.x(), in.y()), 1.f);
      benchmark::DoNotOptimize(ok);
    }
    if (ok != true)
      state.SkipWithError("unexpected inside result");
  }
  BENCHMARK(BM_Inside2DIn);

  void BM_Inside2DOutY(benchmark::State& state) {
    bool ok{};
    for (auto _ : state) {
      ok = bound.inside(Local2DPoint(outY.x(), outY.y()), 1.f);
      benchmark::DoNotOptimize(ok);
    }
    if (ok != false)
      state.SkipWithError("unexpected inside result");
  }
  BENCHMARK(BM_Inside2DOutY);

  void BM_Inside2DOutYWide(benchmark::State& state) {
    bool ok{};
    for (auto _ : state) {
      ok = bound.inside(Local2DPoint(outY.x(), outY.y()), 10.f);
      benchmark::DoNotOptimize(ok);
    }
    if (ok != true)
      state.SkipWithError("unexpected inside result");
  }
  BENCHMARK(BM_Inside2DOutYWide);
}  // namespace

BENCHMARK_MAIN();
