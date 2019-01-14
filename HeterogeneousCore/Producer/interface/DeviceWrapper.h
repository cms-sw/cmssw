#ifndef HeterogeneousCore_Producer_DeviceWrapper_h
#define HeterogeneousCore_Producer_DeviceWrapper_h

namespace heterogeneous {
  template <typename T> struct Mapping;
}

#define DEFINE_DEVICE_WRAPPER(DEVICE, ENUM) \
  template <> \
  struct Mapping<DEVICE> { \
    template <typename ...Args> \
    static void beginStream(DEVICE& algo, Args&&... args) { algo.call_beginStream##DEVICE(std::forward<Args>(args)...); } \
    template <typename ...Args> \
    static bool acquire(DEVICE& algo, Args&&... args) { return algo.call_acquire##DEVICE(std::forward<Args>(args)...); } \
    template <typename ...Args> \
    static void produce(DEVICE& algo, Args&&... args) { algo.call_produce##DEVICE(std::forward<Args>(args)...); } \
    static constexpr HeterogeneousDevice deviceEnum = ENUM; \
  }

#endif
