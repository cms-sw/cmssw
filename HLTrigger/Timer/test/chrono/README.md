chrono
======

Timers based on different time sources, using the C++11 chrono interface.



Intel Time Stamp Counter
========================

Some details on the TSC found in Intel processors, related serialising instructions, 
and benchmarks, can be found at [http://download.intel.com/embedded/software/IA/324264.pdf]

The ratio of TSC clock to bus clock can be read from `MSR_PLATFORM_INFO[15:8]`, see 
Intel 64 and IA-32 Architectures Software Developer’s Manual, Vol. 3C 35-53 (pag. 2852)


Notes on chrono::duration
=========================

A duration has a representation and a tick period (precision).

```
  template <typename Rep, typename Period> class duration;
```

The representation is simply any arithmetic type, or an emulation of such a type. 
The representation stores a count of ticks. This count is the only data member stored in a 
duration. If the representation is floating point, it can store fractions of a tick to the 
precision of the representation. 

The tick period is represented by a ratio (std::ratio) and is encoded into the duration's type, 
not stored. The tick period only has an impact on the behavior of the duration when a conversion 
between different duration's is attempted. The tick period is completely ignored when simply 
doing arithmetic on durations of the same type.

Note that the tick period is defined at *compile time* as an `std:ratio<>`, so it cannot be used to 
optimally represent clocks with period known only at runtime (e.g. x86 TSC, OSX `mach_absolute_time`, 
Windows `QueryPerformanceCounter`).


Notes on `native_duration`
==========================

A `native_duration` has a representation and a tick period (precision).

```
  template <class Rep, class Period> class native_duration;
```

As in an `std::chrono::duration`, the representation is simply any arithmetic type, or an emulation 
of such a type, storing a count of ticks. This count is the only data member stored in a duration. 
If the representation is floating point, it can store fractions of a tick to the precision of the 
representation. 

The tick period is represented by an arbitrary class, responsible for converting any amount of 
"native" ticks into a standard duration, and vice versa.

Since the implementation requires some additions to the `std` namespace, `native_duration` and the
clocks using it are implemented in the `interface/native/` and `src/native/` subdirectories, and live
in the `native` namespace.


Precision of different representations
======================================

A single precision floating point number can represent all integers up to 2^24, or 16'777,216 (~16 millions).
It can store a time interval with nanosecond resolution up to ~16 ms.

A double precision floating point number can represent all integers up to 2^53, or 9,007,199,254,740,992 (~9e15). 
It can store a time interval with nanosecond resolution up to 9,000,000 seconds - slightly more than 100 days.

A signed long integer can represent all integers up to 2^63-1, or 9,223,372,036,854,775,807 (~9e18).
It can store a time interval with nanosecond resolution up to 9,000,000,000 seconds - almost 300 years.


Sample outputs
==============

Some sample outputs are available in the `doc/` directory.
