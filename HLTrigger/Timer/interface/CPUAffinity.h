#ifndef CPUAffinity_h
#define CPUAffinity_h

class CPUAffinity {
public:
  static bool isCpuBound();
  static bool bindToCurrentCpu();
  static int  currentCpu();
};

#endif // CPUAffinity_h
