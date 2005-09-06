#ifndef MFGridFactory_H
#define MFGridFactory_H

#include <string>
class MFGrid;
template <class T> class GloballyPositioned;

class MFGridFactory {
public:

  static MFGrid* build(const std::string& name, const GloballyPositioned<float>& vol);

  static MFGrid* build(const std::string& name, const GloballyPositioned<float>& vol,
		       double phiMin, double phiMax);

};

#endif
