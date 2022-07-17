#ifndef MFGridFactory_h
#define MFGridFactory_h

/** \class MFGridFactory
 *
 *  Factory for field interpolators using binary files.
 *
 *  \author T. Todorov
 */

#include <string>
class MFGrid;
template <class T>
class GloballyPositioned;

namespace magneticfield::interpolation {
  class binary_ifstream;
}

class MFGridFactory {
public:
  using binary_ifstream = magneticfield::interpolation::binary_ifstream;

  /// Build interpolator for a binary grid file
  static MFGrid* build(const std::string& name, const GloballyPositioned<float>& vol);
  static MFGrid* build(binary_ifstream& name, const GloballyPositioned<float>& vol);

  /// Build a 2pi phi-symmetric interpolator for a binary grid file
  static MFGrid* build(const std::string& name, const GloballyPositioned<float>& vol, double phiMin, double phiMax);
};

#endif
