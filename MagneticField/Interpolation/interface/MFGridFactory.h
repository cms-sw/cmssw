#ifndef MFGridFactory_h
#define MFGridFactory_h

/** \class MFGridFactory
 *
 *  Factory for field interpolators using binary files.
 *
 *  $Date: $
 *  $Revision: $
 *  \author T. Todorov
 */

#include <string>
class MFGrid;
template <class T> class GloballyPositioned;

class MFGridFactory {
public:

  /// Build interpolator for a binary grid file
  static MFGrid* build(const std::string& name, const GloballyPositioned<float>& vol);

  /// Build a 2pi phi-symmetric interpolator for a binary grid file
  static MFGrid* build(const std::string& name, const GloballyPositioned<float>& vol,
		       double phiMin, double phiMax);

};

#endif
