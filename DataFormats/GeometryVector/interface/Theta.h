#ifndef GeometryVector_Geom_Theta_h
#define GeometryVector_Geom_Theta_h

namespace Geom {

  /** A class for polar angle represantation.
 *  So far only useful to differentiate from double, for
 * example in function overloading.
 */

  template <class T>
  class Theta {
  public:
    /// Default constructor does not initialise - just as double.
    Theta() {}

    /// Constructor from T, does not provide automatic conversion.
    explicit Theta(const T& val) : theValue(val) {}

    /// conversion operator makes transparent use possible.
    operator T() const { return theValue; }

    /// Explicit access to value in case implicit conversion not OK
    T value() const { return theValue; }

  private:
    T theValue;
  };

}  // namespace Geom
#endif
