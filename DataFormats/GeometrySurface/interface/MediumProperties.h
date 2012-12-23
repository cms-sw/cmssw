#ifndef _Surface_MEDIUMPROPERTIES_H_
#define _Surface_MEDIUMPROPERTIES_H_


/** Constants describing material effects for a surface (for an
 * angle of incidence = pi/2).
 * If thickness = d:
 *   radLen = d/X0 (used for description of multiple scattering
 *                  and energy loss by electrons)
 *   xi = d[g/cm2] * 0.307075[MeV/(g/cm2)] * Z/A * 1/2
 *                 (used for energy loss acc. to Bethe-Bloch)
 */
class MediumProperties {
public:
  MediumProperties() : theRadLen(0), theXi(0) {}
  MediumProperties(float aRadLen, float aXi) :
    theRadLen(aRadLen), theXi(aXi) {}
  ~MediumProperties() {}
  
  /** Thickness in units of X0 (at normal incidence)
   */
  float radLen() const {
    return theRadLen;
  }
  /** Factor for Bethe-Bloch (at normal incidence;
   *  for definition see above)
   */
  float xi() const {
    return theXi;
  }
  
  bool isValid() const { return theRadLen!=0 || theXi!=0;}
  
private:
  float theRadLen;
  float theXi;
  
};
#endif
