#ifndef DDMaterial_h
#define DDMaterial_h

#include <iostream>
#include <vector>
#include <utility>
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDBase.h"

namespace DDI { class Material; }

//! DDMaterial is used to define and access material information
/**
    An object of this class is a reference-object and thus leightweighted.
    It is uniquely identified by its DDName. Further details concerning
    reference-objects can be found in the documentation of DDLogicalPart.
    
    A DDMaterial can recursively consist of compound materials (which are 
    DDMaterials as well). Materials consisting of compound materials are called
    \b mixtures. Mixtures are defined by their Materials which do not consist of compound materials are called
    \b elementary materials.

    To define an \b elementray material, use it like this:
    \code
    DDMaterial hydrogen("Hydrogen", 
                        double z=1, 
			double a=1.1*g/mole, 
			density=2*g/cm3);
    \endcode
    
    To define a mixture: 
    \code
    DDMaterial mixt("Mix", double density = 5*g/cm3);
    mixt.addMaterial(hydrogen,0.3);
    // code for additional compounds belonging to the mixture ...
    \endcode
    
    Note the usage DDUnits replacing CLHEP/SystemOfUnits
    to specify the units of the quantities
    making up a material.
*/
class DDMaterial : public DDBase<DDName,DDI::Material*>
{
  friend std::ostream & operator<<(std::ostream &, const DDMaterial &);
  
public:
  typedef std::vector<std::pair<DDMaterial,double> > FractionV;
  
  //! Creates a uninitialized reference-object (see DDLogicalPart documentation for details on reference objects)
  DDMaterial();
  
  //! Creates a initialized reference-object or a reference to an allready defined material.
  DDMaterial(const DDName & name);

  //! Constructor for construction of an \b elementary material
  DDMaterial(const DDName & name, double z, double a, double d);
  
  //! Constructor for \b mixtures
  DDMaterial(const DDName & name, double density);
  
  //! returns the number of compound materials or 0 for elementary materials        
  int noOfConstituents() const;
  
  //! returns the i-th compound material and its fraction-mass
  FractionV::value_type constituent(int i) const; 
  
  //! adds a material to the mixture proportional to its fraction-mass \a fm.
  int addMaterial(const DDMaterial & m, double fm); 
  
  //! returns the atomic mass
  double a() const; 
  
  //! retruns the atomic number
  double z() const; 
  
  //! returns the density
  double density() const;
};

std::ostream & operator<<(std::ostream &, const DDMaterial &);

#endif
