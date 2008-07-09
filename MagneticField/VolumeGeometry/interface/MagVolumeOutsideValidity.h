#ifndef MagVolumeOutsideValidity_H
#define MagVolumeOutsideValidity_H

#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include <exception> 

class MagVolumeOutsideValidity : public std::exception {
public:

  MagVolumeOutsideValidity( MagVolume::LocalPoint l,
			    MagVolume::LocalPoint u) throw() :
    lower_(l), upper_(u) {}

  MagVolume::LocalPoint lower() const  throw() {return lower_;} 
  MagVolume::LocalPoint upper() const  throw() {return upper_;} 

  virtual ~MagVolumeOutsideValidity() throw() {}

  virtual const char* what() const throw() { return "Magnetic field requested outside of validity of the MagVolume";}

private:

  MagVolume::LocalPoint lower_;
  MagVolume::LocalPoint upper_;
};
#endif
