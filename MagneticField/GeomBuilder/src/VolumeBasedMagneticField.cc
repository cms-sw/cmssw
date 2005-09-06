// #include "Utilities/Configuration/interface/Architecture.h"

#include "MagneticField/GeomBuilder/interface/VolumeBasedMagneticField.h"
#include "MagneticField/GeomBuilder/interface/MagGeometry.h"

// #include "Utilities/Notification/interface/Singleton.h"
#include "DetectorDescription/Base/interface/Singleton.h"
#include "CLHEP/Units/SystemOfUnits.h"

// #include "Utilities/Notification/interface/TimingReport.h"
// #include "Utilities/UI/interface/SimpleConfigurable.h"

// VolumeBasedMagneticField::VolumeBasedMagneticField(seal::Context* ic)  : 
// theGeometry(DDI::Singleton<MagGeometry>::instance()) {
VolumeBasedMagneticField::VolumeBasedMagneticField(seal::Context* ic) {

//   static SimpleConfigurable<bool> timerOn(false,"VolumeBasedMagneticField:timing");
//   (*TimingReport::current()).switchOn("VolumeBasedMagneticField::field",timerOn);
}

VolumeBasedMagneticField::~VolumeBasedMagneticField(){
//   DDI::Singleton<MagGeometry>::deleteInstance();
}

void VolumeBasedMagneticField::field(const double point[3], double Bfield[3]) {
//   static TimingReport::Item & timer= (*TimingReport::current())["VolumeBasedMagneticField::field"];
//   TimeMe t(timer,false);

  Surface::GlobalVector field = 
    theGeometry->fieldInTesla(Surface::GlobalPoint(point[0]/cm, point[1]/cm, point[2]/cm));
  Bfield[0] = field.x()*tesla;
  Bfield[1] = field.y()*tesla;
  Bfield[2] = field.z()*tesla;
}

