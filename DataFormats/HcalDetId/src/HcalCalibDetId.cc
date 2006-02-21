#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"

const HcalCalibDetId HcalCalibDetId::Undefined(0x40000000u);

HcalCalibDetId::HcalCalibDetId() {
}


HcalCalibDetId::HcalCalibDetId(uint32_t rawid) : DetId(rawid) {
}

HcalCalibDetId::HcalCalibDetId(SectorId sector, int rbx, int channel) : DetId(Hcal,HcalCalibration) {
  id_|=(CalibrationBox<<20);
  id_|=(rbx&0x1F)|((sector&0xF)<<5)|((channel&0xF)<<9);
}

HcalCalibDetId::HcalCalibDetId(const DetId& gen) {
  if (gen.det()!=Hcal || gen.subdetId()!=HcalCalibration) {
    throw new std::exception();
  }
  id_=gen.rawId();
}

HcalCalibDetId& HcalCalibDetId::operator=(const DetId& gen) {
  if (gen.det()!=Hcal || gen.subdetId()!=HcalCalibration) {
    throw new std::exception();
  }
  id_=gen.rawId();
  return *this;
}

int HcalCalibDetId::rbx() const {
  return (calibFlavor()==CalibrationBox)?(id_&0x1F):(0);
}

HcalCalibDetId::SectorId HcalCalibDetId::sector() const {
  return (SectorId)((calibFlavor()==CalibrationBox)?((id_>>5)&0xF):(0));
}

std::string HcalCalibDetId::sectorString() const {
  switch (sector()) {
  case(HBplus): return "HBP";
  case(HBminus): return "HBM";
  case(HEplus): return "HEP";
  case(HEminus): return "HEM";
  case(HFplus): return "HFP";
  case(HFminus): return "HFM";
  case(HO2plus): return "HO2P";
  case(HO1plus): return "HO1P";
  case(HOzero): return "HO0";
  case(HO1minus): return "HO1M";
  case(HO2minus): return "HO2M";
  default : return "";
  }
}

int HcalCalibDetId::cboxChannel() const {
  return (calibFlavor()==CalibrationBox)?((id_>>9)&0xF):(0);
}

std::string HcalCalibDetId::cboxChannelString() const {
  switch (cboxChannel()) {
  case(cbox_MixerHigh): return "Mixer-High";
  case(cbox_MixerLow): return "Mixer-Low";
  case(cbox_LaserMegatile): return "Megatile";
  case(cbox_MixerScint): return "Mixer-Scintillator";
  case(cbox_RadDam1): return "RadDam1";
  case(cbox_RadDam2): return "RadDam2";
  case(cbox_RadDam3): return "RadDam3";
  default : return "";
  }
}

std::ostream& operator<<(std::ostream& s,const HcalCalibDetId& id) {
  switch (id.calibFlavor()) {
  case(HcalCalibDetId::CalibrationBox):
    return s << "(HcalCalibBox " << id.sectorString() << ':' << id.rbx() 
	     << ' ' << id.cboxChannelString() << ')';
  default: return s;
  };
}


