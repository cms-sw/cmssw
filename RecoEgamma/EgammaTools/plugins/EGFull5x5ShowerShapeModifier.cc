#include "PhysicsTools/PatAlgos/interface/ModifyObjectValueBase.h"

class EGFull5x5ShowerShapeModifierFromValueMaps : public ModifyObjectValueBase {
public:
  EGFull5x5ShowerShapeModifierFromValueMaps(const edm::ParameterSet& conf) :
    ModifyObjectValueBase(conf) {
  }
  
  void setEvent(const edm::Event&) override final;
  void setEventContent(const edm::EventSetup&) override final;
  void setConsumes(edm::ConsumesCollector&) override final;
  
  void modifyObject(pat::Electron&) const override final;
  void modifyObject(pat::Photon&) const override final;

private:
  
};

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGFull5x5ShowerShapeModifierFromValueMaps,
		  "EGFull5x5ShowerShapeModifierFromValueMaps");

void EGFull5x5ShowerShapeModifierFromValueMaps::
setEvent(const edm::Event&) {
}

void EGFull5x5ShowerShapeModifierFromValueMaps::
setEventContent(const edm::EventSetup&) {
}

void EGFull5x5ShowerShapeModifierFromValueMaps::
setConsumes(edm::ConsumesCollector&) {
}
  
void EGFull5x5ShowerShapeModifierFromValueMaps::
modifyObject(pat::Electron&) const {
}
 
void EGFull5x5ShowerShapeModifierFromValueMaps::
modifyObject(pat::Photon&) const {
}
