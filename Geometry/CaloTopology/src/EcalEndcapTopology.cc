#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"

EEDetId EcalEndcapTopology::incrementIy(const EEDetId& id) const {
  if (!(theGeom_->present(id))) {
    return EEDetId(0);
  }
  EEDetId nextPoint;
  if (EEDetId::validDetId(id.ix(), id.iy() + 1, id.zside()))
    nextPoint = EEDetId(id.ix(), id.iy() + 1, id.zside());
  else
    return EEDetId(0);

  if (theGeom_->present(nextPoint))
    return nextPoint;
  else
    return EEDetId(0);
}

EEDetId EcalEndcapTopology::decrementIy(const EEDetId& id) const {
  if (!(theGeom_->present(id))) {
    return EEDetId(0);
  }
  EEDetId nextPoint;
  if (EEDetId::validDetId(id.ix(), id.iy() - 1, id.zside()))
    nextPoint = EEDetId(id.ix(), id.iy() - 1, id.zside());
  else
    return EEDetId(0);

  if (theGeom_->present(nextPoint))
    return nextPoint;
  else
    return EEDetId(0);
}

EEDetId EcalEndcapTopology::incrementIx(const EEDetId& id) const {
  if (!(theGeom_->present(id))) {
    return EEDetId(0);
  }

  EEDetId nextPoint;
  if (EEDetId::validDetId(id.ix() + 1, id.iy(), id.zside()))
    nextPoint = EEDetId(id.ix() + 1, id.iy(), id.zside());
  else
    return EEDetId(0);

  if (theGeom_->present(nextPoint))
    return nextPoint;
  else
    return EEDetId(0);
}

EEDetId EcalEndcapTopology::decrementIx(const EEDetId& id) const {
  if (!(theGeom_->present(id))) {
    return EEDetId(0);
  }

  EEDetId nextPoint;

  if (EEDetId::validDetId(id.ix() - 1, id.iy(), id.zside()))
    nextPoint = EEDetId(id.ix() - 1, id.iy(), id.zside());
  else
    return EEDetId(0);

  if (theGeom_->present(nextPoint))
    return nextPoint;
  else
    return EEDetId(0);
}
