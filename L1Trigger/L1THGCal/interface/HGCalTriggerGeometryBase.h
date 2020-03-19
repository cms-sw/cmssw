#ifndef __L1Trigger_L1THGCal_HGCalTriggerGeometryBase_h__
#define __L1Trigger_L1THGCal_HGCalTriggerGeometryBase_h__

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

// Pure virtual trigger geometry class
// Provides the interface to access trigger cell and module mappings
class HGCalTriggerGeometryBase {
public:
  typedef std::unordered_map<unsigned, unsigned> geom_map;
  typedef std::unordered_set<unsigned> geom_set;
  typedef std::set<unsigned> geom_ordered_set;

  HGCalTriggerGeometryBase(const edm::ParameterSet& conf);
  virtual ~HGCalTriggerGeometryBase() {}

  const std::string& name() const { return name_; }

  bool isV9Geometry() const { return !calo_geometry_; }
  bool isWithNoseGeometry() const { return isNose_; }

  const CaloGeometry* caloGeometry() const { return calo_geometry_; }

  const HGCalGeometry* noseGeometry() const {
    return (hgc_nose_geometry_
                ? hgc_nose_geometry_
                : (static_cast<const HGCalGeometry*>(calo_geometry_->getSubdetectorGeometry(DetId::Forward, HFNose))));
  }

  const HGCalGeometry* eeGeometry() const {
    return (hgc_ee_geometry_
                ? hgc_ee_geometry_
                : (static_cast<const HGCalGeometry*>(calo_geometry_->getSubdetectorGeometry(DetId::Forward, HGCEE))));
  }
  const HGCalGeometry* fhGeometry() const {
    return (hgc_hsi_geometry_
                ? hgc_hsi_geometry_
                : (static_cast<const HGCalGeometry*>(calo_geometry_->getSubdetectorGeometry(DetId::Forward, HGCHEF))));
  }
  const HcalGeometry* bhGeometry() const {
    if (hgc_hsc_geometry_) {
      throw cms::Exception("HGCalTriggerGeometry") << "bhGeometry cannot be used with the V9 geometry";
    }
    return (static_cast<const HcalGeometry*>(calo_geometry_->getSubdetectorGeometry(DetId::Hcal, HcalEndcap)));
  }
  const HGCalGeometry* hsiGeometry() const { return fhGeometry(); }
  const HGCalGeometry* hscGeometry() const {
    if (!hgc_hsc_geometry_) {
      throw cms::Exception("HGCalTriggerGeometry") << "hscGeometry cannot be used with the V7 and V8 geometries";
    }
    return hgc_hsc_geometry_;
  }
  const HGCalTopology& noseTopology() const { return noseGeometry()->topology(); }
  const HGCalTopology& eeTopology() const { return eeGeometry()->topology(); }
  const HGCalTopology& fhTopology() const { return fhGeometry()->topology(); }
  const HcalTopology& bhTopology() const { return bhGeometry()->topology(); }
  const HGCalTopology& hsiTopology() const { return hsiGeometry()->topology(); }
  const HGCalTopology& hscTopology() const { return hscGeometry()->topology(); }

  void setWithNoseGeometry(const bool isNose) { isNose_ = isNose; }

  // non-const access to the geometry class
  virtual void initialize(const CaloGeometry*) = 0;
  virtual void initialize(const HGCalGeometry*, const HGCalGeometry*, const HGCalGeometry*) = 0;
  virtual void initialize(const HGCalGeometry*, const HGCalGeometry*, const HGCalGeometry*, const HGCalGeometry*) = 0;
  virtual void reset();

  // const access to the geometry class
  virtual unsigned getTriggerCellFromCell(const unsigned cell_det_id) const = 0;
  virtual unsigned getModuleFromCell(const unsigned cell_det_id) const = 0;
  virtual unsigned getModuleFromTriggerCell(const unsigned trigger_cell_det_id) const = 0;

  virtual geom_set getCellsFromTriggerCell(const unsigned cell_det_id) const = 0;
  virtual geom_set getCellsFromModule(const unsigned cell_det_id) const = 0;
  virtual geom_set getTriggerCellsFromModule(const unsigned trigger_cell_det_id) const = 0;

  virtual geom_ordered_set getOrderedCellsFromModule(const unsigned cell_det_id) const = 0;
  virtual geom_ordered_set getOrderedTriggerCellsFromModule(const unsigned trigger_cell_det_id) const = 0;

  virtual geom_set getNeighborsFromTriggerCell(const unsigned trigger_cell_det_id) const = 0;

  virtual unsigned getLinksInModule(const unsigned module_id) const = 0;
  virtual unsigned getModuleSize(const unsigned module_id) const = 0;

  virtual GlobalPoint getTriggerCellPosition(const unsigned trigger_cell_det_id) const = 0;
  virtual GlobalPoint getModulePosition(const unsigned module_det_id) const = 0;

  virtual bool validTriggerCell(const unsigned trigger_cell_id) const = 0;
  virtual bool disconnectedModule(const unsigned module_id) const = 0;
  virtual unsigned lastTriggerLayer() const = 0;
  virtual unsigned triggerLayer(const unsigned id) const = 0;

protected:
  void setCaloGeometry(const CaloGeometry* geom) { calo_geometry_ = geom; }
  void setEEGeometry(const HGCalGeometry* geom) { hgc_ee_geometry_ = geom; }
  void setHSiGeometry(const HGCalGeometry* geom) { hgc_hsi_geometry_ = geom; }
  void setHScGeometry(const HGCalGeometry* geom) { hgc_hsc_geometry_ = geom; }
  void setNoseGeometry(const HGCalGeometry* geom) { hgc_nose_geometry_ = geom; }

private:
  const std::string name_;

  bool isNose_ = false;
  const CaloGeometry* calo_geometry_ = nullptr;
  const HGCalGeometry* hgc_ee_geometry_ = nullptr;
  const HGCalGeometry* hgc_hsi_geometry_ = nullptr;
  const HGCalGeometry* hgc_hsc_geometry_ = nullptr;
  const HGCalGeometry* hgc_nose_geometry_ = nullptr;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<HGCalTriggerGeometryBase*(const edm::ParameterSet&)> HGCalTriggerGeometryFactory;

#endif
