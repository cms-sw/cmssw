// -*- C++ -*-
//
// Package:    RecoLocalCalo/HcalRecAlgos
// Class:      HcalPulseShapesEP
//
/**\class HcalPulseShapesEP

 Description: Builds channel-dependent Hcal pulse shapes

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Wed, 17 Sep 2025 12:52:52 GMT
//
//

// system include files
#include <memory>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <utility>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

// Need to add #include statements for definitions of
// the data type and record type here
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapeLookup.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalPulseShapeLookupRcd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "CondFormats/HcalObjects/interface/HcalPulseDelays.h"
#include "CondFormats/HcalObjects/interface/HcalInterpolatedPulseMap.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

static HcalPulseShapeLookup::Shape makeShiftedPulse(const HcalInterpolatedPulse& pulse,
                                                    const double dt,
                                                    const unsigned len) {
  std::vector<double> values(len);
  for (unsigned i = 0; i < len; ++i)
    values[i] = pulse(i - dt);

  // Do we need to normalize this shape in some way?
  return HcalPulseShapeLookup::Shape(values, len);
}

//
// class declaration
//
class HcalPulseShapesEP : public edm::ESProducer {
public:
  HcalPulseShapesEP(const edm::ParameterSet&);
  ~HcalPulseShapesEP() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  typedef std::unique_ptr<HcalPulseShapeLookup> ReturnType;

  ReturnType produce(const HcalPulseShapeLookupRcd&);

private:
  typedef HcalPulseShapeLookup::LabeledShape LabeledShape;

  // ----------member data ---------------------------
  std::string pulseDumpFile_;
  double globalTimeShift_;
  unsigned pulseShapeLength_;
  unsigned dumpPrecision_;

  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topoToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESGetToken<HcalPulseDelays, HcalPulseDelaysRcd> delaysToken_;
  edm::ESGetToken<HcalInterpolatedPulseMap, HcalInterpolatedPulseMapRcd> pulseMapToken_;

  unsigned callCount_;
};

//
// constructors and destructor
//
HcalPulseShapesEP::HcalPulseShapesEP(const edm::ParameterSet& iConfig)
    : pulseDumpFile_(iConfig.getUntrackedParameter<std::string>("pulseDumpFile", "")),
      globalTimeShift_(iConfig.getParameter<double>("globalTimeShift")),
      pulseShapeLength_(iConfig.getParameter<unsigned>("pulseShapeLength")),
      dumpPrecision_(iConfig.getUntrackedParameter<unsigned>("dumpPrecision", 0U)),
      callCount_(0U) {
  //the following line is needed to tell the framework what
  // data is being produced
  auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("productLabel"));
  topoToken_ = cc.consumes();
  geomToken_ = cc.consumes();
  delaysToken_ = cc.consumes();
  pulseMapToken_ = cc.consumes();

  //now do what ever other initialization is needed
}

HcalPulseShapesEP::~HcalPulseShapesEP() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
HcalPulseShapesEP::ReturnType HcalPulseShapesEP::produce(const HcalPulseShapeLookupRcd& rcd) {
  const HcalTopology& htopo = rcd.get(topoToken_);
  const CaloGeometry& geom = rcd.get(geomToken_);
  const HcalPulseDelays& delays = rcd.get(delaysToken_);
  const HcalInterpolatedPulseMap& pulses = rcd.get(pulseMapToken_);

  // We need to create a LabeledShape object for each unique
  // combination of pulse shape label and pulse shape delay
  std::vector<LabeledShape> shapes;
  typedef std::pair<std::string, float> UniqueKey;
  std::map<UniqueKey, int> shapeNumberLookup;

  // The lookup table from the linearized channel numbers
  // into the pulse shapes. Initialize to an invalid value.
  std::vector<int> shapeTypes(htopo.ncells(), -1);

  // Prepare to cycle over Hcal subdetectors
  constexpr unsigned nSubDet = 2U;
  const HcalSubdetector subdetectors[nSubDet] = {HcalBarrel, HcalEndcap};
  const std::string subDetNames[nSubDet] = {"HB", "HE"};

  // Cycle over Hcal subdetectors
  for (unsigned isub = 0; isub < nSubDet; ++isub) {
    const HcalGeometry* hcalGeom =
        static_cast<const HcalGeometry*>(geom.getSubdetectorGeometry(DetId::Hcal, subdetectors[isub]));
    const std::vector<DetId>& ids = hcalGeom->getValidDetIds(DetId::Hcal, subdetectors[isub]);

    // Cycle over channels for this subdetector
    for (const DetId& id : ids) {
      const unsigned denseId = htopo.detId2denseId(id);
      const HcalPulseDelay* delay = delays.getValues(id, true);
      const float dt = globalTimeShift_ + delay->delay();
      const UniqueKey pulseKey(delay->label(), dt);
      const std::map<UniqueKey, int>::const_iterator it = shapeNumberLookup.find(pulseKey);
      if (it == shapeNumberLookup.end()) {
        // Construct the pulse shape for this delay value
        const HcalInterpolatedPulse& pulse = pulses.get(pulseKey.first);
        const int newShapeNumber = shapes.size();
        shapes.emplace_back(pulseKey.first, dt, makeShiftedPulse(pulse, dt, pulseShapeLength_));
        shapeNumberLookup[pulseKey] = newShapeNumber;
        shapeTypes.at(denseId) = newShapeNumber;
      } else
        shapeTypes.at(denseId) = it->second;
    }
  }

  // Create the product
  auto product = std::make_unique<HcalPulseShapeLookup>(shapes, shapeTypes, &htopo);

  // Dump the pulse shapes if requested
  if (!pulseDumpFile_.empty()) {
    std::ostringstream os;
    os << pulseDumpFile_ << '.' << callCount_;
    product->dumpToTxt(os.str(), dumpPrecision_);
  }

  ++callCount_;
  return product;
}

void HcalPulseShapesEP::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("productLabel", "HcalDataShapes");
  desc.add<unsigned>("pulseShapeLength", 250U);
  desc.add<double>("globalTimeShift", 0.0);
  desc.addUntracked<std::string>("pulseDumpFile", "");
  desc.addUntracked<unsigned>("dumpPrecision", 0U);
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalPulseShapesEP);
