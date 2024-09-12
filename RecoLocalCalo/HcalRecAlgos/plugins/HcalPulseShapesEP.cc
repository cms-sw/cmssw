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
//         Created:  Fri, 30 Aug 2024 08:27:19 GMT
//
//

// system include files
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>

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

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

namespace {
    class IntOrWild
    {
    public:
        inline IntOrWild(const std::string& spec)
            : value_(0),
              wild_(false)
        {
            if (spec == "*")
                wild_ = true;
            else
                value_ = std::stoi(spec);
        }

        inline bool matches(const int i) const
        {
            return wild_ || value_ == i;
        }

    private:
        int value_;
        bool wild_;
    };
}

typedef HcalPulseShapeLookup::LabeledShape LabeledShape;

static LabeledShape parseShapeDescription(const edm::ParameterSet& pset,
                                          const unsigned len,
                                          const int timeShift,
                                          const int dtDefault)
{
    const std::string& label = pset.getParameter<std::string>("label");
    int to = static_cast<int>(pset.getParameter<unsigned>("t0")) +
        (label == "default" ? dtDefault : timeShift);
    const std::vector<double> pulse = pset.getParameter<std::vector<double> >("pulse");
    const unsigned sz = pulse.size();
    std::vector<double> shape(len, 0.0);
    double norm = 0.0;
    const int ilen = len;
    for (unsigned from=0; from < sz && to < ilen; ++from, ++to)
        if (to >= 0 && pulse[from] > 0.0)
        {
            shape[to] = pulse[from];
            norm += pulse[from];
        }
    if (norm > 0.0)
        for (unsigned i = 0; i < len; ++i)
            shape[i] /= norm;
    return LabeledShape(label, HcalPulseShape(shape, len));
}

// Needed for sorting vectors of LabeledShape
inline static bool operator<(const HcalPulseShape&, const HcalPulseShape&)
{
    return false;
}

static void updateChannelMap(const edm::ParameterSet& pset,
                             const HcalTopology& htopo,
                             const std::vector<DetId>& ids,
                             const int shapeNumber,
                             std::vector<int>* shapeTypes)
{
    const IntOrWild ieta(pset.getParameter<std::string>("eta"));
    const IntOrWild iphi(pset.getParameter<std::string>("phi"));
    const IntOrWild depth(pset.getParameter<std::string>("depth"));
    for (const DetId& id : ids)
    {
        const HcalDetId hcalId(id);
        if (ieta.matches(hcalId.ieta()) &&
            iphi.matches(hcalId.iphi()) &&
            depth.matches(hcalId.depth()))
        {
            const unsigned linearId = htopo.detId2denseId(id);
            shapeTypes->at(linearId) = shapeNumber;
        }
    }
}

static void dumpPulseShape(std::ostream& of, const std::string& label,
                           const HcalPulseShapeLookup::Shape& pulseShape)
{
    const std::vector<float>& pulse = pulseShape.data();
    of << label << ' ' << pulse.size();
    for (const float v : pulse)
        of << ' ' << v;
    of << std::endl;
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
  // ----------member data ---------------------------
  const edm::ParameterSet config_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topoToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

//
// constructors and destructor
//
HcalPulseShapesEP::HcalPulseShapesEP(const edm::ParameterSet& iConfig)
    : config_(iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("productLabel"));
  topoToken_ = cc.consumes();
  geomToken_ = cc.consumes();

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
    typedef std::vector<edm::ParameterSet> VPSet;

    // What is the standard pulse shape length?
    const unsigned len = config_.getParameter<unsigned>("pulseShapeLength");

    // What is the global time shift?
    const int deltat = config_.getParameter<int>("globalTimeShift");

    // What is the time shift for the default (original) pulse model?
    const int dtDefault = config_.getParameter<int>("defaultTimeShift");

    // Create the collection of pulse shapes
    const VPSet& shapeDescriptions = config_.getParameter<VPSet>("pulseShapes");
    const unsigned nShapes = shapeDescriptions.size();
    std::vector<LabeledShape> shapes;
    shapes.reserve(nShapes);
    for (const edm::ParameterSet& descr : shapeDescriptions)
        shapes.push_back(parseShapeDescription(descr, len, deltat, dtDefault));

    // Sort pulse shapes by label and make sure that the labels are unique
    std::sort(shapes.begin(), shapes.end());
    for (unsigned i=1; i<nShapes; ++i)
        if (shapes[i].first == shapes[i-1].first)
            throw cms::Exception("HcalPulseShapesEPBadConfig")
                << "Duplicate pulse shape label \""
                << shapes[i].first << "\" encountered" << std::endl;

    // Map to simplify subsequent lookups
    std::map<std::string,unsigned> indexLookup;
    for (unsigned i=0; i<nShapes; ++i)
        indexLookup[shapes[i].first] = i;

    // Create the lookup table from the linearized channel numbers
    // into the pulse shapes. Initialize to an invalid value.
    const HcalTopology& htopo = rcd.get(topoToken_);
    const CaloGeometry& geom = rcd.get(geomToken_);
    std::vector<int> shapeTypes(htopo.ncells(), -1);

    // Prepare to cycle over Hcal subdetectors
    constexpr unsigned nSubDet = 2U;
    const HcalSubdetector subdetectors[nSubDet] = {HcalBarrel, HcalEndcap};
    const std::string subDetNames[nSubDet] = {"HB", "HE"};
    const std::string parNames[nSubDet] = {"HBMaps", "HEMaps"};

    // Cycle over Hcal subdetectors
    for (unsigned isub=0; isub<nSubDet; ++isub)
    {
        const HcalGeometry* hcalGeom = static_cast<const HcalGeometry*>(
            geom.getSubdetectorGeometry(DetId::Hcal, subdetectors[isub]));
        const std::vector<DetId>& ids = hcalGeom->getValidDetIds(DetId::Hcal, subdetectors[isub]);

        // Cycle over channel maps for this subdetector
        const VPSet& channelMaps = config_.getParameter<VPSet>(parNames[isub]);
        for (const edm::ParameterSet& cmap : channelMaps)
        {
            const std::string& label = cmap.getParameter<std::string>("label");
            const std::map<std::string,unsigned>::const_iterator it = indexLookup.find(label);
            if (it == indexLookup.end())
                throw cms::Exception("HcalPulseShapesEPBadConfig")
                    << "Previously undefined pulse shape label \""
                    << label << "\" encountered" << std::endl;
            updateChannelMap(cmap, htopo, ids, it->second, &shapeTypes);
        }
    }

    // Create the product
    auto product = std::make_unique<HcalPulseShapeLookup>(shapes, shapeTypes, &htopo);

    // Dump the pulse shapes if requested
    const std::string& pulseDumpFile = config_.getUntrackedParameter<std::string>("pulseDumpFile", "");
    if (!pulseDumpFile.empty())
    {
        std::ofstream of(pulseDumpFile);
        if (!of.is_open())
            throw cms::Exception("HcalPulseShapesEPBadConfig")
                << "Failed to open pulse dump file \"" << pulseDumpFile << '"' << std::endl;

        const unsigned nPulses = product->nShapeTypes();
        for (unsigned ip=0; ip<nPulses; ++ip)
            dumpPulseShape(of, product->getLabel(ip), product->getShape(ip));
    }

    // Dump the channel map if requested
    const std::string& mapDumpFile = config_.getUntrackedParameter<std::string>("mapDumpFile", "");
    if (!mapDumpFile.empty())
    {
        std::ofstream of(mapDumpFile);
        if (!of.is_open())
            throw cms::Exception("HcalPulseShapesEPBadConfig")
                << "Failed to open map dump file \"" << mapDumpFile << '"' << std::endl;

        for (unsigned isub=0; isub<nSubDet; ++isub)
        {
            const HcalGeometry* hcalGeom = static_cast<const HcalGeometry*>(
                geom.getSubdetectorGeometry(DetId::Hcal, subdetectors[isub]));
            const std::vector<DetId>& ids = hcalGeom->getValidDetIds(DetId::Hcal, subdetectors[isub]);
            for (const DetId& id : ids)
            {
                const HcalDetId hcalId(id);
                of << subDetNames[isub]
                   << std::setw(4) << hcalId.ieta()
                   << std::setw(4) << hcalId.iphi()
                   << std::setw(3) << hcalId.depth()
                   << "  " << product->getChannelLabel(id) << std::endl;
            }
        }
    }

    return product;
}

void HcalPulseShapesEP::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
    edm::ParameterSetDescription desc;

    desc.add<std::string>("productLabel", "HcalDataShapes");
    desc.addUntracked<std::string>("pulseDumpFile", "");
    desc.addUntracked<std::string>("mapDumpFile", "");
    desc.add<unsigned>("pulseShapeLength", 250U);
    desc.add<int>("globalTimeShift", 0);
    desc.add<int>("defaultTimeShift", 0);

    {
        edm::ParameterSetDescription validator;
        validator.add<std::string>("label");
        validator.add<unsigned>("t0");
        validator.add<std::vector<double> >("pulse");

        std::vector<edm::ParameterSet> vDefaults;
        edm::ParameterSet vDefaults0;
        vDefaults0.addParameter<std::string>("label", "default");
        vDefaults0.addParameter<unsigned>("t0", 0);
        const std::vector<double> defaultShape = {
            5.22174e-12, 7.04852e-10, 3.49584e-08, 7.78029e-07, 9.11847e-06, 6.39666e-05, 0.000297587, 0.000996661,
            0.00256618,  0.00535396,  0.00944073,  0.0145521,   0.020145,    0.0255936,   0.0303632,   0.0341078,
            0.0366849,   0.0381183,   0.0385392,   0.0381327,   0.0370956,   0.0356113,   0.0338366,   0.0318978,
            0.029891,    0.0278866,   0.0259336,   0.0240643,   0.0222981,   0.0206453,   0.0191097,   0.0176902,
            0.0163832,   0.0151829,   0.0140826,   0.0130752,   0.0121533,   0.01131,     0.0105382,   0.00983178,
            0.00918467,  0.00859143,  0.00804709,  0.0075471,   0.00708733,  0.00666406,  0.00627393,  0.00591389,
            0.00558122,  0.00527344,  0.00498834,  0.00472392,  0.00447837,  0.00425007,  0.00403754,  0.00383947,
            0.00365465,  0.00348199,  0.00332052,  0.00316934,  0.00302764,  0.0028947,   0.00276983,  0.00265242,
            0.00254193,  0.00243785,  0.00233971,  0.00224709,  0.0021596,   0.00207687,  0.0019986,   0.00192447,
            0.00185421,  0.00178756,  0.0017243,   0.00166419,  0.00160705,  0.00155268,  0.00150093,  0.00145162,
            0.00140461,  0.00135976,  0.00131696,  0.00127607,  0.00123699,  0.00119962,  0.00116386,  0.00112963,
            0.00109683,  0.0010654,   0.00103526,  0.00100634,  0.000978578, 0.000951917, 0.000926299, 0.000901672,
            0.000877987, 0.000855198, 0.00083326,  0.000812133, 0.000791778, 0.000772159, 0.000753242, 0.000734994,
            0.000717384, 0.000700385, 0.000683967, 0.000668107, 0.000652779, 0.00063796,  0.000623629, 0.000609764,
            0.000596346, 0.000583356, 0.000570777, 0.000558592, 0.000546785, 0.00053534,  0.000524243, 0.000513481,
            0.00050304,  0.000492907, 0.000483072, 0.000473523, 0.000464248, 0.000455238, 0.000446483, 0.000437974,
            0.0004297,   0.000421655, 0.00041383,  0.000406216, 0.000398807, 0.000391595, 0.000384574, 0.000377736,
            0.000371076, 0.000364588, 0.000358266, 0.000352104, 0.000346097, 0.00034024,  0.000334528, 0.000328956,
            0.00032352,  0.000318216, 0.000313039, 0.000307986, 0.000303052, 0.000298234, 0.000293528, 0.000288931,
            0.000284439, 0.00028005,  0.000275761, 0.000271567, 0.000267468, 0.000263459, 0.000259538, 0.000255703,
            0.000251951, 0.00024828,  0.000244688, 0.000241172, 0.00023773,  0.000234361, 0.000231061, 0.00022783,
            0.000224666, 0.000221566, 0.000218528, 0.000215553, 0.000212636, 0.000209778, 0.000206977, 0.00020423,
            0.000201537, 0.000198896, 0.000196307, 0.000193767, 0.000191275, 0.000188831, 0.000186432, 0.000184079,
            0.000181769, 0.000179502, 0.000177277, 0.000175092, 0.000172947, 0.000170841, 0.000168772, 0.000166741,
            0.000164745, 0.000162785, 0.000160859, 0.000158967, 0.000157108, 0.00015528,  0.000153484, 0.000151719,
            0.000149984, 0.000148278, 0.000146601, 0.000144951, 0.000143329, 0.000141734, 0.000140165, 0.000138622,
            0.000137104, 0.00013561,  0.000134141, 0.000132695, 0.000131272, 0.000129871, 0.000128493, 0.000127136,
            0.000125801, 0.000124486, 0.000123191, 0.000121917, 0.000120662, 0.000119426, 0.000118209, 0.00011701,
            0.000115829, 0.000114665, 0.000113519, 0.00011239,  0.000111278, 0.000110182, 0.000109102, 0.000108037,
            0.000106988, 0.000105954, 0.000104935, 0.00010393,  0.000102939, 0.000101963, 0.000101,    0.000100051,
            9.91146e-05, 9.81915e-05, 9.7281e-05,  9.63831e-05, 9.54975e-05, 9.46239e-05, 9.37621e-05, 9.2912e-05,
            9.20733e-05, 9.12458e-05};
        vDefaults0.addParameter<std::vector<double> >("pulse", defaultShape);
        vDefaults.push_back(vDefaults0);

        desc.addVPSet("pulseShapes", validator, vDefaults);
    }

    {
        edm::ParameterSetDescription validator;
        validator.add<std::string>("label");
        validator.add<std::string>("eta");
        validator.add<std::string>("phi");
        validator.add<std::string>("depth");

        std::vector<edm::ParameterSet> vDefaults;
        edm::ParameterSet vDefaults0;
        vDefaults0.addParameter<std::string>("label", "default");
        vDefaults0.addParameter<std::string>("eta", "*");
        vDefaults0.addParameter<std::string>("phi", "*");
        vDefaults0.addParameter<std::string>("depth", "*");
        vDefaults.push_back(vDefaults0);

        desc.addVPSet("HBMaps", validator, vDefaults);
        desc.addVPSet("HEMaps", validator, vDefaults);
    }

    descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalPulseShapesEP);
