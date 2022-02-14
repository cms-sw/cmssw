// system include files
#include <fstream>
#include <map>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Provenance/interface/Provenance.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"

//
// class decleration
//
namespace cms {
  class HcalConstantsASCIIWriter : public edm::one::EDAnalyzer<> {
  public:
    explicit HcalConstantsASCIIWriter(const edm::ParameterSet&);
    ~HcalConstantsASCIIWriter();

    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void beginJob();
    virtual void endJob();

  private:
    // ----------member data ---------------------------
    const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
    const edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_resp_;

    std::ofstream* myout_hcal;
    std::string file_input;
    std::string file_output;
  };
}  // namespace cms

//#define EDM_ML_DEBUG

//
// constructors and destructor
//
namespace cms {
  HcalConstantsASCIIWriter::HcalConstantsASCIIWriter(const edm::ParameterSet& iConfig)
      : tok_geom_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
        tok_resp_(esConsumes<HcalRespCorrs, HcalRespCorrsRcd>()) {
    // get name of output file with histogramms
    file_input = "Calibration/HcalCalibAlgos/data/" + iConfig.getParameter<std::string>("fileInput") + ".txt";
    file_output = "Calibration/HcalCalibAlgos/data/" + iConfig.getParameter<std::string>("fileOutput") + ".txt";
  }

  HcalConstantsASCIIWriter::~HcalConstantsASCIIWriter() {
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
  }

  void HcalConstantsASCIIWriter::beginJob() {
    edm::FileInPath f1(file_output);
    std::string fDataFile = f1.fullPath();

    myout_hcal = new std::ofstream(fDataFile.c_str());
    if (!myout_hcal)
      edm::LogVerbatim("HcalCalib") << " Output file not open!!! ";
  }

  void HcalConstantsASCIIWriter::endJob() { delete myout_hcal; }

  //
  // member functions
  //

  // ------------ method called to produce the data  ------------
  void HcalConstantsASCIIWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    edm::LogVerbatim("HcalCalib") << " Start HcalConstantsASCIIWriter::analyze";

    HcalRespCorrs* oldRespCorrs = new HcalRespCorrs(iSetup.getData(tok_resp_));
    //    std::vector<DetId> dd = oldRespCorrs->getAllChannels();

    const CaloGeometry* geo = &iSetup.getData(tok_geom_);
    //   iSetup.get<HcalDbRecord>().get(conditions);

    std::vector<DetId> did = geo->getValidDetIds();

    std::map<HcalDetId, float> corrold;
    //map<HcalDetId,float> corrnew;

    int mysubd, depth, ieta, iphi;
    float coradd, corerr;

    std::vector<HcalDetId> theVector;
    for (std::vector<DetId>::iterator i = did.begin(); i != did.end(); i++) {
      if ((*i).det() == DetId::Hcal) {
        HcalDetId hid = HcalDetId(*i);
        theVector.push_back(hid);
        corrold[hid] = (oldRespCorrs->getValues(*i))->getValue();
        edm::LogVerbatim("HcalCalib") << " Old calibration " << hid.depth() << " " << hid.ieta() << " " << hid.iphi();
      }
    }

    edm::LogVerbatim("HcalCalib") << " Get old calibration ";
    // Read new corrections from file

    edm::FileInPath f1(file_input);
    std::string fDataFile = f1.fullPath();

    std::ifstream in(fDataFile.c_str());
    std::string line;

    double corrnew_p[5][5][45][75];
    double corrnew_m[5][5][45][75];
    edm::LogVerbatim("HcalCalib") << " Start to read txt file " << fDataFile.c_str() << std::endl;
    while (std::getline(in, line)) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalCalib") << " Line size " << line.size() << " " << line;
#endif

      if (!line.size() || line[0] == '#')
        continue;
      std::istringstream linestream(line);

      linestream >> mysubd >> depth >> ieta >> iphi >> coradd >> corerr;
#ifdef EDM_ML_DEBUG
      HcalDetId hid(HcalSubdetector(mysubd), ieta, iphi, depth);
      edm::LogVerbatim("HcalCalib") << " Check mysubd " << hid.subdet() << " depth " << hid.depth() << " ieta "
                                    << hid.ieta() << " iphi " << hid.iphi() << " " << hid.rawId();
#endif
      int ietak = ieta;
      if (ieta < 0)
        ietak = -1 * ieta;
      if (ieta > 0)
        corrnew_p[mysubd][depth][ietak][iphi] = coradd;
      if (ieta < 0)
        corrnew_m[mysubd][depth][ietak][iphi] = coradd;
      edm::LogVerbatim("HcalCalib") << " Try to initialize mysubd " << mysubd << " depth " << depth << " ieta " << ieta
                                    << " " << ietak << " iphi " << iphi << " " << coradd;
    }

    HcalRespCorrs* mycorrections = new HcalRespCorrs(oldRespCorrs->topo());

    for (std::vector<HcalDetId>::iterator it = theVector.begin(); it != theVector.end(); it++) {
      float cc1 = (*corrold.find(*it)).second;
      //    float cc2 = (*corrnew.find(*it)).second;
      float cc2 = 0.;
      int ietak = (*it).ieta();

      if ((*it).ieta() < 0)
        ietak = -1 * (*it).ieta();

      if ((*it).ieta() > 0)
        cc2 = corrnew_p[(*it).subdet()][(*it).depth()][ietak][(*it).iphi()];
      if ((*it).ieta() < 0)
        cc2 = corrnew_m[(*it).subdet()][(*it).depth()][ietak][(*it).iphi()];

      float cc = cc1 * cc2;
      edm::LogVerbatim("HcalCalib") << " Multiply " << (*it).subdet() << " " << (*it).depth() << " " << (*it).ieta()
                                    << " " << ietak << " " << (*it).iphi() << " " << (*it).rawId() << " " << cc1 << " "
                                    << cc2;

      // now make the basic object for one cell with HcalDetId myDetId containing the value myValue
      HcalRespCorr item((*it).rawId(), cc);
      mycorrections->addValues(item);
    }

    HcalRespCorrs mycc = *mycorrections;
    HcalDbASCIIIO::dumpObject(*myout_hcal, mycc);
  }
}  // namespace cms
//define this as a plug-in

#include "FWCore/Framework/interface/MakerMacros.h"

using cms::HcalConstantsASCIIWriter;
DEFINE_FWK_MODULE(HcalConstantsASCIIWriter);
