// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "HcalConstantsASCIIWriter.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

//
// constructors and destructor
//
namespace cms {
  HcalConstantsASCIIWriter::HcalConstantsASCIIWriter(const edm::ParameterSet& iConfig) {
    tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
    tok_resp_ = esConsumes<HcalRespCorrs, HcalRespCorrsRcd>();
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
      std::cout << " Output file not open!!! " << std::endl;
  }

  void HcalConstantsASCIIWriter::endJob() { delete myout_hcal; }

  //
  // member functions
  //

  // ------------ method called to produce the data  ------------
  void HcalConstantsASCIIWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    std::cout << " Start HcalConstantsASCIIWriter::analyze " << std::endl;

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
        std::cout << " Old calibration " << hid.depth() << " " << hid.ieta() << " " << hid.iphi() << std::endl;
      }
    }

    std::cout << " Get old calibration " << std::endl;
    // Read new corrections from file

    edm::FileInPath f1(file_input);
    std::string fDataFile = f1.fullPath();

    std::ifstream in(fDataFile.c_str());
    std::string line;

    double corrnew_p[5][5][45][75];
    double corrnew_m[5][5][45][75];
    std::cout << " Start to read txt file " << fDataFile.c_str() << std::endl;
    while (std::getline(in, line)) {
      //    std::cout<<" Line size "<<line.size()<< " "<<line<< std::endl;

      if (!line.size() || line[0] == '#')
        continue;
      std::istringstream linestream(line);

      linestream >> mysubd >> depth >> ieta >> iphi >> coradd >> corerr;
      //      DetId mydid(DetId::Hcal,HcalSubdetector(mysubd));
      //      HcalDetId  hid(HcalSubdetector(mysubd),ieta,iphi,depth);
      //        HcalDetId hid(mydid);
      //      std::cout<<" Check mysubd "<<hid.subdet()<<" depth "<<hid.depth()<<" ieta "<<hid.ieta()<<" iphi "<<hid.iphi()<<" "<<hid.rawId()<< std::endl;
      int ietak = ieta;
      if (ieta < 0)
        ietak = -1 * ieta;
      if (ieta > 0)
        corrnew_p[mysubd][depth][ietak][iphi] = coradd;
      if (ieta < 0)
        corrnew_m[mysubd][depth][ietak][iphi] = coradd;
      std::cout << " Try to initialize mysubd " << mysubd << " depth " << depth << " ieta " << ieta << " " << ietak
                << " iphi " << iphi << " " << coradd << std::endl;
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
      std::cout << " Multiply " << (*it).subdet() << " " << (*it).depth() << " " << (*it).ieta() << " " << ietak << " "
                << (*it).iphi() << " " << (*it).rawId() << " " << cc1 << " " << cc2 << std::endl;

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
