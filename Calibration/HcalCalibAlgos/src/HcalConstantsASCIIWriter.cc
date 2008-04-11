// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Calibration/HcalCalibAlgos/interface/HcalConstantsASCIIWriter.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

using namespace std;
using namespace reco;
//
// constructors and destructor
//
namespace cms{
HcalConstantsASCIIWriter::HcalConstantsASCIIWriter(const edm::ParameterSet& iConfig)
{
  // get name of output file with histogramms
  file_input="/Calibration/HcalCalibAlgos/data/"+iConfig.getParameter <std::string> ("tagName1")+".txt";
  
}

HcalConstantsASCIIWriter::~HcalConstantsASCIIWriter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void HcalConstantsASCIIWriter::beginJob( const edm::EventSetup& iSetup)
{

  std::string ccc = "hcal_new.dat";

  myout_hcal = new ofstream(ccc.c_str());
  if(!myout_hcal) cout << " Output file not open!!! "<<endl;
    
}

void HcalConstantsASCIIWriter::endJob()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HcalConstantsASCIIWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::cout<<" Start HcalConstantsASCIIWriter::analyze "<<std::endl;
  
    edm::ESHandle<HcalRespCorrs> r;
    iSetup.get<HcalRespCorrsRcd>().get(r);
    HcalRespCorrs* oldRespCorrs = new HcalRespCorrs(*r.product());
    std::vector<DetId> dd = oldRespCorrs->getAllChannels();
   
    map<HcalDetId,float> corrold;
    map<HcalDetId,float> corrnew; 
    
    int mydet,mysubd,depth,ieta,iphi;
    float coradd;
     
    vector<HcalDetId> theVector; 
    for(std::vector<DetId>::iterator i = dd.begin(); i != dd.end(); i++)
    {
    
      HcalDetId hid = HcalDetId(*i);
      theVector.push_back(hid);
      corrold[hid] = (oldRespCorrs->getValues(*i))->getValue();
      
    }

// Read new corrections from file

    edm::FileInPath f1(file_input);
    string fDataFile = f1.fullPath();
     
    std::ifstream in( fDataFile.c_str() );
    string line;
    
    
    
    while( std::getline( in, line)){
      if(!line.size() || line[0]=='#') continue;
      istringstream linestream(line);
      double par;
      int type;
      linestream>>mydet>>mysubd>>depth>>ieta>>iphi>>coradd;
        HcalDetId  hid(HcalSubdetector(mysubd),ieta,iphi,depth);
        corrnew[hid] =  coradd; 
    } 
    
    HcalRespCorrs* mycorrections = new HcalRespCorrs(); 
    
   for(vector<HcalDetId>::iterator it = theVector.begin(); it != theVector.end(); it++)
   {
     float cc1 = (*corrold.find(*it)).second;
     float cc2 = (*corrnew.find(*it)).second;
     float cc = cc1*cc2;
  // now make the basic object for one cell with HcalDetId myDetId containing the value myValue
      HcalRespCorr item ((*it).rawId(), cc);
      bool rr = mycorrections->addValues(item);
   }   

    HcalRespCorrs mycc = *mycorrections;
    HcalDbASCIIIO::dumpObject (*myout_hcal, mycc);

}
}
//define this as a plug-in
//DEFINE_ANOTHER_FWK_MODULE(HcalConstantsASCIIWriter)

