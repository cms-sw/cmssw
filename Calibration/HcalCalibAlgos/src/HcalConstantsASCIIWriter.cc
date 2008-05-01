// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Calibration/HcalCalibAlgos/interface/HcalConstantsASCIIWriter.h"
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
  file_input="Calibration/HcalCalibAlgos/data/"+iConfig.getParameter <std::string> ("fileInput")+".txt";
  file_output="Calibration/HcalCalibAlgos/data/"+iConfig.getParameter <std::string> ("fileOutput")+".txt"; 
}

HcalConstantsASCIIWriter::~HcalConstantsASCIIWriter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void HcalConstantsASCIIWriter::beginJob( const edm::EventSetup& iSetup)
{
    edm::FileInPath f1(file_output);
    string fDataFile = f1.fullPath();

  myout_hcal = new ofstream(fDataFile.c_str());
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
//    std::vector<DetId> dd = oldRespCorrs->getAllChannels();
  
   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);
   const CaloGeometry* geo = pG.product();
//   iSetup.get<HcalDbRecord>().get(conditions);
   
   std::vector<DetId> did =  geo->getValidDetIds();

 
    map<HcalDetId,float> corrold;
    //map<HcalDetId,float> corrnew; 
    
    int mydet,mysubd,depth,ieta,iphi;
    float coradd,corerr;
     
    vector<HcalDetId> theVector; 
    for(std::vector<DetId>::iterator i = did.begin(); i != did.end(); i++)
    {
      if( (*i).det() == DetId::Hcal ) { 
      HcalDetId hid = HcalDetId(*i);
      theVector.push_back(hid);
      corrold[hid] = (oldRespCorrs->getValues(*i))->getValue();
      std::cout<<" Old calibration "<<hid.depth()<<" "<<hid.ieta()<<" "<<hid.iphi()<<std::endl;
      } 
    }

    std::cout<<" Get old calibration "<<std::endl;
// Read new corrections from file

    edm::FileInPath f1(file_input);
    string fDataFile = f1.fullPath();
     
    std::ifstream in( fDataFile.c_str() );
    string line;
    
   double corrnew_p[5][5][45][75]; 
   double corrnew_m[5][5][45][75];
   std::cout<<" Start to read txt file "<<fDataFile.c_str()<<std::endl; 
    while( std::getline( in, line)){

//    std::cout<<" Line size "<<line.size()<< " "<<line<< std::endl;

      if(!line.size() || line[0]=='#') continue;
      istringstream linestream(line);
      double par;
      int type;
      linestream>>mysubd>>depth>>ieta>>iphi>>coradd>>corerr;
//      DetId mydid(DetId::Hcal,HcalSubdetector(mysubd));
//      HcalDetId  hid(HcalSubdetector(mysubd),ieta,iphi,depth);
//        HcalDetId hid(mydid);
//      std::cout<<" Check mysubd "<<hid.subdet()<<" depth "<<hid.depth()<<" ieta "<<hid.ieta()<<" iphi "<<hid.iphi()<<" "<<hid.rawId()<< std::endl;
      int ietak = ieta; 
      if(ieta<0) ietak = -1*ieta;
      if(ieta>0) corrnew_p[mysubd][depth][ietak][iphi] =  coradd; 
      if(ieta<0) corrnew_m[mysubd][depth][ietak][iphi] =  coradd;
      std::cout<<" Try to initialize mysubd "<<mysubd<<" depth "<<depth<<" ieta "<<ieta<<" "<<ietak<<" iphi "<<iphi<<" "<<coradd<<
      std::endl;
    } 
    
    HcalRespCorrs* mycorrections = new HcalRespCorrs(); 
    
   for(vector<HcalDetId>::iterator it = theVector.begin(); it != theVector.end(); it++)
   {
     float cc1 = (*corrold.find(*it)).second;
 //    float cc2 = (*corrnew.find(*it)).second;
     float cc2 = 0.;
     int ietak = (*it).ieta();

     if((*it).ieta()<0) ietak=-1*(*it).ieta();

     if((*it).ieta()>0) cc2 = corrnew_p[(*it).subdet()][(*it).depth()][ietak][(*it).iphi()];
     if((*it).ieta()<0) cc2 = corrnew_m[(*it).subdet()][(*it).depth()][ietak][(*it).iphi()];

     float cc = cc1*cc2;
      std::cout<<" Multiply "<<(*it).subdet()<<" "<<(*it).depth()<<" "<<(*it).ieta()<<" "<<ietak<<" "<<(*it).iphi()<<" "<<(*it).rawId()<<" "<<cc1<<" "<<cc2<<std::endl;

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

