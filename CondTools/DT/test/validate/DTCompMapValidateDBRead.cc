
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondTools/DT/test/validate/DTCompMapValidateDBRead.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

DTCompMapValidateDBRead::DTCompMapValidateDBRead(edm::ParameterSet const& p)
    : dataFileName(p.getParameter<std::string>("chkFile")), elogFileName(p.getParameter<std::string>("logFile")) {}

DTCompMapValidateDBRead::DTCompMapValidateDBRead(int i) {}

DTCompMapValidateDBRead::~DTCompMapValidateDBRead() {}

void DTCompMapValidateDBRead::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
  std::stringstream run_fn;
  //  run_fn << "run" << e.id().run() << dataFileName;
  //  std::ifstream chkFile( run_fn.str().c_str() );
  std::ifstream chkFile(dataFileName.c_str());
  std::ofstream logFile(elogFileName.c_str(), std::ios_base::app);
  edm::ESHandle<DTReadOutMapping> ro;
  context.get<DTReadOutMappingRcd>().get(ro);
  std::cout << ro->mapRobRos() << " " << ro->mapCellTdc() << std::endl;
  std::cout << std::distance(ro->begin(), ro->end()) << " data in the container" << std::endl;
  int whe;
  int sta;
  int sec;
  int qua;
  int lay;
  int cel;
  int ddu;
  int ros;
  int rob;
  int tdc;
  int cha;
  int ckwhe;
  int cksta;
  int cksec;
  int ckqua;
  int cklay;
  int ckcel;
  int ckddu;
  int ckros;
  int ckrob;
  int cktdc;
  int ckcha;
  while (chkFile >> ckddu >> ckros >> ckrob >> cktdc >> ckcha >> ckwhe >> cksta >> cksec >> ckqua >> cklay >> ckcel) {
    /*
    if ( ( cklay ==  4 ) &&
         ( ckcel ==  1 ) ) continue;
    if ( ( cksta ==  4 ) &&
         ( cksec ==  8 ) &&
         ( ckcel >  92 ) ) continue;
    if ( ( cksta ==  4 ) &&
         ( cksec == 12 ) &&
         ( ckcel >  92 ) ) continue;
    if ( ( cksta ==  4 ) &&
         ( cksec ==  8 ) &&
         ( cklay ==  1 ) &&
         ( ckcel >  91 ) ) continue;
    if ( ( cksta ==  4 ) &&
         ( cksec == 12 ) &&
         ( cklay ==  1 ) &&
         ( ckcel >  91 ) ) continue;
    if ( ( ckwhe == -1 ) &&
         ( cksec ==  3 ) &&
         ( ckqua ==  2 ) &&
         ( ckcel >  48 ) ) continue;
    if ( ( ckwhe ==  1 ) &&
         ( cksec ==  4 ) &&
         ( ckqua ==  2 ) &&
         ( ckcel >  48 ) ) continue;
*/
    ro->readOutToGeometry(ckddu, ckros, ckrob, cktdc, ckcha, whe, sta, sec, qua, lay, cel);
    if ((ckwhe != whe) || (cksta != sta) || (cksec != sec) || (ckqua != qua) || (cklay != lay) || (ckcel != cel))
      logFile << "MISMATCH IN WRITING AND READING chan->cell map " << ckddu << " " << ckros << " " << ckrob << " "
              << cktdc << " " << ckcha << " : " << ckwhe << " " << cksta << " " << cksec << " " << ckqua << " " << cklay
              << " " << ckcel << " -> " << whe << " " << sta << " " << sec << " " << qua << " " << lay << " " << cel
              << std::endl;
    ro->geometryToReadOut(ckwhe, cksta, cksec, ckqua, cklay, ckcel, ddu, ros, rob, tdc, cha);
    if ((ckddu != ddu) || (ckros != ros) || (ckrob != rob) || (cktdc != tdc) || (ckcha != cha))
      logFile << "MISMATCH IN WRITING AND READING cell->chan map " << ckwhe << " " << cksta << " " << cksec << " "
              << ckqua << " " << cklay << " " << ckcel << " : " << ckddu << " " << ckros << " " << ckrob << " " << cktdc
              << " " << ckcha << " -> " << ddu << " " << ros << " " << rob << " " << tdc << " " << cha << std::endl;
  }
}

void DTCompMapValidateDBRead::endJob() {
  std::ifstream logFile(elogFileName.c_str());
  char* line = new char[1000];
  int errors = 0;
  std::cout << "ReadOut Map validation result:" << std::endl;
  while (logFile.getline(line, 1000)) {
    std::cout << line << std::endl;
    errors++;
  }
  if (!errors) {
    std::cout << " ********************************* " << std::endl;
    std::cout << " ***                           *** " << std::endl;
    std::cout << " ***      NO ERRORS FOUND      *** " << std::endl;
    std::cout << " ***                           *** " << std::endl;
    std::cout << " ********************************* " << std::endl;
  }
  return;
}

DEFINE_FWK_MODULE(DTCompMapValidateDBRead);
