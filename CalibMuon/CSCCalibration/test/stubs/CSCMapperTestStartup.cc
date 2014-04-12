#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperBase.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperRecord.h"
#include <iostream>

class CSCMapperTestStartup : public edm::EDAnalyzer {
public:
  explicit CSCMapperTestStartup(const edm::ParameterSet&);
  ~CSCMapperTestStartup();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) ;
  virtual void endJob() ;

  std::string algoName;
};

CSCMapperTestStartup::CSCMapperTestStartup(const edm::ParameterSet& pset) {}

CSCMapperTestStartup::~CSCMapperTestStartup() {}

void CSCMapperTestStartup::analyze(const edm::Event& ev, const edm::EventSetup& esu)
{
  const int egeo[] = {  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2 }; // endcap 1=+z, 2=-z
  const int sgeo[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // station 1-4
  const int rgeo[] = {  1,  1,  1,  4,  4,  4,  1,  1,  1,  4,  4,  4 }; // ring 1-4
  const int cgeo[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // chamber 1-18/36
  const int lgeo[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // layer 1-6
  const int tgeo[] = {  1, 32, 64,  1,  8, 16,  1, 32, 64,  1,  8, 16 }; // strip 1-80 (16, 48 64)

  const int eraw[] = {  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2 }; // endcap 1=+z, 2=-z
  const int sraw[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // station 1-4
  const int rraw[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // ring 1-3
  const int craw[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // chamber 1-18/36
  const int lraw[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // layer 1-6
  const int traw[] = {  1, 32, 64, 65, 72, 80,  1, 32, 64, 65, 72, 80 }; // strip 1-80 (16, 48 64)

  const int estrip[] = {  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2 }; // endcap 1=+z, 2=-z
  const int sstrip[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // station 1-4
  const int rstrip[] = {  1,  1,  1,  4,  4,  4,  1,  1,  1,  4,  4,  4 }; // ring 1-3
  const int cstrip[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // chamber 1-18/36
  const int lstrip[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // layer 1-6
  const int tstrip[] = {  1, 32, 64, 65, 72, 80,  1, 32, 64, 65, 72, 80 }; // strip 1-80 (16, 48 64)

  const int edetid[] = {  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2 }; // endcap 1=+z, 2=-z
  const int sdetid[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // station 1-4
  const int rdetid[] = {  1,  1,  1,  4,  4,  4,  1,  1,  1,  4,  4,  4 }; // ring 1-3
  const int cdetid[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // chamber 1-18/36
  const int ldetid[] = {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 }; // layer 1-6

  const CSCChannelMapperRecord& irec = esu.get<CSCChannelMapperRecord>();
  edm::ESHandle<CSCChannelMapperBase> mapper_;
  irec.get(mapper_);

  algoName = mapper_->name();

  std::cout << "CSCMapperTestStartup: analyze sees algorithm " << algoName  << " in Event Setup" << std::endl;

  size_t isiz = 12;

  std::cout << "\n" << algoName << "::rawStripChannel for:" << std::endl;

  std::cout << " E   S   R   C   L   geom   raw\n"
            << "                     chan   chan" << std::endl;

  for (size_t i=0; i<isiz; ++i ) {
    int ie=egeo[i];
    int is=sgeo[i];
    int ir=rgeo[i];
    int ic=cgeo[i];
    int il=lgeo[i];
    int igeo=tgeo[i];
  
    CSCDetId id(ie, is, ir, ic, il);
    int iraw = mapper_->rawStripChannel(id, igeo);
 
    std::cout << std::setw(2) << ie << std::setw(4) << is << std::setw(4) << ir << std::setw(4)
	      << ic << std::setw(4) << il << std::setw(7) << igeo << std::setw(7) << iraw << std::endl;

  }

  std::cout << "\n" << algoName << "::geomStripChannel for:" << std::endl;

  std::cout << " E   S   R   C   L    raw   geom\n"
            << "                     chan   chan" << std::endl;

  for (size_t i=0; i<isiz; ++i ) {
    int ie=eraw[i];
    int is=sraw[i];
    int ir=rraw[i];
    int ic=craw[i];
    int il=lraw[i];
    int igeo=traw[i];
  
    CSCDetId id(ie, is, ir, ic, il);
    int iraw = mapper_->geomStripChannel(id, igeo);
 
    std::cout << std::setw(2) << ie << std::setw(4) << is << std::setw(4) << ir << std::setw(4)
	      << ic << std::setw(4) << il << std::setw(7) << igeo << std::setw(7) << iraw << std::endl;

  }

  std::cout << "\n" << algoName << "::channelFromStrip for:" << std::endl;
  
  std::cout << " E   S   R   C   L   strip  chan" << std::endl;

  for (size_t i=0; i<isiz; ++i ) {
    int ie=estrip[i];
    int is=sstrip[i];
    int ir=rstrip[i];
    int ic=cstrip[i];
    int il=lstrip[i];
    int istrip=tstrip[i];
  
    CSCDetId id(ie, is, ir, ic, il);
    int ichan = mapper_->channelFromStrip(id, istrip);
 
    std::cout << std::setw(2) << ie << std::setw(4) << is << std::setw(4) << ir << std::setw(4)
	      << ic << std::setw(4) << il << std::setw(7) << istrip << std::setw(7) << ichan << std::endl;

  }

  std::cout << "\n" << algoName << "::rawCSCDetId for:" << std::endl;
  
  std::cout << "          " << "CSCDetId IN" << "         " << "CSCDetId OUT" << std::endl;

  for (size_t i=0; i<isiz; ++i ) {
    int ie=edetid[i];
    int is=sdetid[i];
    int ir=rdetid[i];
    int ic=cdetid[i];
    int il=ldetid[i];
  
    CSCDetId id(ie, is, ir, ic, il);
    CSCDetId idout = mapper_->rawCSCDetId(id);
 
    std::cout << " " << id << " " << idout << std::endl;
  }

}


void CSCMapperTestStartup::beginJob() {}

void CSCMapperTestStartup::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCMapperTestStartup);
