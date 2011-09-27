#include <string>
#include <map>
#include <vector>

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Utilities/General/interface/ClassName.h"
#include "DataFormats/DetId/interface/DetId.h"

// Database
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"

#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CastorAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CastorAlignmentErrorRcd.h"

class CaloAlignmentRcdWrite : public edm::EDAnalyzer
{
public:

  explicit CaloAlignmentRcdWrite(const edm::ParameterSet& /*iConfig*/)
    :nEventCalls_(0) {}
  ~CaloAlignmentRcdWrite() {}
  
  template<typename T>
  void writeAlignments(const edm::EventSetup& evtSetup);

  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);

private:
  
  unsigned int nEventCalls_;
};

template<typename T>
void CaloAlignmentRcdWrite::writeAlignments(const edm::EventSetup& evtSetup)
{
  edm::ESHandle<Alignments> alignmentsHandle;
  evtSetup.get<T>().get(alignmentsHandle);
  
  std::string recordName = Demangle(typeid(T).name())();

  std::cout << "Uploading alignments to the database: " << recordName << std::endl;

  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (!poolDbService.isAvailable())
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
  
  Alignments * alignments = new Alignments(*alignmentsHandle);

  poolDbService->writeOne<Alignments>(&(*alignments), 
				      poolDbService->currentTime(),
				      recordName);
}
  
void CaloAlignmentRcdWrite::analyze(const edm::Event& /*evt*/, const edm::EventSetup& evtSetup)
{
   if (nEventCalls_ > 0) {
     std::cout << "Writing to DB to be done only once, "
	       << "set 'untracked PSet maxEvents = {untracked int32 input = 1}'."
	       << "(Your writing should be fine.)" << std::endl;
     return;
   }

   writeAlignments<EBAlignmentRcd>(evtSetup);
   writeAlignments<EEAlignmentRcd>(evtSetup);
   writeAlignments<ESAlignmentRcd>(evtSetup);
   
   std::cout << "done!" << std::endl;
   nEventCalls_++;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloAlignmentRcdWrite);
