#include <string>
#include <map>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Utilities/General/interface/ClassName.h"
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

class CaloAlignmentRcdRead : public edm::EDAnalyzer
{
public:

  explicit CaloAlignmentRcdRead( const edm::ParameterSet& /*iConfig*/ )
    :nEventCalls_(0) {}
  ~CaloAlignmentRcdRead() {}
  
  template<typename T>
  void dumpAlignments(const edm::EventSetup& evtSetup);

  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup); 
  
private:

  unsigned int nEventCalls_;
};

template<typename T>
void CaloAlignmentRcdRead::dumpAlignments(const edm::EventSetup& evtSetup)
{
  edm::ESHandle<Alignments> alignments;
  evtSetup.get<T>().get(alignments);
  
  std::string recordName = Demangle(typeid(T).name())();

  std::cout << "Dumping alignments: " << recordName << std::endl;

  for (std::vector<AlignTransform>::const_iterator i = alignments->m_align.begin();
       i != alignments->m_align.end();  
       ++i) {
    std::cout << "entry " << i->rawId() 
	      << " translation " << i->translation() 
	      << " angles " << i->rotation().eulerAngles() << std::endl;
  }
}

void CaloAlignmentRcdRead::analyze(const edm::Event& /*evt*/, const edm::EventSetup& evtSetup)
{
  if (nEventCalls_>0) {
    std::cout << "Reading from DB to be done only once, "
	      << "set 'untracked PSet maxEvents = {untracked int32 input = 1}'." << std::endl;
 
    return;
  }

  std::cout << "Reading from database in CaloAlignmentRcdRead::analyze...\n" << std::endl;
  
  dumpAlignments<EBAlignmentRcd>(evtSetup);
  dumpAlignments<EEAlignmentRcd>(evtSetup);
  dumpAlignments<ESAlignmentRcd>(evtSetup);

  std::cout << "\ndone!" << std::endl;

  nEventCalls_++;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloAlignmentRcdRead);
