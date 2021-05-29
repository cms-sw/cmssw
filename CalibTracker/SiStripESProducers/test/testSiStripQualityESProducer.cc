#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "CalibTracker/SiStripESProducers/test/testSiStripQualityESProducer.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <sstream>

testSiStripQualityESProducer::testSiStripQualityESProducer(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)),
      m_cacheID_(0),
      firstIOV(true),
      twoRecordComparison_(iConfig.getUntrackedParameter<bool>("twoRecordComparison", false)),
      dataLabel_(iConfig.getUntrackedParameter<std::string>("dataLabel", "")),
      dataLabelTwo_(iConfig.getUntrackedParameter<std::string>("dataLabelTwo", "")) {}

void testSiStripQualityESProducer::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  unsigned long long cacheID = iSetup.get<SiStripQualityRcd>().cacheIdentifier();

  if (m_cacheID_ == cacheID)
    return;

  m_cacheID_ = cacheID;

  char canvas[1024] = "\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n";

  //&&&&&&&&&&&&&&&&&&
  //First Record
  //&&&&&&&&&&&&&&&&&&
  edm::ESHandle<SiStripQuality> SiStripQualityESH_;
  iSetup.get<SiStripQualityRcd>().get(dataLabel_, SiStripQualityESH_);
  edm::LogInfo("testSiStripQualityESProducer")
      << canvas << "[testSiStripQualityESProducer::analyze] Print SiStripQualityRecord" << canvas << std::endl;
  printObject(SiStripQualityESH_.product());

  //&&&&&&&&&&&&&&&&&&
  //Second Record
  //&&&&&&&&&&&&&&&&&&
  edm::ESHandle<SiStripQuality> twoSiStripQualityESH_;
  if (twoRecordComparison_) {
    iSetup.get<SiStripQualityRcd>().get(dataLabelTwo_, twoSiStripQualityESH_);
    edm::LogInfo("testSiStripQualityESProducer")
        << canvas << "[testSiStripQualityESProducer::analyze] Print Second SiStripQualityRecord" << canvas << std::endl;
    printObject(twoSiStripQualityESH_.product());

    edm::LogInfo("testSiStripQualityESProducer")
        << canvas << "[testSiStripQualityESProducer::analyze] Print difference (First - Second) SiStripQuality Rcd"
        << canvas << std::endl;
    const SiStripQuality& tmp1 = SiStripQualityESH_->difference(*twoSiStripQualityESH_);
    printObject(&tmp1);
    edm::LogInfo("testSiStripQualityESProducer")
        << canvas << "[testSiStripQualityESProducer::analyze] Print difference (Second - First) SiStripQuality Rcd"
        << canvas << std::endl;
    const SiStripQuality& tmp2 = twoSiStripQualityESH_->difference(*SiStripQualityESH_);
    printObject(&tmp2);
  }

  if (!firstIOV) {
    edm::LogInfo("testSiStripQualityESProducer")
        << canvas << "[testSiStripQualityESProducer::analyze] Print previous SiStripQuality Rcd " << canvas
        << std::endl;
    printObject(m_Quality_);
    edm::LogInfo("testSiStripQualityESProducer")
        << canvas << "[testSiStripQualityESProducer::analyze] Print difference (Current - Previous) SiStripQuality Rcd"
        << canvas << std::endl;
    const SiStripQuality& tmp1 = SiStripQualityESH_->difference(*m_Quality_);
    printObject(&tmp1);
    edm::LogInfo("testSiStripQualityESProducer")
        << canvas << "[testSiStripQualityESProducer::analyze] Print difference (Previous - Current) SiStripQuality Rcd"
        << canvas << std::endl;
    const SiStripQuality& tmp2 = m_Quality_->difference(*SiStripQualityESH_);
    printObject(&tmp2);
  }

  firstIOV = false;
  edm::LogInfo("testSiStripQualityESProducer")
      << "[testSiStripQualityESProducer::analyze] Constructing new object " << std::endl;
  m_Quality_ = new SiStripQuality(*SiStripQualityESH_);
}

void testSiStripQualityESProducer::printObject(const SiStripQuality* SiStripQuality_) {
  std::stringstream ss;

  SiStripBadStrip::RegistryIterator rbegin = SiStripQuality_->getRegistryVectorBegin();
  SiStripBadStrip::RegistryIterator rend = SiStripQuality_->getRegistryVectorEnd();
  uint32_t detid;

  if (rbegin == rend)
    return;

  for (SiStripBadStrip::RegistryIterator rp = rbegin; rp != rend; ++rp) {
    detid = rp->detid;
    SiStripBadStrip::Range range = SiStripBadStrip::Range(SiStripQuality_->getDataVectorBegin() + rp->ibegin,
                                                          SiStripQuality_->getDataVectorBegin() + rp->iend);

    SiStripBadStrip::ContainerIterator it = range.first;
    ss << "Full Info";

    for (; it != range.second; ++it) {
      unsigned int value = (*it);

      ss << "\n\tdetid " << detid << " \t"
         << " firstBadStrip " << SiStripQuality_->decode(value).firstStrip << "\t "
         << " NconsecutiveBadStrips " << SiStripQuality_->decode(value).range << "\t "
         << " flag " << SiStripQuality_->decode(value).flag << "\t "
         << " packed integer 0x" << std::hex << value << std::dec << "\t ";
    }

    ss << "\n\nDetBase Info\n\t  IsModuleBad()=" << SiStripQuality_->IsModuleBad(detid)
       << "\t IsFiberBad(n)=" << SiStripQuality_->IsFiberBad(detid, 0) << " " << SiStripQuality_->IsFiberBad(detid, 1)
       << " " << SiStripQuality_->IsFiberBad(detid, 2) << "\t getBadFibers()=" << SiStripQuality_->getBadFibers(detid)
       << "\t IsApvBad()=" << SiStripQuality_->IsApvBad(detid, 0) << " " << SiStripQuality_->IsApvBad(detid, 1) << " "
       << SiStripQuality_->IsApvBad(detid, 2) << " " << SiStripQuality_->IsApvBad(detid, 3) << " "
       << SiStripQuality_->IsApvBad(detid, 4) << " " << SiStripQuality_->IsApvBad(detid, 5)
       << "\t getBadApvs()=" << SiStripQuality_->getBadApvs(detid);
    ss << "\n--------------------------------------------------\n";
  }
  ss << "\nGlobal Info";
  std::vector<SiStripQuality::BadComponent> BC = SiStripQuality_->getBadComponentList();
  ss << "\n\t detid \t IsModuleBad \t BadFibers \t BadApvs";

  for (size_t i = 0; i < BC.size(); ++i)
    ss << "\n\t" << BC[i].detid << "\t " << BC[i].BadModule << "\t " << BC[i].BadFibers << "\t " << BC[i].BadApvs
       << "\t ";
  ss << std::endl;

  edm::LogInfo("testSiStripQualityESProducer") << ss.str();
}
