// system include files
#include <memory>
#include <cstdio>
#include <string>

// user include files
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class SiStripBadStripFromASCIIFile : public ConditionDBWriter<SiStripBadStrip> {
public:
  explicit SiStripBadStripFromASCIIFile(const edm::ParameterSet& iConfig);
  ~SiStripBadStripFromASCIIFile() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  std::unique_ptr<SiStripBadStrip> getNewObject() override;
  const bool printdebug_;
  const bool isFlagAvailable_;
  const edm::FileInPath fp_;
};

using namespace std;
SiStripBadStripFromASCIIFile::SiStripBadStripFromASCIIFile(const edm::ParameterSet& iConfig)
    : ConditionDBWriter<SiStripBadStrip>(iConfig),
      printdebug_(iConfig.getParameter<bool>("printDebug")),
      isFlagAvailable_(iConfig.getParameter<bool>("isFlagAvailable")),
      fp_(iConfig.getParameter<edm::FileInPath>("file")) {}

void SiStripBadStripFromASCIIFile::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  ConditionDBWriter::fillPSetDescription(desc);  // inherited from mother class
  desc.setComment("Conditions Builder for SiStripBadStrip Objects from input ASCII file");
  desc.add<edm::FileInPath>("file", edm::FileInPath("CondTools/SiStrip/data/DefectsFromConstructionDB.dat"));
  desc.add<bool>("printDebug", false)->setComment("prints debug level messages");
  desc.add<bool>("isFlagAvailable", true)->setComment("is the flag available in the input file");
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<SiStripBadStrip> SiStripBadStripFromASCIIFile::getNewObject() {
  auto SiStripBadStrip_ = std::make_unique<SiStripBadStrip>();

  // open file and fill DB
  ifstream infile((fp_.fullPath()).c_str());
  if (!infile) {
    edm::LogError("SiStripBadStripFromASCIIFile") << "[SiStripBadStripFromASCIIFile::GetNewObject]"
                                                  << " Problem while trying to open File: " << (fp_.fullPath()).c_str();
  }

  //variables needed for reading file and filling of SiStripBadStripObject
  uint32_t detid;
  short flag;
  short channel;

  bool firstrun = true;
  short tempchannel = 0;
  int count = 0;
  std::vector<unsigned int> theSiStripVector;
  short tempflag = 0;
  uint32_t tempdetid = 0;

  while (!infile.eof()) {
    // get data from file:
    if (isFlagAvailable_) {
      infile >> detid >> channel >> flag;
    } else {
      //if no flag is available, use the following:
      infile >> detid >> channel;
      flag = 1;
    }

    unsigned int theBadStripRange = 0;

    // first loop ?
    if (firstrun) {
      tempdetid = detid;
      tempchannel = channel;
      tempflag = flag;
      count = 0;
      firstrun = false;
    }

    if (detid == tempdetid) {
      if (channel != tempchannel + count || flag != tempflag) {
        // 1.badstrip, nconsectrips, flag
        theBadStripRange = SiStripBadStrip_->encode(
            tempchannel - 1,
            count,
            tempflag);  // In the quality object, strips are counted from 0 to 767!!! Therefore "tempchannel-1"!
                        // In the input txt-file, they have to be from 1 to 768 instead!!!

        if (printdebug_) {
          edm::LogInfo("SiStripBadStripFromASCIIFile")
              << "detid " << tempdetid << " \t"
              << " firstBadStrip " << tempchannel << "\t "
              << " NconsecutiveBadStrips " << count << "\t "
              << "flag " << tempflag << "\t"
              << " packed integer " << std::hex << theBadStripRange << std::dec;
        }

        theSiStripVector.push_back(theBadStripRange);

        if (infile.eof()) {  // Don't forget to save the last strip before eof!!!
          SiStripBadStrip::Range range(theSiStripVector.begin(), theSiStripVector.end());
          if (!SiStripBadStrip_->put(tempdetid, range))
            edm::LogError("SiStripBadStripFromASCIIFile")
                << "[SiStripBadStripFromASCIIFile::GetNewObject] detid already exists";
          theSiStripVector.clear();
        }

        count = 1;
        tempchannel = channel;
        tempflag = flag;

      } else {
        count++;
      }
    }

    if (detid != tempdetid) {
      // 1.badstrip, nconsectrips, flag
      theBadStripRange = SiStripBadStrip_->encode(
          tempchannel - 1,
          count,
          tempflag);  // In the quality object, strips are counted from 0 to 767!!! Therefore "tempchannel-1"!
                      // In the input txt-file, they have to be from 1 to 768 instead!!!
      if (printdebug_) {
        edm::LogInfo("SiStripBadStripFromASCIIFile") << "detid " << tempdetid << " \t"
                                                     << " firstBadStrip " << tempchannel << "\t "
                                                     << " NconsecutiveBadStrips " << count << "\t "
                                                     << "flag " << tempflag << "\t"
                                                     << " packed integer " << std::hex << theBadStripRange << std::dec;
      }

      theSiStripVector.push_back(theBadStripRange);

      // populate db  object
      SiStripBadStrip::Range range(theSiStripVector.begin(), theSiStripVector.end());
      if (!SiStripBadStrip_->put(tempdetid, range))
        edm::LogError("SiStripBadStripFromASCIIFile")
            << "[SiStripBadStripFromASCIIFile::GetNewObject] detid already exists";
      theSiStripVector.clear();

      count = 1;
      tempdetid = detid;
      tempchannel = channel;
      tempflag = flag;
    }
  }

  return SiStripBadStrip_;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripBadStripFromASCIIFile);
