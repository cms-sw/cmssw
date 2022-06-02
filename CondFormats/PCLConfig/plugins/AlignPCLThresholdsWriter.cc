// -*- C++ -*-
//
// Package:    CondFormats/PCLConfig
// Class:      AlignPCLThresholdsWriter
//
/**\class AlignPCLThresholdsWriter AlignPCLThresholdsWriter.cc CondFormats/PCLConfig/plugins/AlignPCLThresholdsWriter.cc

 Description: class to build the SiPixelAli PCL thresholds

*/
//
// Original Author:  Marco Musich
//         Created:  Wed, 22 Feb 2017 12:04:36 GMT
//
//

// system include files
#include <array>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/PCLConfig/interface/AlignPCLThresholds.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

//
// class declaration
//

namespace DOFs {
  enum dof { X, Y, Z, thetaX, thetaY, thetaZ, extraDOF };
}

class AlignPCLThresholdsWriter : public edm::one::EDAnalyzer<> {
public:
  explicit AlignPCLThresholdsWriter(const edm::ParameterSet&);
  ~AlignPCLThresholdsWriter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  DOFs::dof mapOntoEnum(std::string coord);

  // ----------member data ---------------------------
  const std::string m_record;
  const unsigned int m_minNrecords;
  const std::vector<edm::ParameterSet> m_parameters;
};

//
// constructors and destructor
//
AlignPCLThresholdsWriter::AlignPCLThresholdsWriter(const edm::ParameterSet& iConfig)
    : m_record(iConfig.getParameter<std::string>("record")),
      m_minNrecords(iConfig.getParameter<unsigned int>("minNRecords")),
      m_parameters(iConfig.getParameter<std::vector<edm::ParameterSet> >("thresholds")) {}

//
// member functions
//

// ------------ method called for each event  ------------
void AlignPCLThresholdsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // output object
  AlignPCLThresholds myThresholds{};

  edm::LogInfo("AlignPCLThresholdsWriter") << "Size of AlignPCLThresholds object " << myThresholds.size() << std::endl;

  // loop on the PSet and insert the conditions
  std::array<std::string, 6> mandatories = {{"X", "Y", "Z", "thetaX", "thetaY", "thetaZ"}};
  std::vector<std::string> alignables;

  // fill the list of alignables
  for (auto& thePSet : m_parameters) {
    const std::string alignableId(thePSet.getParameter<std::string>("alignableId"));
    // only if it is not yet in the list
    if (std::find(alignables.begin(), alignables.end(), alignableId) == alignables.end()) {
      alignables.push_back(alignableId);
    }
  }

  for (auto& alignable : alignables) {
    AlignPCLThreshold::coordThresholds my_X;
    AlignPCLThreshold::coordThresholds my_Y;
    AlignPCLThreshold::coordThresholds my_Z;
    AlignPCLThreshold::coordThresholds my_tX;
    AlignPCLThreshold::coordThresholds my_tY;
    AlignPCLThreshold::coordThresholds my_tZ;

    std::vector<std::string> presentDOF;

    // extra degrees of freedom
    std::vector<AlignPCLThreshold::coordThresholds> extraDOFs = std::vector<AlignPCLThreshold::coordThresholds>();

    for (auto& thePSet : m_parameters) {
      const std::string alignableId(thePSet.getParameter<std::string>("alignableId"));
      const std::string DOF(thePSet.getParameter<std::string>("DOF"));

      const double cutoff(thePSet.getParameter<double>("cut"));
      const double sigCut(thePSet.getParameter<double>("sigCut"));
      const double maxMoveCut(thePSet.getParameter<double>("maxMoveCut"));
      const double maxErrorCut(thePSet.getParameter<double>("maxErrorCut"));

      if (alignableId == alignable) {
        presentDOF.push_back(DOF);
        // create the objects

        switch (mapOntoEnum(DOF)) {
          case DOFs::X:
            my_X.setThresholds(cutoff, sigCut, maxErrorCut, maxMoveCut, DOF);
            break;
          case DOFs::Y:
            my_Y.setThresholds(cutoff, sigCut, maxErrorCut, maxMoveCut, DOF);
            break;
          case DOFs::Z:
            my_Z.setThresholds(cutoff, sigCut, maxErrorCut, maxMoveCut, DOF);
            break;
          case DOFs::thetaX:
            my_tX.setThresholds(cutoff, sigCut, maxErrorCut, maxMoveCut, DOF);
            break;
          case DOFs::thetaY:
            my_tY.setThresholds(cutoff, sigCut, maxErrorCut, maxMoveCut, DOF);
            break;
          case DOFs::thetaZ:
            my_tZ.setThresholds(cutoff, sigCut, maxErrorCut, maxMoveCut, DOF);
            break;
          default:
            edm::LogInfo("AlignPCLThresholdsWriter")
                << "Appending Extra degree of freeedom: " << DOF << " " << mapOntoEnum(DOF) << std::endl;
            AlignPCLThreshold::coordThresholds ExtraDOF;
            ExtraDOF.setThresholds(cutoff, sigCut, maxErrorCut, maxMoveCut, DOF);
            extraDOFs.push_back(ExtraDOF);
        }

        AlignPCLThreshold a(my_X, my_tX, my_Y, my_tY, my_Z, my_tZ, extraDOFs);
        myThresholds.setAlignPCLThreshold(alignableId, a);

      }  // if alignable is found in the PSet
    }    // loop on the PSets

    // checks if all mandatories are present
    edm::LogInfo("AlignPCLThresholdsWriter")
        << "Size of AlignPCLThresholds object  " << myThresholds.size() << std::endl;
    for (auto& mandatory : mandatories) {
      if (std::find(presentDOF.begin(), presentDOF.end(), mandatory) == presentDOF.end()) {
        edm::LogWarning("AlignPCLThresholdsWriter")
            << "Configuration for DOF: " << mandatory << " for alignable " << alignable << "is not present \n"
            << "Will build object with defaults!" << std::endl;
      }
    }

  }  // ends loop on the alignable units

  // set the minimum number of records to be used in pede
  myThresholds.setNRecords(m_minNrecords);
  edm::LogInfo("AlignPCLThresholdsWriter") << "Content of AlignPCLThresholds " << std::endl;

  // use buil-in method in the CondFormat
  myThresholds.printAll();

  // Form the data here
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    cond::Time_t valid_time = poolDbService->currentTime();
    // this writes the payload to begin in current run defined in cfg
    poolDbService->writeOneIOV(myThresholds, valid_time, m_record);
  }
}

DOFs::dof AlignPCLThresholdsWriter::mapOntoEnum(std::string coord) {
  if (coord == "X") {
    return DOFs::X;
  } else if (coord == "Y") {
    return DOFs::Y;
  } else if (coord == "Z") {
    return DOFs::Z;
  } else if (coord == "thetaX") {
    return DOFs::thetaX;
  } else if (coord == "thetaY") {
    return DOFs::thetaY;
  } else if (coord == "thetaZ") {
    return DOFs::thetaZ;
  } else {
    return DOFs::extraDOF;
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void AlignPCLThresholdsWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Plugin to write payloads of type AlignPCLThresholds");
  desc.add<std::string>("record", "AlignPCLThresholdsRcd");
  desc.add<unsigned int>("minNRecords", 25000);
  edm::ParameterSetDescription desc_thresholds;

  desc_thresholds.add<std::string>("alignableId");
  desc_thresholds.add<std::string>("DOF");
  desc_thresholds.add<double>("cut");
  desc_thresholds.add<double>("sigCut");
  desc_thresholds.add<double>("maxMoveCut");
  desc_thresholds.add<double>("maxErrorCut");

  std::vector<edm::ParameterSet> default_thresholds(1);
  desc.addVPSet("thresholds", desc_thresholds, default_thresholds);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlignPCLThresholdsWriter);
