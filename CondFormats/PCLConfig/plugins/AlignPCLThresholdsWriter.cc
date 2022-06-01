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
#include "CondFormats/PCLConfig/interface/AlignPCLThresholdsHG.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

//
// class declaration
//

namespace DOFs {
  enum dof { X, Y, Z, thetaX, thetaY, thetaZ, extraDOF };
}

template <typename T>
class AlignPCLThresholdsWriter : public edm::one::EDAnalyzer<> {
public:
  explicit AlignPCLThresholdsWriter(const edm::ParameterSet&);
  ~AlignPCLThresholdsWriter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  DOFs::dof mapOntoEnum(std::string coord);

  void writePayload(T& myThresholds);
  void storeHGthresholds(AlignPCLThresholdsHG& myThresholds, const std::vector<std::string>& alignables);

  // ----------member data ---------------------------
  const std::string m_record;
  const unsigned int m_minNrecords;
  const std::vector<edm::ParameterSet> m_parameters;
};

//
// constructors and destructor
//
template <typename T>
AlignPCLThresholdsWriter<T>::AlignPCLThresholdsWriter(const edm::ParameterSet& iConfig)
    : m_record(iConfig.getParameter<std::string>("record")),
      m_minNrecords(iConfig.getParameter<unsigned int>("minNRecords")),
      m_parameters(iConfig.getParameter<std::vector<edm::ParameterSet> >("thresholds")) {}

//
// member functions
//

// ------------ method called for each event  ------------
template <typename T>
void AlignPCLThresholdsWriter<T>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // detect if new payload is used
  bool newClass = false;
  for (auto& thePSet : m_parameters) {
    if (thePSet.exists("fractionCut")) {
      newClass = true;
      break;
    }
  }

  T myThresholds{};
  if constexpr (std::is_same_v<T, AlignPCLThresholdsHG>) {
    if (newClass) {
      this->writePayload(myThresholds);
    } else {
      throw cms::Exception("AlignPCLThresholdsWriter") << "mismatched configuration";
    }
  } else {
    if (!newClass) {
      this->writePayload(myThresholds);
    } else {
      throw cms::Exception("AlignPCLThresholdsWriter") << "mismatched configuration";
    }
  }
}

template <typename T>
DOFs::dof AlignPCLThresholdsWriter<T>::mapOntoEnum(std::string coord) {
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

// ------------ templated method to write the payload  ------------
template <typename T>
void AlignPCLThresholdsWriter<T>::writePayload(T& myThresholds) {
  using namespace edm;

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

  // additional thresholds for AlignPCLThresholdsHG
  if constexpr (std::is_same_v<T, AlignPCLThresholdsHG>) {
    storeHGthresholds(myThresholds, alignables);
  }

  // use built-in method in the CondFormat
  myThresholds.printAll();

  // Form the data here
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    cond::Time_t valid_time = poolDbService->currentTime();
    // this writes the payload to begin in current run defined in cfg
    poolDbService->writeOneIOV(myThresholds, valid_time, m_record);
  }
}

// ------------ method to store additional HG thresholds ------------
template <typename T>
void AlignPCLThresholdsWriter<T>::storeHGthresholds(AlignPCLThresholdsHG& myThresholds,
                                                    const std::vector<std::string>& alignables) {
  edm::LogInfo("AlignPCLThresholdsWriter")
      << "Found type AlignPCLThresholdsHG, additional thresholds are written" << std::endl;

  for (auto& alignable : alignables) {
    for (auto& thePSet : m_parameters) {
      const std::string alignableId(thePSet.getParameter<std::string>("alignableId"));
      const std::string DOF(thePSet.getParameter<std::string>("DOF"));

      // Get coordType from DOF
      AlignPCLThresholds::coordType type = static_cast<AlignPCLThresholds::coordType>(mapOntoEnum(DOF));

      if (alignableId == alignable) {
        if (thePSet.exists("fractionCut")) {
          const double fractionCut(thePSet.getParameter<double>("fractionCut"));
          myThresholds.setFractionCut(alignableId, type, fractionCut);
        }
      }
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename T>
void AlignPCLThresholdsWriter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Plugin to write payloads of type AlignPCLThresholds");
  desc.add<unsigned int>("minNRecords", 25000);
  edm::ParameterSetDescription desc_thresholds;

  desc_thresholds.add<std::string>("alignableId");
  desc_thresholds.add<std::string>("DOF");
  desc_thresholds.add<double>("cut");
  desc_thresholds.add<double>("sigCut");
  desc_thresholds.add<double>("maxMoveCut");
  desc_thresholds.add<double>("maxErrorCut");
  if constexpr (std::is_same_v<T, AlignPCLThresholdsHG>) {
    desc.add<std::string>("record", "AlignPCLThresholdsHGRcd");
    //optional thresholds from new payload version (not for all the alignables)
    desc_thresholds.addOptional<double>("fractionCut");
  } else {
    desc.add<std::string>("record", "AlignPCLThresholdsRcd");
  }

  std::vector<edm::ParameterSet> default_thresholds(1);
  desc.addVPSet("thresholds", desc_thresholds, default_thresholds);
  descriptions.addWithDefaultLabel(desc);
}

typedef AlignPCLThresholdsWriter<AlignPCLThresholds> AlignPCLThresholdsLGWriter;
typedef AlignPCLThresholdsWriter<AlignPCLThresholdsHG> AlignPCLThresholdsHGWriter;

DEFINE_FWK_MODULE(AlignPCLThresholdsLGWriter);
DEFINE_FWK_MODULE(AlignPCLThresholdsHGWriter);
