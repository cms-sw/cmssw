#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/OpticalAlignmentsRcd.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurementInfo.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaFit/interface/Fit.h"
#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaUtilities/interface/ALIFileOut.h"
#include "Alignment/CocoaModel/interface/CocoaDaqReaderRoot.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include "Alignment/CocoaFit/interface/CocoaDBMgr.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurements.h"

class CocoaAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit CocoaAnalyzer(edm::ParameterSet const& p);
  explicit CocoaAnalyzer(int i) {}
  ~CocoaAnalyzer() override {}

  void beginJob() override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  void readXMLFile(const edm::EventSetup& evts);

  std::vector<OpticalAlignInfo> readCalibrationDB(const edm::EventSetup& evts);
  void correctAllOpticalAlignments(std::vector<OpticalAlignInfo>& allDBOpticalAlignments);
  void correctOpticalAlignmentParameter(OpticalAlignParam& myXMLParam, const OpticalAlignParam& myDBParam);

  void runCocoa();

private:
  OpticalAlignments oaList_;
  OpticalAlignMeasurements measList_;
  std::string theCocoaDaqRootFileName_;
};

using namespace cms_units::operators;

CocoaAnalyzer::CocoaAnalyzer(edm::ParameterSet const& pset) {
  theCocoaDaqRootFileName_ = pset.getParameter<std::string>("cocoaDaqRootFile");
  int maxEvents = pset.getParameter<int32_t>("maxEvents");
  GlobalOptionMgr::getInstance()->setDefaultGlobalOptions();
  GlobalOptionMgr::getInstance()->setGlobalOption("maxEvents", maxEvents);
  GlobalOptionMgr::getInstance()->setGlobalOption("writeDBAlign", 1);
  GlobalOptionMgr::getInstance()->setGlobalOption("writeDBOptAlign", 1);
  usesResource("CocoaAnalyzer");
}

void CocoaAnalyzer::beginJob() {}

void CocoaAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& evts) {
  ALIUtils::setDebugVerbosity(5);

  // Get ideal geometry description + measurements for simulation.
  readXMLFile(evts);

  // Correct ideal geometry with data from DB.
  std::vector<OpticalAlignInfo> oaListCalib = readCalibrationDB(evts);
  correctAllOpticalAlignments(oaListCalib);

  // Run the least-squared fit and store results in DB.
  runCocoa();
}

/*
 * This is used to create the ideal geometry description from the XMLs.
 * Also get measurements from XMLs for simulation.
 * Resulting optical alignment info is stored in oaList_ and measList_.
 */
void CocoaAnalyzer::readXMLFile(const edm::EventSetup& evts) {
  edm::ESTransientHandle<cms::DDCompactView> myCompactView;
  evts.get<IdealGeometryRecord>().get(myCompactView);

  const cms::DDDetector* mySystem = myCompactView->detector();

  if (mySystem) {
    // Always store world volume first.
    const dd4hep::Volume& worldVolume = mySystem->worldVolume();

    if (ALIUtils::debug >= 3) {
      edm::LogInfo("Alignment") << "CocoaAnalyzer::ReadXML: world object = " << worldVolume.name();
    }

    OpticalAlignInfo worldInfo;
    worldInfo.ID_ = 0;
    worldInfo.name_ = worldVolume.name();
    worldInfo.type_ = "system";
    worldInfo.parentName_ = "";
    worldInfo.x_.value_ = 0.;
    worldInfo.x_.error_ = 0.;
    worldInfo.x_.quality_ = 0;
    worldInfo.y_.value_ = 0.;
    worldInfo.y_.error_ = 0.;
    worldInfo.y_.quality_ = 0;
    worldInfo.z_.value_ = 0.;
    worldInfo.z_.error_ = 0.;
    worldInfo.z_.quality_ = 0;
    worldInfo.angx_.value_ = 0.;
    worldInfo.angx_.error_ = 0.;
    worldInfo.angx_.quality_ = 0;
    worldInfo.angy_.value_ = 0.;
    worldInfo.angy_.error_ = 0.;
    worldInfo.angy_.quality_ = 0;
    worldInfo.angz_.value_ = 0.;
    worldInfo.angz_.error_ = 0.;
    worldInfo.angz_.quality_ = 0;
    oaList_.opticalAlignments_.emplace_back(worldInfo);

    // This gathers all the 'SpecPar' sections from the loaded XMLs.
    // NB: Definition of a SpecPar section:
    // It is a block in the XML file(s), containing paths to specific volumes,
    // and ALLOWING THE ASSOCIATION OF SPECIFIC PARAMETERS AND VALUES TO THESE VOLUMES.
    const cms::DDSpecParRegistry& allSpecParSections = myCompactView->specpars();

    // CREATION OF A COCOA FILTERED VIEW
    // Creation of the dd4hep-based filtered view.
    // NB: not filtered yet!
    cms::DDFilteredView myFilteredView(mySystem, worldVolume);
    // Declare a container which will gather all the filtered SpecPar sections.
    cms::DDSpecParRefs cocoaParameterSpecParSections;
    // Define a COCOA filter
    const std::string cocoaParameterAttribute = "COCOA";
    const std::string cocoaParameterValue = "COCOA";
    // All the COCOA SpecPar sections are filtered from allSpecParSections,
    // and assigned to cocoaParameterSpecParSections.
    allSpecParSections.filter(cocoaParameterSpecParSections, cocoaParameterAttribute, cocoaParameterValue);
    // This finally allows to filter the filtered view, with the COCOA filter.
    // This means that we now have, in myFilteredView, all volumes whose paths were selected:
    // ie all volumes with "COCOA" parameter and value in a SpecPar section from a loaded XML.
    myFilteredView.mergedSpecifics(cocoaParameterSpecParSections);

    // Loop on parts
    int nObjects = 0;
    bool doCOCOA = myFilteredView.firstChild();

    // Loop on all COCOA volumes from filtered view
    while (doCOCOA) {
      ++nObjects;

      OpticalAlignInfo oaInfo;
      OpticalAlignParam oaParam;
      OpticalAlignMeasurementInfo oaMeas;

      // Current volume
      const dd4hep::PlacedVolume& myPlacedVolume = myFilteredView.volume();
      const std::string& name = myPlacedVolume.name();
      const std::string& nodePath = myFilteredView.path();
      oaInfo.name_ = nodePath;

      // Parent name
      oaInfo.parentName_ = nodePath.substr(0, nodePath.rfind('/', nodePath.length()));

      if (ALIUtils::debug >= 4) {
        edm::LogInfo("Alignment") << " CocoaAnalyzer::ReadXML reading object " << name;
        edm::LogInfo("Alignment") << " @@ Name built= " << oaInfo.name_ << " short_name= " << name
                                  << " parent= " << oaInfo.parentName_;
      }

      // TRANSLATIONS

      // A) GET TRANSLATIONS FROM DDETECTOR.
      // Directly get translation from parent to child volume
      const dd4hep::Direction& transl = myPlacedVolume.position();

      if (ALIUtils::debug >= 4) {
        edm::LogInfo("Alignment") << "Local translation in cm = " << transl;
      }

      // B) READ INFO FROM XMLS
      // X
      oaInfo.x_.name_ = "X";
      oaInfo.x_.dim_type_ = "centre";
      oaInfo.x_.value_ = transl.x() / (1._m);  // COCOA units are m
      oaInfo.x_.error_ = cms::getParameterValueFromSpecParSections<double>(allSpecParSections,
                                                                           nodePath,
                                                                           "centre_X_sigma",
                                                                           0) /
                         (1._m);  // COCOA units are m
      oaInfo.x_.quality_ = static_cast<int>(
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "centre_X_quality", 0));
      // Y
      oaInfo.y_.name_ = "Y";
      oaInfo.y_.dim_type_ = "centre";
      oaInfo.y_.value_ = transl.y() / (1._m);  // COCOA units are m
      oaInfo.y_.error_ = cms::getParameterValueFromSpecParSections<double>(allSpecParSections,
                                                                           nodePath,
                                                                           "centre_Y_sigma",
                                                                           0) /
                         (1._m);  // COCOA units are m
      oaInfo.y_.quality_ = static_cast<int>(
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "centre_Y_quality", 0));
      // Z
      oaInfo.z_.name_ = "Z";
      oaInfo.z_.dim_type_ = "centre";
      oaInfo.z_.value_ = transl.z() / (1._m);  // COCOA units are m
      oaInfo.z_.error_ = cms::getParameterValueFromSpecParSections<double>(allSpecParSections,
                                                                           nodePath,
                                                                           "centre_Z_sigma",
                                                                           0) /
                         (1._m);  // COCOA units are m
      oaInfo.z_.quality_ = static_cast<int>(
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "centre_Z_quality", 0));

      // ROTATIONS

      // A) GET ROTATIONS FROM DDETECTOR.

      // Unlike in the initial code, here we manage to directly get the rotation matrix placement
      // of the child in parent, EXPRESSED IN THE PARENT FRAME OF REFERENCE.
      // Hence the (ugly) initial block of code is replaced by just 2 lines.
      // PlacedVolume::matrix() returns the rotation matrix IN THE PARENT FRAME OF REFERENCE.
      // NB: Not using DDFilteredView::rotation(),
      // because it returns the rotation matrix IN THE WORLD FRAME OF REFERENCE.
      const TGeoHMatrix parentToChild = myPlacedVolume.matrix();
      // COCOA convention is FROM CHILD TO PARENT
      const TGeoHMatrix& childToParent = parentToChild.Inverse();

      // Convert it to CLHEP::Matrix
      // Below is not my code, below block is untouched (apart from bug fix).
      // I would just directly use childToParent...
      const Double_t* rot = childToParent.GetRotationMatrix();
      const double xx = rot[0];
      const double xy = rot[1];
      const double xz = rot[2];
      const double yx = rot[3];
      const double yy = rot[4];
      const double yz = rot[5];
      const double zx = rot[6];
      const double zy = rot[7];
      const double zz = rot[8];
      if (ALIUtils::debug >= 4) {
        edm::LogInfo("Alignment") << "Local rotation = ";
        edm::LogInfo("Alignment") << xx << "  " << xy << "  " << xz;
        edm::LogInfo("Alignment") << yx << "  " << yy << "  " << yz;
        edm::LogInfo("Alignment") << zx << "  " << zy << "  " << zz;
      }
      const CLHEP::Hep3Vector colX(xx, yx, zx);
      const CLHEP::Hep3Vector colY(xy, yy, zy);
      const CLHEP::Hep3Vector colZ(xz, yz, zz);
      const CLHEP::HepRotation rotclhep(colX, colY, colZ);
      const std::vector<double>& angles = ALIUtils::getRotationAnglesFromMatrix(rotclhep, 0., 0., 0.);

      // B) READ INFO FROM XMLS
      // X
      oaInfo.angx_.name_ = "X";
      oaInfo.angx_.dim_type_ = "angles";
      oaInfo.angx_.value_ =
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "angles_X_value", 0);
      oaInfo.angx_.error_ =
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "angles_X_sigma", 0);
      oaInfo.angx_.quality_ = static_cast<int>(
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "angles_X_quality", 0));
      // Y
      oaInfo.angy_.name_ = "Y";
      oaInfo.angy_.dim_type_ = "angles";
      oaInfo.angy_.value_ =
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "angles_Y_value", 0);
      oaInfo.angy_.error_ =
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "angles_Y_sigma", 0);
      oaInfo.angy_.quality_ = static_cast<int>(
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "angles_Y_quality", 0));
      // Z
      oaInfo.angz_.name_ = "Z";
      oaInfo.angz_.dim_type_ = "angles";
      oaInfo.angz_.value_ =
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "angles_Z_value", 0);
      oaInfo.angz_.error_ =
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "angles_Z_sigma", 0);
      oaInfo.angz_.quality_ = static_cast<int>(
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "angles_Z_quality", 0));

      oaInfo.type_ =
          cms::getParameterValueFromSpecParSections<std::string>(allSpecParSections, nodePath, "cocoa_type", 0);

      oaInfo.ID_ = static_cast<int>(
          cms::getParameterValueFromSpecParSections<double>(allSpecParSections, nodePath, "cmssw_ID", 0));

      if (ALIUtils::debug >= 4) {
        edm::LogInfo("Alignment") << "CocoaAnalyzer::ReadXML OBJECT " << oaInfo.name_ << " pos/angles read ";
      }

      // Check that rotations match with values from XMLs.
      // Same, that ugly code is not mine ;p
      if (fabs(oaInfo.angx_.value_ - angles[0]) > 1.E-9 || fabs(oaInfo.angy_.value_ - angles[1]) > 1.E-9 ||
          fabs(oaInfo.angz_.value_ - angles[2]) > 1.E-9) {
        edm::LogError("Alignment") << " WRONG ANGLE IN OBJECT " << oaInfo.name_ << oaInfo.angx_.value_ << " =? "
                                   << angles[0] << oaInfo.angy_.value_ << " =? " << angles[1] << oaInfo.angz_.value_
                                   << " =? " << angles[2];
      }

      // EXTRA PARAM ENTRIES (FROM XMLS)
      // Here initial code to define the containers was fully removed, this is much more compact.
      const std::vector<std::string>& names =
          cms::getAllParameterValuesFromSpecParSections<std::string>(allSpecParSections, nodePath, "extra_entry");
      const std::vector<std::string>& dims =
          cms::getAllParameterValuesFromSpecParSections<std::string>(allSpecParSections, nodePath, "dimType");
      const std::vector<double>& values =
          cms::getAllParameterValuesFromSpecParSections<double>(allSpecParSections, nodePath, "value");
      const std::vector<double>& errors =
          cms::getAllParameterValuesFromSpecParSections<double>(allSpecParSections, nodePath, "sigma");
      const std::vector<double>& quality =
          cms::getAllParameterValuesFromSpecParSections<double>(allSpecParSections, nodePath, "quality");

      if (ALIUtils::debug >= 4) {
        edm::LogInfo("Alignment") << " CocoaAnalyzer::ReadXML:  Fill extra entries with read parameters ";
      }

      if (names.size() == dims.size() && dims.size() == values.size() && values.size() == errors.size() &&
          errors.size() == quality.size()) {
        for (size_t i = 0; i < names.size(); ++i) {
          double dimFactor = 1.;
          const std::string& type = dims[i];
          if (type == "centre" || type == "length") {
            dimFactor = 1. / (1._m);  // was converted to cm with getParameterValueFromSpecPar, COCOA unit is m
          } else if (type == "angles" || type == "angle" || type == "nodim") {
            dimFactor = 1.;
          }
          oaParam.value_ = values[i] * dimFactor;
          oaParam.error_ = errors[i] * dimFactor;
          oaParam.quality_ = static_cast<int>(quality[i]);
          oaParam.name_ = names[i];
          oaParam.dim_type_ = dims[i];
          oaInfo.extraEntries_.emplace_back(oaParam);
          oaParam.clear();
        }

        oaList_.opticalAlignments_.emplace_back(oaInfo);
      } else {
        edm::LogInfo("Alignment") << "WARNING FOR NOW: sizes of extra parameters (names, dimType, value, quality) do"
                                  << " not match!  Did not add " << nObjects << " item to OpticalAlignments.";
      }

      // MEASUREMENTS (FROM XMLS)
      const std::vector<std::string>& measNames =
          cms::getAllParameterValuesFromSpecParSections<std::string>(allSpecParSections, nodePath, "meas_name");
      const std::vector<std::string>& measTypes =
          cms::getAllParameterValuesFromSpecParSections<std::string>(allSpecParSections, nodePath, "meas_type");

      std::map<std::string, std::vector<std::string>> measObjectNames;
      std::map<std::string, std::vector<std::string>> measParamNames;
      std::map<std::string, std::vector<double>> measParamValues;
      std::map<std::string, std::vector<double>> measParamSigmas;
      std::map<std::string, std::vector<double>> measIsSimulatedValue;
      for (const auto& name : measNames) {
        measObjectNames[name] = cms::getAllParameterValuesFromSpecParSections<std::string>(
            allSpecParSections, nodePath, "meas_object_name_" + name);
        measParamNames[name] = cms::getAllParameterValuesFromSpecParSections<std::string>(
            allSpecParSections, nodePath, "meas_value_name_" + name);
        measParamValues[name] =
            cms::getAllParameterValuesFromSpecParSections<double>(allSpecParSections, nodePath, "meas_value_" + name);
        measParamSigmas[name] =
            cms::getAllParameterValuesFromSpecParSections<double>(allSpecParSections, nodePath, "meas_sigma_" + name);
        measIsSimulatedValue[name] = cms::getAllParameterValuesFromSpecParSections<double>(
            allSpecParSections, nodePath, "meas_is_simulated_value_" + name);
      }

      if (ALIUtils::debug >= 4) {
        edm::LogInfo("Alignment") << " CocoaAnalyzer::ReadXML:  Fill measurements with read parameters ";
      }

      if (measNames.size() == measTypes.size()) {
        for (size_t i = 0; i < measNames.size(); ++i) {
          oaMeas.ID_ = i;
          oaMeas.name_ = measNames[i];
          oaMeas.type_ = measTypes[i];
          oaMeas.measObjectNames_ = measObjectNames[oaMeas.name_];
          if (measParamNames.size() == measParamValues.size() && measParamValues.size() == measParamSigmas.size()) {
            for (size_t i2 = 0; i2 < measParamNames[oaMeas.name_].size(); i2++) {
              oaParam.name_ = measParamNames[oaMeas.name_][i2];
              oaParam.value_ = measParamValues[oaMeas.name_][i2];
              oaParam.error_ = measParamSigmas[oaMeas.name_][i2];
              oaParam.quality_ = 2;
              if (oaMeas.type_ == "SENSOR2D" || oaMeas.type_ == "COPS" || oaMeas.type_ == "DISTANCEMETER" ||
                  oaMeas.type_ == "DISTANCEMETER!DIM" || oaMeas.type_ == "DISTANCEMETER3DIM") {
                oaParam.dim_type_ = "length";
              } else if (oaMeas.type_ == "TILTMETER") {
                oaParam.dim_type_ = "angle";
              } else {
                edm::LogError("Alignment") << "CocoaAnalyzer::readXMLFile. Invalid measurement type: " << oaMeas.type_;
              }

              oaMeas.values_.emplace_back(oaParam);
              oaMeas.isSimulatedValue_.emplace_back(measIsSimulatedValue[oaMeas.name_][i2]);
              if (ALIUtils::debug >= 5) {
                edm::LogInfo("Alignment") << oaMeas.name_ << " copying issimu "
                                          << oaMeas.isSimulatedValue_[oaMeas.isSimulatedValue_.size() - 1] << " = "
                                          << measIsSimulatedValue[oaMeas.name_][i2];
              }
              oaParam.clear();
            }
          } else {
            if (ALIUtils::debug >= 2) {
              edm::LogWarning("Alignment") << "WARNING FOR NOW: sizes of measurement parameters (name, value, sigma) do"
                                           << " not match! for measurement " << oaMeas.name_
                                           << " !Did not fill parameters for this measurement ";
            }
          }
          measList_.oaMeasurements_.emplace_back(oaMeas);
          if (ALIUtils::debug >= 5) {
            edm::LogInfo("Alignment") << "CocoaAnalyser: MEASUREMENT " << oaMeas.name_ << " extra entries read "
                                      << oaMeas;
          }
          oaMeas.clear();
        }

      } else {
        if (ALIUtils::debug >= 2) {
          edm::LogWarning("Alignment") << "WARNING FOR NOW: sizes of measurements (names, types do"
                                       << " not match!  Did not add " << nObjects << " item to XXXMeasurements";
        }
      }

      oaInfo.clear();
      doCOCOA = myFilteredView.firstChild();
    }  // while (doCOCOA)

    if (ALIUtils::debug >= 3) {
      edm::LogInfo("Alignment") << "CocoaAnalyzer::ReadXML: Finished building " << nObjects + 1
                                << " OpticalAlignInfo objects"
                                << " and " << measList_.oaMeasurements_.size()
                                << " OpticalAlignMeasurementInfo objects ";
    }
    if (ALIUtils::debug >= 5) {
      edm::LogInfo("Alignment") << " @@@@@@ OpticalAlignments " << oaList_;
      edm::LogInfo("Alignment") << " @@@@@@ OpticalMeasurements " << measList_;
    }
  }
}

/*
 * This is used to get the OpticalAlignInfo from DB, 
 * which can be used to correct the OpticalAlignInfo from IdealGeometry.
 */
std::vector<OpticalAlignInfo> CocoaAnalyzer::readCalibrationDB(const edm::EventSetup& evts) {
  if (ALIUtils::debug >= 3) {
    edm::LogInfo("Alignment") << "$$$ CocoaAnalyzer::readCalibrationDB: ";
  }

  using namespace edm::eventsetup;
  edm::ESHandle<OpticalAlignments> pObjs;
  evts.get<OpticalAlignmentsRcd>().get(pObjs);
  const std::vector<OpticalAlignInfo>& infoFromDB = pObjs.product()->opticalAlignments_;

  if (ALIUtils::debug >= 5) {
    edm::LogInfo("Alignment") << "CocoaAnalyzer::readCalibrationDB:  Number of OpticalAlignInfo READ "
                              << infoFromDB.size();
    for (const auto& myInfoFromDB : infoFromDB) {
      edm::LogInfo("Alignment") << "CocoaAnalyzer::readCalibrationDB:  OpticalAlignInfo READ " << myInfoFromDB;
    }
  }

  return infoFromDB;
}

/*
 * Correct all OpticalAlignInfo from IdealGeometry with values from DB.
 */
void CocoaAnalyzer::correctAllOpticalAlignments(std::vector<OpticalAlignInfo>& allDBOpticalAlignments) {
  if (ALIUtils::debug >= 3) {
    edm::LogInfo("Alignment") << "$$$ CocoaAnalyzer::correctAllOpticalAlignments: ";
  }

  for (const auto& myDBInfo : allDBOpticalAlignments) {
    if (ALIUtils::debug >= 5) {
      edm::LogInfo("Alignment") << "CocoaAnalyzer::findOpticalAlignInfoXML:  Looking for OAI " << myDBInfo.name_;
    }

    std::vector<OpticalAlignInfo>& allXMLOpticalAlignments = oaList_.opticalAlignments_;
    const auto& myXMLInfo = std::find_if(
        allXMLOpticalAlignments.begin(), allXMLOpticalAlignments.end(), [&](const auto& myXMLInfoCandidate) {
          return myXMLInfoCandidate.name_ == myDBInfo.name_;
        });

    if (myXMLInfo != allXMLOpticalAlignments.end()) {
      if (ALIUtils::debug >= 4) {
        edm::LogInfo("Alignment") << "CocoaAnalyzer::findOpticalAlignInfoXML:  OAI found " << myXMLInfo->name_;
        edm::LogInfo("Alignment")
            << "CocoaAnalyzer::correctAllOpticalAlignments: correcting data from XML with DB info.";
      }
      correctOpticalAlignmentParameter(myXMLInfo->x_, myDBInfo.x_);
      correctOpticalAlignmentParameter(myXMLInfo->y_, myDBInfo.y_);
      correctOpticalAlignmentParameter(myXMLInfo->z_, myDBInfo.z_);
      correctOpticalAlignmentParameter(myXMLInfo->angx_, myDBInfo.angx_);
      correctOpticalAlignmentParameter(myXMLInfo->angy_, myDBInfo.angy_);
      correctOpticalAlignmentParameter(myXMLInfo->angz_, myDBInfo.angz_);

      // Also correct extra entries
      const std::vector<OpticalAlignParam>& allDBExtraEntries = myDBInfo.extraEntries_;
      std::vector<OpticalAlignParam>& allXMLExtraEntries = myXMLInfo->extraEntries_;
      for (const auto& myDBExtraEntry : allDBExtraEntries) {
        const auto& myXMLExtraEntry = std::find_if(
            allXMLExtraEntries.begin(), allXMLExtraEntries.end(), [&](const auto& myXMLExtraEntryCandidate) {
              return myXMLExtraEntryCandidate.name_ == myDBExtraEntry.name_;
            });

        if (myXMLExtraEntry != allXMLExtraEntries.end()) {
          correctOpticalAlignmentParameter(*myXMLExtraEntry, myDBExtraEntry);
        } else {
          if (myDBExtraEntry.name_ != "None") {
            if (ALIUtils::debug >= 2) {
              edm::LogError("Alignment")
                  << "CocoaAnalyzer::correctAllOpticalAlignments:  extra entry read from DB is not present in XML "
                  << myDBExtraEntry << " in object " << myDBInfo;
            }
          }
        }
      }

      if (ALIUtils::debug >= 5) {
        edm::LogInfo("Alignment") << "CocoaAnalyzer::correctAllOpticalAlignments: corrected OpticalAlingInfo "
                                  << oaList_;
      }
    } else {
      if (ALIUtils::debug >= 2) {
        edm::LogError("Alignment") << "CocoaAnalyzer::correctAllOpticalAlignments:  OpticalAlignInfo read from DB "
                                   << myDBInfo << " is not present in XML.";
      }
    }
  }
}

/*
 * Correct an OpticalAlignment parameter from IdealGeometry with the value from DB.
 */
void CocoaAnalyzer::correctOpticalAlignmentParameter(OpticalAlignParam& myXMLParam,
                                                     const OpticalAlignParam& myDBParam) {
  if (myDBParam.value_ != -9.999E9) {
    const std::string& type = myDBParam.dimType();
    double dimFactor = 1.;

    if (type == "centre" || type == "length") {
      dimFactor = 1. / 1._m;  // in DB it is in cm
    } else if (type == "angles" || type == "angle" || type == "nodim") {
      dimFactor = 1.;
    } else {
      edm::LogError("Alignment") << "Incorrect OpticalAlignParam type = " << type;
    }

    const double correctedValue = myDBParam.value_ * dimFactor;
    if (ALIUtils::debug >= 4) {
      edm::LogInfo("Alignment") << "CocoaAnalyzer::correctOpticalAlignmentParameter  old value= " << myXMLParam.value_
                                << " new value= " << correctedValue;
    }
    myXMLParam.value_ = correctedValue;
  }
}

/*
 * Collect all information, do the fitting, and store results in DB.
 */
void CocoaAnalyzer::runCocoa() {
  if (ALIUtils::debug >= 3) {
    edm::LogInfo("Alignment") << "$$$ CocoaAnalyzer::runCocoa: ";
  }

  // Geometry model built from XML file (corrected with values from DB)
  Model& model = Model::getInstance();
  model.BuildSystemDescriptionFromOA(oaList_);

  if (ALIUtils::debug >= 3) {
    edm::LogInfo("Alignment") << "$$ CocoaAnalyzer::runCocoa: geometry built ";
  }

  // Build measurements
  model.BuildMeasurementsFromOA(measList_);

  if (ALIUtils::debug >= 3) {
    edm::LogInfo("Alignment") << "$$ CocoaAnalyzer::runCocoa: measurements built ";
  }

  // Do fit and store results in DB
  Fit::getInstance();
  Fit::startFit();

  if (ALIUtils::debug >= 0)
    edm::LogInfo("Alignment") << "............ program ended OK";
  if (ALIUtils::report >= 1) {
    ALIFileOut& fileout = ALIFileOut::getInstance(Model::ReportFName());
    fileout << "............ program ended OK";
  }
}

DEFINE_FWK_MODULE(CocoaAnalyzer);
