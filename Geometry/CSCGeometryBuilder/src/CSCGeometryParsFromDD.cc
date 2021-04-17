/*
// \class CSCGeometryParsFromDDD
//
//  Description: CSC Geometry Pars for DD4hep
//              
//
// \author Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//         Created:  Thu, 05 March 2020 
//         Modified: Thu, 04 June 2020, following what made in PR #30047               
//         Modified: Wed, 23 December 2020 
//         Original author: Tim Cox
*/
#include "CSCGeometryParsFromDD.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/MuonNumbering/interface/CSCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/CSCGeometry/src/CSCWireGroupPackage.h"
#include "CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/Rounding.h"

using namespace std;
using namespace cms_units::operators;
using namespace geant_units::operators;

CSCGeometryParsFromDD::CSCGeometryParsFromDD() : myName("CSCGeometryParsFromDD") {}

CSCGeometryParsFromDD::~CSCGeometryParsFromDD() {}

//ddd

bool CSCGeometryParsFromDD::build(const DDCompactView* cview,
                                  const MuonGeometryConstants& muonConstants,
                                  RecoIdealGeometry& rig,
                                  CSCRecoDigiParameters& rdp) {
  std::string attribute = "MuStructure";  // could come from outside
  std::string value = "MuonEndcapCSC";    // could come from outside

  // Asking for a specific section of the MuStructure

  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};

  DDFilteredView fv(*cview, filter);

  bool doSubDets = fv.firstChild();

  if (!doSubDets) {
    edm::LogError("CSCGeometryParsFromDD")
        << "Can not proceed, no CSC parts found with the filter.  The current node is: " << fv.logicalPart().toString();
    return false;
  }
  int noOfAnonParams = 0;
  std::vector<const DDsvalues_type*> spec = fv.specifics();
  std::vector<const DDsvalues_type*>::const_iterator spit = spec.begin();
  std::vector<double> uparvals;
  std::vector<double> fpar;
  std::vector<double> dpar;
  std::vector<double> gtran(3);
  std::vector<double> grmat(9);
  std::vector<double> trm(9);

  edm::LogVerbatim("CSCGeometryParsFromDD") << "(0) CSCGeometryParsFromDD - DDD ";

  while (doSubDets) {
    spec = fv.specifics();
    spit = spec.begin();

    // get numbering information early for possible speed up of code.

    LogTrace(myName) << myName << ": create numbering scheme...";

    MuonGeometryNumbering mdn(muonConstants);
    MuonBaseNumber mbn = mdn.geoHistoryToBaseNumber(fv.geoHistory());
    CSCNumberingScheme mens(muonConstants);

    LogTrace(myName) << myName << ": find detid...";

    int id = mens.baseNumberToUnitNumber(mbn);  //@@ FIXME perhaps should return CSCDetId itself?

    LogTrace(myName) << myName << ": raw id for this detector is " << id << ", octal " << std::oct << id << ", hex "
                     << std::hex << id << std::dec;

    CSCDetId detid = CSCDetId(id);
    int jendcap = detid.endcap();
    int jstation = detid.station();
    int jring = detid.ring();
    int jchamber = detid.chamber();
    int jlayer = detid.layer();

    edm::LogVerbatim("CSCGeometryParsFromDD")
        << "(1) detId: " << id << " jendcap: " << jendcap << " jstation: " << jstation << " jring: " << jring
        << " jchamber: " << jchamber << " jlayer: " << jlayer;

    // Package up the wire group info as it's decoded
    CSCWireGroupPackage wg;
    uparvals.clear();
    LogDebug(myName) << "size of spec=" << spec.size();

    // if the specs are made no need to get all this stuff!
    int chamberType = CSCChamberSpecs::whatChamberType(jstation, jring);

    LogDebug(myName) << "Chamber Type: " << chamberType;
    size_t ct = 0;
    bool chSpecsAlreadyExist = false;
    for (; ct < rdp.pChamberType.size(); ++ct) {
      if (chamberType == rdp.pChamberType[ct]) {
        break;
      }
    }
    if (ct < rdp.pChamberType.size() && rdp.pChamberType[ct] == chamberType) {
      // it was found, therefore no need to load all the intermediate stuff from DD.
      LogDebug(myName) << "already found a " << chamberType << " at index " << ct;

      chSpecsAlreadyExist = true;
    } else {
      for (; spit != spec.end(); spit++) {
        DDsvalues_type::const_iterator it = (**spit).begin();
        for (; it != (**spit).end(); it++) {
          LogDebug(myName) << "it->second.name()=" << it->second.name();
          if (it->second.name() == "upar") {
            uparvals.emplace_back(it->second.doubles().size());
            for (double i : it->second.doubles()) {
              uparvals.emplace_back(i);
            }

            LogDebug(myName) << "found upars ";
          } else if (it->second.name() == "NoOfAnonParams") {
            noOfAnonParams = static_cast<int>(it->second.doubles()[0]);
          } else if (it->second.name() == "NumWiresPerGrp") {
            for (double i : it->second.doubles()) {
              wg.wiresInEachGroup.emplace_back(int(i));
            }
            LogDebug(myName) << "found upars " << std::endl;
          } else if (it->second.name() == "NumGroups") {
            for (double i : it->second.doubles()) {
              wg.consecutiveGroups.emplace_back(int(i));
            }
          } else if (it->second.name() == "WireSpacing") {
            wg.wireSpacing = it->second.doubles()[0];
            edm::LogVerbatim("CSCGeometryParsFromDD") << "(2) wireSpacing: " << wg.wireSpacing;
          } else if (it->second.name() == "AlignmentPinToFirstWire") {
            wg.alignmentPinToFirstWire = it->second.doubles()[0];
            edm::LogVerbatim("CSCGeometryParsFromDD") << "(3) alignmentPinToFirstWire: " << wg.alignmentPinToFirstWire;
          } else if (it->second.name() == "TotNumWireGroups") {
            wg.numberOfGroups = int(it->second.doubles()[0]);
          } else if (it->second.name() == "LengthOfFirstWire") {
            wg.narrowWidthOfWirePlane = it->second.doubles()[0];
            edm::LogVerbatim("CSCGeometryParsFromDD") << "(4) narrowWidthOfWirePlane: " << wg.narrowWidthOfWirePlane;
          } else if (it->second.name() == "LengthOfLastWire") {
            wg.wideWidthOfWirePlane = it->second.doubles()[0];
            edm::LogVerbatim("CSCGeometryParsFromDD") << "(5) wideWidthOfWirePlane: " << wg.wideWidthOfWirePlane;
          } else if (it->second.name() == "RadialExtentOfWirePlane") {
            wg.lengthOfWirePlane = it->second.doubles()[0];
            edm::LogVerbatim("CSCGeometryParsFromDD") << "(6) lengthOfWirePlane: " << wg.lengthOfWirePlane;
          }
        }
      }

      /** stuff: using a constructed wg to deconstruct it and put it in db... alternative?
	  use temporary (not wg!) storage.
	  
	  format as inserted is best documented by the actualy emplace_back statements below.
	  
	  fupar size now becomes origSize+6+wg.wiresInEachGroup.size()+wg.consecutiveGroups.size()
      **/
      uparvals.emplace_back(wg.wireSpacing);

      uparvals.emplace_back(wg.alignmentPinToFirstWire);
      uparvals.emplace_back(wg.numberOfGroups);
      uparvals.emplace_back(wg.narrowWidthOfWirePlane);
      uparvals.emplace_back(wg.wideWidthOfWirePlane);
      uparvals.emplace_back(wg.lengthOfWirePlane);
      uparvals.emplace_back(wg.wiresInEachGroup.size());
      for (CSCWireGroupPackage::Container::const_iterator it = wg.wiresInEachGroup.begin();
           it != wg.wiresInEachGroup.end();
           ++it) {
        uparvals.emplace_back(*it);
      }
      for (CSCWireGroupPackage::Container::const_iterator it = wg.consecutiveGroups.begin();
           it != wg.consecutiveGroups.end();
           ++it) {
        uparvals.emplace_back(*it);
      }

      /** end stuff **/
    }

    fpar.clear();

    if (fv.logicalPart().solid().shape() == DDSolidShape::ddsubtraction) {
      const DDSubtraction& first = fv.logicalPart().solid();
      const DDSubtraction& second = first.solidA();
      const DDSolid& third = second.solidA();
      dpar = third.parameters();
      std::transform(
          dpar.begin(), dpar.end(), dpar.begin(), [](double i) -> double { return cms_rounding::roundIfNear0(i); });

    } else {
      dpar = fv.logicalPart().solid().parameters();
      std::transform(
          dpar.begin(), dpar.end(), dpar.begin(), [](double i) -> double { return cms_rounding::roundIfNear0(i); });
    }

    LogTrace(myName) << myName << ": noOfAnonParams=" << noOfAnonParams;
    LogTrace(myName) << myName << ": fill fpar...";
    LogTrace(myName) << myName << ": dpars are... " << convertMmToCm(dpar[4]) << ", " << convertMmToCm(dpar[8]) << ", "
                     << convertMmToCm(dpar[3]) << ", " << convertMmToCm(dpar[0]);
    edm::LogVerbatim("CSCGeometryParsFromDD")
        << "(7) dpar[4]: " << convertMmToCm(dpar[4]) << " dpar[8]:  " << convertMmToCm(dpar[8])
        << " dpar[3]: " << convertMmToCm(dpar[3]) << " dpar[0]: " << convertMmToCm(dpar[0]);

    fpar.emplace_back(convertMmToCm(dpar[4]));
    fpar.emplace_back(convertMmToCm(dpar[8]));
    fpar.emplace_back(convertMmToCm(dpar[3]));
    fpar.emplace_back(convertMmToCm(dpar[0]));

    LogTrace(myName) << myName << ": fill gtran...";

    gtran[0] = (float)1.0 * (convertMmToCm(fv.translation().X()));
    gtran[1] = (float)1.0 * (convertMmToCm(fv.translation().Y()));
    gtran[2] = (float)1.0 * (convertMmToCm(fv.translation().Z()));

    LogTrace(myName) << myName << ": gtran[0]=" << gtran[0] << ", gtran[1]=" << gtran[1] << ", gtran[2]=" << gtran[2];

    edm::LogVerbatim("CSCGeometryParsFromDD")
        << "(8) gtran[0]: " << gtran[0] << " gtran[1]: " << gtran[1] << " gtran[2]: " << gtran[2];

    LogTrace(myName) << myName << ": fill grmat...";

    fv.rotation().GetComponents(trm.begin(), trm.end());
    size_t rotindex = 0;
    for (size_t i = 0; i < 9; ++i) {
      grmat[i] = (float)1.0 * trm[rotindex];
      rotindex = rotindex + 3;
      if ((i + 1) % 3 == 0) {
        rotindex = (i + 1) / 3;
      }
    }
    LogTrace(myName) << myName << ": looking for wire group info for layer "
                     << "E" << CSCDetId::endcap(id) << " S" << CSCDetId::station(id) << " R" << CSCDetId::ring(id)
                     << " C" << CSCDetId::chamber(id) << " L" << CSCDetId::layer(id);

    if (wg.numberOfGroups != 0) {
      LogTrace(myName) << myName << ": fv.geoHistory:      = " << fv.geoHistory();
      LogTrace(myName) << myName << ": TotNumWireGroups     = " << wg.numberOfGroups;
      LogTrace(myName) << myName << ": WireSpacing          = " << wg.wireSpacing;
      LogTrace(myName) << myName << ": AlignmentPinToFirstWire = " << wg.alignmentPinToFirstWire;
      LogTrace(myName) << myName << ": Narrow width of wire plane = " << wg.narrowWidthOfWirePlane;
      LogTrace(myName) << myName << ": Wide width of wire plane = " << wg.wideWidthOfWirePlane;
      LogTrace(myName) << myName << ": Length in y of wire plane = " << wg.lengthOfWirePlane;
      LogTrace(myName) << myName << ": wg.consecutiveGroups.size() = " << wg.consecutiveGroups.size();
      LogTrace(myName) << myName << ": wg.wiresInEachGroup.size() = " << wg.wiresInEachGroup.size();
      LogTrace(myName) << myName << ": \tNumGroups\tWiresInGroup";
      for (size_t i = 0; i < wg.consecutiveGroups.size(); i++) {
        LogTrace(myName) << myName << " \t" << wg.consecutiveGroups[i] << "\t\t" << wg.wiresInEachGroup[i];
      }
    } else {
      LogTrace(myName) << myName << ": DDD is MISSING SpecPars for wire groups";
    }
    LogTrace(myName) << myName << ": end of wire group info. ";

    LogTrace(myName) << myName << ":_z_ E" << jendcap << " S" << jstation << " R" << jring << " C" << jchamber << " L"
                     << jlayer << " gx=" << gtran[0] << ", gy=" << gtran[1] << ", gz=" << gtran[2]
                     << " thickness=" << fpar[2] * 2.;

    if (jlayer == 0) {  // Can only build chambers if we're filtering them

      LogTrace(myName) << myName << ":_z_ frame=" << uparvals[31] / 10. << " gap=" << uparvals[32] / 10.
                       << " panel=" << uparvals[33] / 10. << " offset=" << uparvals[34] / 10.;

      if (jstation == 1 && jring == 1) {
        // set up params for ME1a and ME1b and call buildChamber *for each*
        // Both get the full ME11 dimensions

        // detid is for ME11 and that's what we're using for ME1b in the software

        std::transform(gtran.begin(), gtran.end(), gtran.begin(), [](double i) -> double {
          return cms_rounding::roundIfNear0(i);
        });
        std::transform(grmat.begin(), grmat.end(), grmat.begin(), [](double i) -> double {
          return cms_rounding::roundIfNear0(i);
        });
        std::transform(
            fpar.begin(), fpar.end(), fpar.begin(), [](double i) -> double { return cms_rounding::roundIfNear0(i); });

        rig.insert(id, gtran, grmat, fpar);
        if (!chSpecsAlreadyExist) {
          LogDebug(myName) << " inserting chamber type " << chamberType << std::endl;
          rdp.pChamberType.emplace_back(chamberType);
          rdp.pUserParOffset.emplace_back(rdp.pfupars.size());
          rdp.pUserParSize.emplace_back(uparvals.size());
          std::copy(uparvals.begin(), uparvals.end(), std::back_inserter(rdp.pfupars));
        }

        // No. of anonymous parameters per chamber type should be read from cscSpecs file...
        // Only required for ME11 splitting into ME1a and ME1b values,
        // If it isn't seen may as well try to get further but this value will depend
        // on structure of the file so may not even match!
        const int kNoOfAnonParams = 35;
        if (noOfAnonParams == 0) {
          noOfAnonParams = kNoOfAnonParams;
        }  // in case it wasn't seen

        // copy ME1a params from back to the front
        std::copy(
            uparvals.begin() + noOfAnonParams + 1, uparvals.begin() + (2 * noOfAnonParams) + 2, uparvals.begin() + 1);

        CSCDetId detid1a = CSCDetId(jendcap, 1, 4, jchamber, 0);  // reset to ME1A

        std::transform(gtran.begin(), gtran.end(), gtran.begin(), [](double i) -> double {
          return cms_rounding::roundIfNear0(i);
        });
        std::transform(grmat.begin(), grmat.end(), grmat.begin(), [](double i) -> double {
          return cms_rounding::roundIfNear0(i);
        });
        std::transform(
            fpar.begin(), fpar.end(), fpar.begin(), [](double i) -> double { return cms_rounding::roundIfNear0(i); });

        rig.insert(detid1a.rawId(), gtran, grmat, fpar);
        int chtypeA = CSCChamberSpecs::whatChamberType(1, 4);
        ct = 0;
        for (; ct < rdp.pChamberType.size(); ++ct) {
          if (chtypeA == rdp.pChamberType[ct]) {
            break;
          }
        }
        if (ct < rdp.pChamberType.size() && rdp.pChamberType[ct] == chtypeA) {
          // then its in already, don't put it
          LogDebug(myName) << "found chamber type " << chtypeA << " so don't put it in! ";
        } else {
          LogDebug(myName) << " inserting chamber type " << chtypeA;
          rdp.pChamberType.emplace_back(chtypeA);
          rdp.pUserParOffset.emplace_back(rdp.pfupars.size());
          rdp.pUserParSize.emplace_back(uparvals.size());
          std::copy(uparvals.begin(), uparvals.end(), std::back_inserter(rdp.pfupars));
        }

      } else {
        std::transform(gtran.begin(), gtran.end(), gtran.begin(), [](double i) -> double {
          return cms_rounding::roundIfNear0(i);
        });
        std::transform(grmat.begin(), grmat.end(), grmat.begin(), [](double i) -> double {
          return cms_rounding::roundIfNear0(i);
        });
        std::transform(
            fpar.begin(), fpar.end(), fpar.begin(), [](double i) -> double { return cms_rounding::roundIfNear0(i); });
        rig.insert(id, gtran, grmat, fpar);
        if (!chSpecsAlreadyExist) {
          LogDebug(myName) << " inserting chamber type " << chamberType;
          rdp.pChamberType.emplace_back(chamberType);
          rdp.pUserParOffset.emplace_back(rdp.pfupars.size());
          rdp.pUserParSize.emplace_back(uparvals.size());
          std::copy(uparvals.begin(), uparvals.end(), std::back_inserter(rdp.pfupars));
        }
      }

    }  // filtering chambers.

    doSubDets = fv.next();
  }
  return true;
}

// dd4hep
bool CSCGeometryParsFromDD::build(const cms::DDCompactView* cview,
                                  const MuonGeometryConstants& muonConstants,
                                  RecoIdealGeometry& rig,
                                  CSCRecoDigiParameters& rdp) {
  const std::string attribute = "MuStructure";
  const std::string value = "MuonEndcapCSC";
  const cms::DDSpecParRegistry& mypar = cview->specpars();
  const cms::DDFilter filter(attribute, value);
  cms::DDFilteredView fv(*cview, filter);

  int noOfAnonParams = 0;

  std::vector<double> uparvals;
  std::vector<double> fpar;
  std::vector<double> dpar;
  std::vector<double> gtran(3);
  std::vector<double> grmat(9);
  std::vector<double> trm(9);

  edm::LogVerbatim("CSCGeometryParsFromDD") << "(0) CSCGeometryParsFromDD - DD4HEP ";

  while (fv.firstChild()) {
    MuonGeometryNumbering mbn(muonConstants);
    CSCNumberingScheme cscnum(muonConstants);
    int id = cscnum.baseNumberToUnitNumber(mbn.geoHistoryToBaseNumber(fv.history()));
    CSCDetId detid = CSCDetId(id);

    int jendcap = detid.endcap();
    int jstation = detid.station();
    int jring = detid.ring();
    int jchamber = detid.chamber();
    int jlayer = detid.layer();

    edm::LogVerbatim("CSCGeometryParsFromDD")
        << "(1) detId: " << id << " jendcap: " << jendcap << " jstation: " << jstation << " jring: " << jring
        << " jchamber: " << jchamber << " jlayer: " << jlayer;

    // Package up the wire group info as it's decoded
    CSCWireGroupPackage wg;
    uparvals.clear();

    // if the specs are made no need to get all this stuff!
    int chamberType = CSCChamberSpecs::whatChamberType(jstation, jring);

    size_t ct = 0;
    bool chSpecsAlreadyExist = false;

    for (; ct < rdp.pChamberType.size(); ++ct) {
      if (chamberType == rdp.pChamberType[ct]) {
        break;
      }
    }

    if (ct < rdp.pChamberType.size() && rdp.pChamberType[ct] == chamberType) {
      chSpecsAlreadyExist = true;
    } else {
      std::string_view my_name_bis = fv.name();
      std::string my_name_tris = std::string(my_name_bis);
      std::vector<std::string_view> namesInPath = mypar.names("//" + my_name_tris);
      std::string my_string = "ChamberSpecs_";
      int index = -1;
      for (vector<string_view>::size_type i = 0; i < namesInPath.size(); ++i) {
        std::size_t found = namesInPath[i].find(my_string);
        if (found != std::string::npos)
          index = i;
      }
      uparvals = fv.get<std::vector<double>>(std::string(namesInPath[index]), "upar");

      auto it = uparvals.begin();
      it = uparvals.insert(it, uparvals.size());
      auto noofanonparams = fv.get<double>("NoOfAnonParams");
      noOfAnonParams = static_cast<int>(noofanonparams);

      for (auto i : fv.get<std::vector<double>>(std::string(namesInPath[index]), "NumWiresPerGrp")) {
        wg.wiresInEachGroup.emplace_back(int(i));
      }

      for (auto i : fv.get<std::vector<double>>(std::string(namesInPath[index]), "NumGroups")) {
        wg.consecutiveGroups.emplace_back(int(i));
      }

      auto wirespacing = fv.get<double>("WireSpacing");
      wg.wireSpacing = static_cast<double>(wirespacing / dd4hep::mm);
      edm::LogVerbatim("CSCGeometryParsFromDD") << "(2) wireSpacing: " << wg.wireSpacing;

      auto alignmentpintofirstwire = fv.get<double>("AlignmentPinToFirstWire");
      wg.alignmentPinToFirstWire = static_cast<double>(alignmentpintofirstwire / dd4hep::mm);
      edm::LogVerbatim("CSCGeometryParsFromDD") << "(3) alignmentPinToFirstWire: " << wg.alignmentPinToFirstWire;

      auto totnumwiregroups = fv.get<double>("TotNumWireGroups");
      wg.numberOfGroups = static_cast<int>(totnumwiregroups);

      auto lengthoffirstwire = fv.get<double>("LengthOfFirstWire");
      wg.narrowWidthOfWirePlane = static_cast<double>(lengthoffirstwire / dd4hep::mm);
      edm::LogVerbatim("CSCGeometryParsFromDD") << "(4) narrowWidthOfWirePlane: " << wg.narrowWidthOfWirePlane;

      auto lengthoflastwire = fv.get<double>("LengthOfLastWire");
      wg.wideWidthOfWirePlane = static_cast<double>(lengthoflastwire / dd4hep::mm);
      edm::LogVerbatim("CSCGeometryParsFromDD") << "(5) wideWidthOfWirePlane: " << wg.wideWidthOfWirePlane;

      auto radialextentofwireplane = fv.get<double>("RadialExtentOfWirePlane");
      wg.lengthOfWirePlane = static_cast<double>(radialextentofwireplane / dd4hep::mm);
      edm::LogVerbatim("CSCGeometryParsFromDD") << "(6) lengthOfWirePlane: " << wg.lengthOfWirePlane;

      uparvals.emplace_back(wg.wireSpacing);
      uparvals.emplace_back(wg.alignmentPinToFirstWire);
      uparvals.emplace_back(wg.numberOfGroups);
      uparvals.emplace_back(wg.narrowWidthOfWirePlane);
      uparvals.emplace_back(wg.wideWidthOfWirePlane);
      uparvals.emplace_back(wg.lengthOfWirePlane);
      uparvals.emplace_back(wg.wiresInEachGroup.size());

      for (CSCWireGroupPackage::Container::const_iterator it = wg.wiresInEachGroup.begin();
           it != wg.wiresInEachGroup.end();
           ++it) {
        uparvals.emplace_back(*it);
      }
      for (CSCWireGroupPackage::Container::const_iterator it = wg.consecutiveGroups.begin();
           it != wg.consecutiveGroups.end();
           ++it) {
        uparvals.emplace_back(*it);
      }

      /** end stuff **/
    }

    fpar.clear();

    std::string my_title(fv.solid()->GetTitle());

    if (my_title == "Subtraction") {
      cms::DDSolid mysolid(fv.solid());
      auto solidA = mysolid.solidA();
      std::vector<double> dpar = solidA.dimensions();

      std::transform(
          dpar.begin(), dpar.end(), dpar.begin(), [](double i) -> double { return cms_rounding::roundIfNear0(i); });

      fpar.emplace_back((dpar[1] / dd4hep::cm));
      fpar.emplace_back((dpar[2] / dd4hep::cm));
      fpar.emplace_back((dpar[3] / dd4hep::cm));
      fpar.emplace_back((dpar[4] / dd4hep::cm));
      edm::LogVerbatim("CSCGeometryParsFromDD")
          << "(7) - Subtraction - dpar[1] (ddd dpar[4]): " << dpar[1] / dd4hep::cm
          << " (ddd dpar[8]):  " << dpar[2] / dd4hep::cm << " dpar[3] (as ddd): " << dpar[3] / dd4hep::cm
          << " dpar[4] (ddd dpar[0]): " << dpar[4] / dd4hep::cm;
    } else {
      dpar = fv.parameters();

      std::transform(
          dpar.begin(), dpar.end(), dpar.begin(), [](double i) -> double { return cms_rounding::roundIfNear0(i); });

      fpar.emplace_back((dpar[0] / dd4hep::cm));
      fpar.emplace_back((dpar[1] / dd4hep::cm));
      fpar.emplace_back((dpar[2] / dd4hep::cm));
      fpar.emplace_back((dpar[3] / dd4hep::cm));
      edm::LogVerbatim("CSCGeometryParsFromDD")
          << "(7)Bis - Else - dpar[0]: " << dpar[0] / dd4hep::cm << " dpar[1]: " << dpar[1] / dd4hep::cm
          << " dpar[2]:  " << dpar[2] / dd4hep::cm << " dpar[3]: " << dpar[3] / dd4hep::cm;
    }

    gtran[0] = static_cast<float>(fv.translation().X() / dd4hep::cm);
    gtran[1] = static_cast<float>(fv.translation().Y() / dd4hep::cm);
    gtran[2] = static_cast<float>(fv.translation().Z() / dd4hep::cm);

    edm::LogVerbatim("CSCGeometryParsFromDD")
        << "(8) gtran[0]: " << gtran[0] / dd4hep::cm << " gtran[1]: " << gtran[1] / dd4hep::cm
        << " gtran[2]: " << gtran[2] / dd4hep::cm;

    std::transform(
        gtran.begin(), gtran.end(), gtran.begin(), [](double i) -> double { return cms_rounding::roundIfNear0(i); });

    fv.rotation().GetComponents(trm.begin(), trm.end());
    size_t rotindex = 0;
    for (size_t i = 0; i < 9; ++i) {
      grmat[i] = static_cast<float>(trm[rotindex]);
      rotindex = rotindex + 3;
      if ((i + 1) % 3 == 0) {
        rotindex = (i + 1) / 3;
      }
    }

    if (wg.numberOfGroups == 0) {
      LogTrace(myName) << myName << " wg.numberOfGroups == 0 ";
    }

    if (jlayer == 0) {  // Can only build chambers if we're filtering them

      if (jstation == 1 && jring == 1) {
        std::transform(gtran.begin(), gtran.end(), gtran.begin(), [](double i) -> double {
          return cms_rounding::roundIfNear0(i);
        });

        std::transform(grmat.begin(), grmat.end(), grmat.begin(), [](double i) -> double {
          return cms_rounding::roundIfNear0(i);
        });

        std::transform(
            fpar.begin(), fpar.end(), fpar.begin(), [](double i) -> double { return cms_rounding::roundIfNear0(i); });

        rig.insert(id, gtran, grmat, fpar);
        if (!chSpecsAlreadyExist) {
          rdp.pChamberType.emplace_back(chamberType);
          rdp.pUserParOffset.emplace_back(rdp.pfupars.size());
          rdp.pUserParSize.emplace_back(uparvals.size());
          std::copy(uparvals.begin(), uparvals.end(), std::back_inserter(rdp.pfupars));
        }

        // No. of anonymous parameters per chamber type should be read from cscSpecs file...
        // Only required for ME11 splitting into ME1a and ME1b values,
        // If it isn't seen may as well try to get further but this value will depend
        // on structure of the file so may not even match!
        const int kNoOfAnonParams = 35;
        if (noOfAnonParams == 0) {
          noOfAnonParams = kNoOfAnonParams;
        }  // in case it wasn't seen

        // copy ME1a params from back to the front
        std::copy(
            uparvals.begin() + noOfAnonParams + 1, uparvals.begin() + (2 * noOfAnonParams) + 2, uparvals.begin() + 1);

        CSCDetId detid1a = CSCDetId(jendcap, 1, 4, jchamber, 0);  // reset to ME1A

        std::transform(gtran.begin(), gtran.end(), gtran.begin(), [](double i) -> double {
          return cms_rounding::roundIfNear0(i);
        });

        std::transform(grmat.begin(), grmat.end(), grmat.begin(), [](double i) -> double {
          return cms_rounding::roundIfNear0(i);
        });

        std::transform(
            fpar.begin(), fpar.end(), fpar.begin(), [](double i) -> double { return cms_rounding::roundIfNear0(i); });

        rig.insert(detid1a.rawId(), gtran, grmat, fpar);
        int chtypeA = CSCChamberSpecs::whatChamberType(1, 4);
        ct = 0;
        for (; ct < rdp.pChamberType.size(); ++ct) {
          if (chtypeA == rdp.pChamberType[ct]) {
            break;
          }
        }
        if (ct < rdp.pChamberType.size() && rdp.pChamberType[ct] == chtypeA) {
          // then its in already, don't put it
          LogTrace(myName) << myName << " ct < rdp.pChamberType.size() && rdp.pChamberType[ct] == chtypeA ";
        } else {
          rdp.pChamberType.emplace_back(chtypeA);
          rdp.pUserParOffset.emplace_back(rdp.pfupars.size());
          rdp.pUserParSize.emplace_back(uparvals.size());
          std::copy(uparvals.begin(), uparvals.end(), std::back_inserter(rdp.pfupars));
        }

      } else {
        std::transform(gtran.begin(), gtran.end(), gtran.begin(), [](double i) -> double {
          return cms_rounding::roundIfNear0(i);
        });

        std::transform(grmat.begin(), grmat.end(), grmat.begin(), [](double i) -> double {
          return cms_rounding::roundIfNear0(i);
        });

        std::transform(
            fpar.begin(), fpar.end(), fpar.begin(), [](double i) -> double { return cms_rounding::roundIfNear0(i); });

        rig.insert(id, gtran, grmat, fpar);
        if (!chSpecsAlreadyExist) {
          rdp.pChamberType.emplace_back(chamberType);
          rdp.pUserParOffset.emplace_back(rdp.pfupars.size());
          rdp.pUserParSize.emplace_back(uparvals.size());
          std::copy(uparvals.begin(), uparvals.end(), std::back_inserter(rdp.pfupars));
        }
      }

    }  // filtering chambers.
  }

  return true;
}
