#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#include "tinyxml2.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

namespace {
  // split into tokens and convert them to uint32_t
  inline std::vector<uint32_t> split_string_to_uints(const std::string& str) {
    std::vector<uint32_t> out{};
    std::size_t iStart{str.find_first_not_of(" ,\n")}, iEnd{};
    while (std::string::npos != iStart) {
      iEnd = str.find_first_of(" ,\n", iStart);
      out.push_back(std::stoul(str.substr(iStart, iEnd), nullptr, 0));
      iStart = str.find_first_not_of(" ,\n", iEnd);
    }
    return out;
  }

  class TrackerTopologyExtractor : public tinyxml2::XMLVisitor {
  public:
    bool VisitEnter(const tinyxml2::XMLElement& elem, const tinyxml2::XMLAttribute*) override {
      if (std::strcmp(elem.Value(), "Vector") == 0) {
        const std::string att_type{elem.Attribute("type")};
        if (att_type == "numeric") {
          const std::string att_name{elem.Attribute("name")};
          if (0 == att_name.compare(0, SubdetName.size(), SubdetName)) {  // starts with
            const std::string att_nEntries{elem.Attribute("nEntries")};
            const std::size_t nEntries = att_nEntries.empty() ? 0 : std::stoul(att_nEntries);
            const auto vals = split_string_to_uints(elem.GetText());

            if (nEntries != vals.size()) {
              throw cms::Exception(
                  "StandaloneTrackerTopology",
                  ("Problem parsing element with name '" + att_name + "': " + "'nEntries' attribute claims " +
                   std::to_string(nEntries) + " elements, but parsed " + std::to_string(vals.size())));
            }
            const auto subDet = std::stoi(att_name.substr(SubdetName.size()));
            switch (subDet) {
              case PixelSubdetector::PixelBarrel:  // layer, ladder module

                /*
		  In the case of the phase-2 IT there is an additional layer of hierarcy, due ot split sensors in Layer 1
		  What follows is a ugly hack, but at least is consistent with TrackerTopologyEP.cc
		*/

                if (vals.size() > 6) {  // Phase 2: extra hierarchy level for 3D sensors
                  pxbVals_.layerStartBit_ = vals[0];
                  pxbVals_.ladderStartBit_ = vals[1];
                  pxbVals_.moduleStartBit_ = vals[2];
                  pxbVals_.doubleStartBit_ = vals[3];

                  pxbVals_.layerMask_ = vals[4];
                  pxbVals_.ladderMask_ = vals[5];
                  pxbVals_.moduleMask_ = vals[6];
                  pxbVals_.doubleMask_ = vals[7];
                } else {  // Phase-0 or Phase-1
                  pxbVals_.layerStartBit_ = vals[0];
                  pxbVals_.ladderStartBit_ = vals[1];
                  pxbVals_.moduleStartBit_ = vals[2];

                  pxbVals_.layerMask_ = vals[3];
                  pxbVals_.ladderMask_ = vals[4];
                  pxbVals_.moduleMask_ = vals[5];
                }

                foundPXB = true;
                break;

              case PixelSubdetector::PixelEndcap:  // side, disk, blade, panel, module
                pxfVals_.sideStartBit_ = vals[0];
                pxfVals_.diskStartBit_ = vals[1];
                pxfVals_.bladeStartBit_ = vals[2];
                pxfVals_.panelStartBit_ = vals[3];
                pxfVals_.moduleStartBit_ = vals[4];

                pxfVals_.sideMask_ = vals[5];
                pxfVals_.diskMask_ = vals[6];
                pxfVals_.bladeMask_ = vals[7];
                pxfVals_.panelMask_ = vals[8];
                pxfVals_.moduleMask_ = vals[9];

                foundPXF = true;
                break;

              case StripSubdetector::TIB:  // layer, str_fw_bw, str_int_ext, str, module, ster
                tibVals_.layerStartBit_ = vals[0];
                tibVals_.str_fw_bwStartBit_ = vals[1];
                tibVals_.str_int_extStartBit_ = vals[2];
                tibVals_.strStartBit_ = vals[3];
                tibVals_.moduleStartBit_ = vals[4];
                tibVals_.sterStartBit_ = vals[5];

                tibVals_.layerMask_ = vals[6];
                tibVals_.str_fw_bwMask_ = vals[7];
                tibVals_.str_int_extMask_ = vals[8];
                tibVals_.strMask_ = vals[9];
                tibVals_.moduleMask_ = vals[10];
                tibVals_.sterMask_ = vals[11];

                foundTIB = true;
                break;

              case StripSubdetector::TID:  // side, wheel, ring, module_fw_bw, module, ster
                tidVals_.sideStartBit_ = vals[0];
                tidVals_.wheelStartBit_ = vals[1];
                tidVals_.ringStartBit_ = vals[2];
                tidVals_.module_fw_bwStartBit_ = vals[3];
                tidVals_.moduleStartBit_ = vals[4];
                tidVals_.sterStartBit_ = vals[5];

                tidVals_.sideMask_ = vals[6];
                tidVals_.wheelMask_ = vals[7];
                tidVals_.ringMask_ = vals[8];
                tidVals_.module_fw_bwMask_ = vals[9];
                tidVals_.moduleMask_ = vals[10];
                tidVals_.sterMask_ = vals[11];

                foundTID = true;
                break;

              case StripSubdetector::TOB:  // layer, rod_fw_bw, rod, module, ster
                tobVals_.layerStartBit_ = vals[0];
                tobVals_.rod_fw_bwStartBit_ = vals[1];
                tobVals_.rodStartBit_ = vals[2];
                tobVals_.moduleStartBit_ = vals[3];
                tobVals_.sterStartBit_ = vals[4];

                tobVals_.layerMask_ = vals[5];
                tobVals_.rod_fw_bwMask_ = vals[6];
                tobVals_.rodMask_ = vals[7];
                tobVals_.moduleMask_ = vals[8];
                tobVals_.sterMask_ = vals[9];

                foundTOB = true;
                break;

              case StripSubdetector::TEC:  // side, wheel, petal_fw_bw, petal, ring, module, ster
                tecVals_.sideStartBit_ = vals[0];
                tecVals_.wheelStartBit_ = vals[1];
                tecVals_.petal_fw_bwStartBit_ = vals[2];
                tecVals_.petalStartBit_ = vals[3];
                tecVals_.ringStartBit_ = vals[4];
                tecVals_.moduleStartBit_ = vals[5];
                tecVals_.sterStartBit_ = vals[6];

                tecVals_.sideMask_ = vals[7];
                tecVals_.wheelMask_ = vals[8];
                tecVals_.petal_fw_bwMask_ = vals[9];
                tecVals_.petalMask_ = vals[10];
                tecVals_.ringMask_ = vals[11];
                tecVals_.moduleMask_ = vals[12];
                tecVals_.sterMask_ = vals[13];

                foundTEC = true;
                break;
            }
          }
        }
      }
      return true;
    }

    TrackerTopology getTrackerTopology() const {
      if (!(foundPXB && foundPXF && foundTIB && foundTID && foundTOB && foundTEC)) {
        throw cms::Exception("StandaloneTrackerTopology", "Could not find parameters for all tracker subdetectors");
      }
      return TrackerTopology(pxbVals_, pxfVals_, tecVals_, tibVals_, tidVals_, tobVals_);
    }

  private:
    TrackerTopology::PixelBarrelValues pxbVals_;
    TrackerTopology::PixelEndcapValues pxfVals_;
    TrackerTopology::TIBValues tibVals_;
    TrackerTopology::TIDValues tidVals_;
    TrackerTopology::TOBValues tobVals_;
    TrackerTopology::TECValues tecVals_;

    bool foundPXB = false, foundPXF = false, foundTIB = false, foundTID = false, foundTOB = false, foundTEC = false;

    const std::string SubdetName = "Subdetector";
  };
}  // namespace

namespace StandaloneTrackerTopology {
  TrackerTopology fromTrackerParametersXMLFile(const std::string& xmlFileName) {
    tinyxml2::XMLDocument xmlDoc;
    xmlDoc.LoadFile(xmlFileName.c_str());
    if (!xmlDoc.Error()) {
      TrackerTopologyExtractor extr{};
      xmlDoc.Accept(&extr);
      return extr.getTrackerTopology();
    } else {
      throw cms::Exception("StandaloneTrackerTopology",
                           std::string{"Failed to parse file "} + xmlFileName + ": " + xmlDoc.ErrorStr());
    }
  }
  TrackerTopology fromTrackerParametersXMLString(const std::string& xmlContent) {
    tinyxml2::XMLDocument xmlDoc;
    xmlDoc.Parse(xmlContent.c_str());
    if (!xmlDoc.Error()) {
      TrackerTopologyExtractor extr{};
      xmlDoc.Accept(&extr);
      return extr.getTrackerTopology();
    } else {
      throw cms::Exception("StandaloneTrackerTopology", std::string{"Error while parsing XML: "} + xmlDoc.ErrorStr());
    }
  }
}  // namespace StandaloneTrackerTopology
