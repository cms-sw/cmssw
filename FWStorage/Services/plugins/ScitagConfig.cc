/**\class edm::service::ScitagConfig

Description: This is the implementation for a service
that will append scitags to Physical File Names (PFNs).
The scitags are implemented as CGI parameters that
are appended to the PFN URL. A scitag looks like
"?scitag.flow=<scitag_id>", where <scitag_id> is
a numerical value that identifies a CMS use case.

XRootD is the only protocol that currently supports
this feature and knows how to use these scitags.
A PFN that uses XRootD starts with the prefix "root:".
Currently, for other protocols nothing is appended
to the PFN (some protocols would ignore the scitag
and for others its presence would cause an error).

The service can be configured to use different scitags
for analysis and production use cases. The WM system
is expected to set the "productionCase" parameter
appropriately when configuring the service. It defaults
to the analysis case.

The Framework will automatically select
a scitag from 3 possible choices.

  - PreMixedPileup scitag: An embedded source run by
   the PreMixingModule
  - Embedded scitag: All other embedded sources
  - Primary scitag: Everything else

For each of the 6 cases (analysis/production x 3 scitag types)
the numerical value in the scitag can be independently
configured. The default values are in the fillDescriptions
function below.

The enable parameter can be used to turn off scitags
(nothing is appended to the PFN). Independent of that
a source can use the "Undefined" scitag category
and that will also disable scitags.

XRootD reads the scitag value from the CGI parameter
and uses it to label packets it sends out over the network.
There are network monitoring tools that can use the
packet labels to analyze network traffic not only for CMS
but also for many other scientific projects that label
their network packets in this way. CMS is only responsible
for setting the scitag value in the PFN sent to XRootD.
Other organizations handle the network monitoring.
*/
//
// Original Author: W. David Dagenhart
//         Created: 29 Dec 2025

#include <format>
#include <string>
#include <vector>

#include "FWCore/Catalog/interface/StorageURLModifier.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace {
  constexpr char const* const kXRootPrefix = "root:";
  // These are the values assigned to CMS.
  // Values outside this range are used by other organizations.
  constexpr int kCMSSciTagRangeStart = 196612;
  constexpr int kCMSSciTagRangeEnd = 196860;
}  // namespace

namespace edm {
  namespace service {

    class ScitagConfig : public StorageURLModifier {
    public:
      explicit ScitagConfig(ParameterSet const& pset);

      void modify(SciTagCategory sciTagCategory, std::string& url) const override;

      static void fillDescriptions(ConfigurationDescriptions& descriptions);

    private:
      std::vector<unsigned int> sciTags_;
      bool enable_;
    };

    ScitagConfig::ScitagConfig(ParameterSet const& pset) : enable_(pset.getUntrackedParameter<bool>("enable")) {
      bool productionCase = pset.getUntrackedParameter<bool>("productionCase");
      if (enable_) {
        if (productionCase) {
          ParameterSet productionPSet = pset.getUntrackedParameter<ParameterSet>("production");
          sciTags_ = {productionPSet.getUntrackedParameter<unsigned int>("primarySciTag"),
                      productionPSet.getUntrackedParameter<unsigned int>("embeddedSciTag"),
                      productionPSet.getUntrackedParameter<unsigned int>("preMixedPileupSciTag")};
        } else {
          ParameterSet analysisPSet = pset.getUntrackedParameter<ParameterSet>("analysis");
          sciTags_ = {analysisPSet.getUntrackedParameter<unsigned int>("primarySciTag"),
                      analysisPSet.getUntrackedParameter<unsigned int>("embeddedSciTag"),
                      analysisPSet.getUntrackedParameter<unsigned int>("preMixedPileupSciTag")};
        }
        for (const auto& tag : sciTags_) {
          if (tag < kCMSSciTagRangeStart || tag > kCMSSciTagRangeEnd) {
            throw cms::Exception("ScitagConfig") << "SciTag value " << tag << " is outside the CMS assigned range of "
                                                 << kCMSSciTagRangeStart << " to " << kCMSSciTagRangeEnd << ".";
          }
        }
      }
    }

    void ScitagConfig::modify(SciTagCategory sciTagCategory, std::string& url) const {
      if (!enable_ || sciTagCategory == SciTagCategory::Undefined) {
        return;
      }

      if (url.starts_with(kXRootPrefix)) {
        unsigned char index = static_cast<unsigned char>(sciTagCategory);
        if (index > static_cast<unsigned char>(SciTagCategory::Undefined)) {
          // This should never happen unless there is a bug elsewhere
          throw cms::Exception("ScitagConfig") << "Invalid SciTagCategory value in modify.";
        }

        // If there are multiple CGI parameters, then only the
        // first one starts with '?' and rest start with '&'.
        bool const isFirst = url.find('?') == std::string::npos;
        url += (isFirst ? '?' : '&');
        url += "scitag.flow=";

        url.append(std::format("{}", sciTags_[index]));
      }
    }

    void ScitagConfig::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;

      ParameterSetDescription analysisDesc;
      analysisDesc.addUntracked<unsigned int>("primarySciTag", 196664);
      analysisDesc.addUntracked<unsigned int>("embeddedSciTag", 196700);
      analysisDesc.addUntracked<unsigned int>("preMixedPileupSciTag", 196704);
      desc.addUntracked<ParameterSetDescription>("analysis", analysisDesc);

      ParameterSetDescription productionDesc;
      productionDesc.addUntracked<unsigned int>("primarySciTag", 196656);
      productionDesc.addUntracked<unsigned int>("embeddedSciTag", 196700);
      productionDesc.addUntracked<unsigned int>("preMixedPileupSciTag", 196704);
      desc.addUntracked<ParameterSetDescription>("production", productionDesc);

      desc.addUntracked<bool>("enable", true);
      desc.addUntracked<bool>("productionCase", false);

      descriptions.add("ScitagConfig", desc);
    }
  }  // namespace service
}  // namespace edm

using edm::service::ScitagConfig;
using StorageURLModifierMaker = edm::serviceregistry::ParameterSetMaker<edm::StorageURLModifier, ScitagConfig>;
DEFINE_FWK_SERVICE_MAKER(ScitagConfig, StorageURLModifierMaker);
