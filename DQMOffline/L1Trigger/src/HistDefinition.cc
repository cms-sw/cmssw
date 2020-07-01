#include "DQMOffline/L1Trigger/interface/HistDefinition.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

namespace dqmoffline {
  namespace l1t {

    typedef std::vector<double> vdouble;

    HistDefinition::HistDefinition()
        : name("name"), title("title"), nbinsX(0), nbinsY(0), xmin(0), xmax(0), ymin(0), ymax(0), binsX(), binsY() {}

    HistDefinition::HistDefinition(const edm::ParameterSet &ps)
        : name(ps.getUntrackedParameter<std::string>("name")),
          title(ps.getUntrackedParameter<std::string>("title")),
          nbinsX(ps.getUntrackedParameter<unsigned int>("nbinsX", 0)),
          nbinsY(ps.getUntrackedParameter<unsigned int>("nbinsY", 0)),
          xmin(ps.getUntrackedParameter<double>("xmin", 0)),
          xmax(ps.getUntrackedParameter<double>("xmax", 0)),
          ymin(ps.getUntrackedParameter<double>("ymin", 0)),
          ymax(ps.getUntrackedParameter<double>("ymax", 0)),
          binsXtmp(ps.getUntrackedParameter<vdouble>("binsX", vdouble())),
          binsYtmp(ps.getUntrackedParameter<vdouble>("binsY", vdouble())),
          binsX(binsXtmp.begin(), binsXtmp.end()),
          binsY(binsYtmp.begin(), binsYtmp.end()) {}

    HistDefinition::~HistDefinition() {}

    HistDefinitions readHistDefinitions(const edm::ParameterSet &ps,
                                        const std::map<std::string, unsigned int> &mapping) {
      HistDefinitions definitions;
      std::vector<std::string> names = ps.getParameterNames();
      std::vector<unsigned int> map_values;

      map_values.reserve(mapping.size());

for (auto const &imap : mapping) {
        map_values.push_back(imap.second);
      }
      unsigned int max_size = *std::max_element(map_values.begin(), map_values.end());
      max_size = std::max(max_size, (unsigned int)mapping.size());
      definitions.resize(max_size);

      for (const auto& name : names) {
        if (mapping.find(name) != mapping.end()) {
          const edm::ParameterSet &hd(ps.getParameter<edm::ParameterSet>(name));
          definitions[mapping.at(name)] = HistDefinition(hd);
        } else {
          edm::LogError("HistDefinition::readHistDefinitions")
              << "Could not find histogram definition for '" << name << "'" << std::endl;
        }
      }
      return definitions;
    }
  }  // namespace l1t
}  // namespace dqmoffline
