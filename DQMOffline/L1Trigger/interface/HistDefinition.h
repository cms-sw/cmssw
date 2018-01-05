#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>
#include <vector>

namespace dqmoffline {
namespace l1t {

class HistDefinition;
typedef std::vector<HistDefinition> HistDefinitions;

class HistDefinition {
public:
  HistDefinition();
  HistDefinition(const edm::ParameterSet &ps);
  ~HistDefinition();
  // static HistDefinitions readHistDefinitions(const edm::ParameterSet &ps,
  // std::map<std::string, unsigned int>);

  std::string name;
  std::string title;
  unsigned int nbinsX;
  unsigned int nbinsY;
  double xmin;
  double xmax;
  double ymin;
  double ymax;
  std::vector<double> binsX;
  std::vector<double> binsY;
};

HistDefinitions
readHistDefinitions(const edm::ParameterSet &ps,
                    const std::map<std::string, unsigned int> &mapping);

} // end l1t
} // end dqmoffline
