#include "DQMOffline/L1Trigger/interface/HistDefinition.h"

namespace dqmoffline {
namespace l1t {

typedef std::vector<double> vdouble;

HistDefinition::HistDefinition():
  name("name"),
  title("title"),
  nbinsX(0),
  nbinsY(0),
  xmin(0),
  xmax(0),
  ymin(0),
  ymax(0),
  binsX(),
  binsY() {

}


HistDefinition::HistDefinition(const edm::ParameterSet &ps):
  name(ps.getUntrackedParameter<std::string>("name")),
  title(ps.getUntrackedParameter<std::string>("title")),
  nbinsX(ps.getUntrackedParameter<unsigned int>("nbinsX", 0)),
  nbinsY(ps.getUntrackedParameter<unsigned int>("nbinsY", 0)),
  xmin(ps.getUntrackedParameter<double>("xmin", 0)),
  xmax(ps.getUntrackedParameter<double>("xmax", 0)),
  ymin(ps.getUntrackedParameter<double>("ymin", 0)),
  ymax(ps.getUntrackedParameter<double>("ymax", 0)),
  binsX(ps.getUntrackedParameter<vdouble>("binsX", vdouble())),
  binsY(ps.getUntrackedParameter<vdouble>("binsY", vdouble())) {

}

HistDefinition::~HistDefinition(){

}

HistDefinitions readHistDefinitions(const edm::ParameterSet &ps) {
  HistDefinitions definitions;
  std::vector<std::string> names = ps.getParameterNames();
  for(auto name: names){
    const edm::ParameterSet &hd(ps.getParameter<edm::ParameterSet>(name));
    definitions[name] = HistDefinition(hd);
  }
  return definitions;
}

}
}
