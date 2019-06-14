#ifndef DQMOffline_L1Trigger_HistDefinition_h
#define DQMOffline_L1Trigger_HistDefinition_h

/**
 * \class HistDefinition
 *
 *
 * Description: Class for parsing histogram definitions from a ParameterSet of
 * ParameterSets defined in the python configuration. Acceptable parameters are
 * all public members of the HistDefinition class.
 *
 * Python configuration example:
 * https://github.com/cms-sw/cmssw/blob/master/DQMOffline/L1Trigger/python/L1THistDefinitions_cff.py
 *
 * Usage in CMSSW module
 * PlotConfig enum, PlotConfigNames map:
 * https://github.com/cms-sw/cmssw/blob/master/DQMOffline/L1Trigger/interface/L1TEGammaOffline.h
 *
 * enum PlotConfig {
 *   nVertex
 * };
 *
 * static const std::map<std::string, unsigned int> PlotConfigNames;
 *
 * https://github.com/cms-sw/cmssw/blob/master/DQMOffline/L1Trigger/src/L1TEGammaOffline.cc
 *
 * const std::map<std::string, unsigned int> L1TEGammaOffline::PlotConfigNames = {
 *  {"nVertex", PlotConfig::nVertex}
 * };
 * Read from ParameterSet in module constructor:
 * histDefinitions_(dqmoffline::l1t::readHistDefinitions(ps.getParameterSet("histDefinitions"), PlotConfigNames)),
 *
 * Use to create a histogram:
 * dqmoffline::l1t::HistDefinition nVertexDef = histDefinitions_[PlotConfig::nVertex];
 * h_nVertex_ = ibooker.book1D(
 *   nVertexDef.name, nVertexDef.title, nVertexDef.nbinsX, nVertexDef.xmin, nVertexDef.xmax
 * );
 *
 *
 * \author: Luke Kreczko - kreczko@cern.ch
 *
 **/

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
      std::vector<double> binsXtmp;
      std::vector<double> binsYtmp;
      std::vector<float> binsX;
      std::vector<float> binsY;
    };

    HistDefinitions readHistDefinitions(const edm::ParameterSet &ps,
                                        const std::map<std::string, unsigned int> &mapping);

  }  // namespace l1t
}  // namespace dqmoffline

#endif
