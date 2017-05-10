#ifndef DQMHistogramJson_H
#define DQMHistogramJson_H

#include "DQMHistogramStats.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace dqmservices {

class DQMHistogramJson : public DQMHistogramStats {
 public:
  	DQMHistogramJson(edm::ParameterSet const & iConfig);

  	void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;

  	void dqmEndRun(DQMStore::IBooker &, DQMStore::IGetter &,
              edm::Run const&, 
              edm::EventSetup const&) override;

 private:
 	std::string toString(boost::property_tree::ptree doc);
 	void writeMemoryJson(const std::string &fn, const HistoStats &stats);

};
}

#endif