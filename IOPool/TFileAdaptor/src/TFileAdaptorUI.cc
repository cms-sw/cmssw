#include "IOPool/TFileAdaptor/interface/TFileAdaptor.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include <iostream>
#include <boost/shared_ptr.hpp>


class TFileAdaptorUI {
public:
  
  TFileAdaptorUI();
  ~TFileAdaptorUI();
  void stats() const;
  
private:
  boost::shared_ptr<TFileAdaptor> me;
};

TFileAdaptorUI::TFileAdaptorUI() {
  TFileAdaptorParams param;
  param.mode = "default";
  me.reset(new TFileAdaptor(param));
}

TFileAdaptorUI::~TFileAdaptorUI() {}

void TFileAdaptorUI::stats() const {
  me->stats(std::cout);
}
