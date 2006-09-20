#include "IOPool/TFileAdaptor/interface/TFileAdaptor.h"
#include <iostream>
#include <boost/shared_ptr.hpp>


/* 
 * wrapper to bind TFileAdaptor to root, python etc
 * loading IOPoolTFileAdaptor library and instantiating
 * TFileAdaptorUI will make root to use StorageAdaptor for I/O instead
 * of its own plugins
 */
class TFileAdaptorUI {
public:
  
  TFileAdaptorUI();
  ~TFileAdaptorUI();

  // print current Storage statistics on cout
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
