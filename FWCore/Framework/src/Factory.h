#ifndef EDM_FACTORY_H
#define EDM_FACTORY_H

#include "PluginManager/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/CoreFramework/src/Worker.h"
#include "FWCore/CoreFramework/src/WorkerMaker.h"

#include <map>
#include <string>
#include <memory>

namespace edm {

  class Factory : public seal::PluginFactory<Maker* ()>
  {
  public:
    typedef std::map<std::string, Maker*> MakerMap;

    ~Factory();

    static Factory* get();

    std::auto_ptr<Worker> makeWorker(ParameterSet const&,
				     std::string const& pn,
				     unsigned long vn,
				     unsigned long pass
				     ) const;


  private:
    Factory();

    mutable MakerMap makers_;
  };

}
#endif
