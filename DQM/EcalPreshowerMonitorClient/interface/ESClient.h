#ifndef ESClient_H
#define ESClient_H

#include <string>

#include "DQMServices/Core/interface/DQMStore.h"

namespace edm {
  class ParameterSet;
}

class ESClient {
public:
  ESClient(edm::ParameterSet const &);
  virtual ~ESClient() {}

  virtual void endLumiAnalyze(DQMStore::IGetter &) {}
  virtual void endJobAnalyze(DQMStore::IGetter &) {}

  void setup(DQMStore::IBooker &);

  template <typename T>
  T *getHisto(MonitorElement *, bool = false, T * = 0) const;

protected:
  virtual void book(DQMStore::IBooker &) {}

  bool initialized_;
  std::string prefixME_;
  bool cloneME_;
  bool verbose_;
  bool debug_;
};

template <typename T>
T *ESClient::getHisto(MonitorElement *_me, bool _clone /* = false*/, T *_current /* = 0*/) const {
  if (!_me) {
    if (_clone)
      return _current;
    else
      return nullptr;
  }

  TObject *obj(_me->getRootObject());

  if (!obj)
    return nullptr;

  if (_clone) {
    delete _current;
    _current = dynamic_cast<T *>(obj->Clone(("ME " + _me->getName()).c_str()));
    if (_current)
      _current->SetDirectory(nullptr);
    return _current;
  } else
    return dynamic_cast<T *>(obj);
}

#endif  // ESClient_H
