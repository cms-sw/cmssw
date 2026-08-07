#ifndef CondFormats_EcalObjects_EcalEBPhase2TPGSpikeTaggerParams_h
#define CondFormats_EcalObjects_EcalEBPhase2TPGSpikeTaggerParams_h

#include <string>
#include <vector>

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "CondFormats/Serialization/interface/Serializable.h"

class EcalEBPhase2TPGSpikeTaggerParams {
public:
  EcalEBPhase2TPGSpikeTaggerParams() : version_(0) {};
  ~EcalEBPhase2TPGSpikeTaggerParams() {};

  // generic node that holds parameters
  class Node {
  public:
    Node() {};
    ~Node() {};

    std::vector<double> dparams_;
    std::vector<unsigned int> uparams_;
    std::vector<int> iparams_;
    std::vector<std::string> sparams_;

    COND_SERIALIZABLE;
  };

  unsigned int version() const { return version_; };
  void setVersion(const unsigned int version) { version_ = version; };

protected:
  // conditions version
  unsigned int version_;

  // global parameters
  std::vector<Node> nodes_;

  // per crystal parameters
  EcalCondObjectContainer<std::vector<Node>> crystalNodes_;

  COND_SERIALIZABLE;
};

#endif
