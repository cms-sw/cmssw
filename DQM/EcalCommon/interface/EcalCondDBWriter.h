#ifndef EcalCondDBWriter_H
#define EcalCondDBWriter_H

#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "DQM/EcalCommon/interface/MESet.h"

template <class Key, class V>
class PtrMap : public std::map<Key, V*> {
  typedef V* T;
  typedef V const* ConstT;
  typedef std::map<Key, T> BaseType;
public:
  PtrMap() : std::map<Key, T>() {}
  ~PtrMap()
  {
    for(typename BaseType::iterator itr(this->begin()); itr != this->end(); ++itr)
      delete itr->second;
  }

  T& operator[](const Key& _k)
  {
    return this->insert(std::make_pair(_k, T(0))).first->second;
  }
  ConstT operator[](const Key& _k) const
  {
    typename BaseType::const_iterator itr(this->find(_k));
    if(itr == this->end()) return ConstT(0);
    else return itr->second;
  }
  void clear()
  {
    for(typename BaseType::iterator itr(this->begin()); itr != this->end(); ++itr)
      delete itr->second;
    BaseType::clear();
  }
  void erase(typename BaseType::iterator _itr)
  {
    delete _itr->second;
    BaseType::erase(_itr);
  }
  size_t erase(Key const& _k)
  {
    typename BaseType::iterator itr(this->find(_k));
    if(itr == this->end()) return 0;
    delete itr->second;
    BaseType::erase(itr);
    return 1;
  }
  void erase(typename BaseType::iterator _first, typename BaseType::iterator _last)
  {
    for(typename BaseType::iterator itr(_first); itr != _last; ++itr)
      delete itr->second;
    BaseType::erase(_first, _last);
  }
};

class EcalCondDBWriter : public edm::EDAnalyzer {
 public:
  EcalCondDBWriter(edm::ParameterSet const&);
  ~EcalCondDBWriter();

 private:
  void analyze(edm::Event const&, edm::EventSetup const&);

  // DON'T CHANGE - ORDER MATTERS IN DB
  enum Clients {
    Integrity = 0,
    Cosmic = 1,
    Laser = 2,
    Pedestal = 3,
    Presample = 4,
    TestPulse = 5,
    BeamCalo = 6,
    BeamHodo = 7,
    TriggerPrimitives = 8,
    Cluster = 9,
    Timing = 10,
    Led = 11,
    RawData = 12,
    Occupancy = 13,
    nClients = 14
  };

  bool writeIntegrity(PtrMap<std::string, ecaldqm::MESet const> const&, MonRunIOV&);
  bool writeLaser(PtrMap<std::string, ecaldqm::MESet const> const&, MonRunIOV&);
  bool writePedestal(PtrMap<std::string, ecaldqm::MESet const> const&, MonRunIOV&);
  bool writePresample(PtrMap<std::string, ecaldqm::MESet const> const&, MonRunIOV&);
  bool writeTestPulse(PtrMap<std::string, ecaldqm::MESet const> const&, MonRunIOV&);
  bool writeTiming(PtrMap<std::string, ecaldqm::MESet const> const&, MonRunIOV&);
  bool writeLed(PtrMap<std::string, ecaldqm::MESet const> const&, MonRunIOV&);
  bool writeRawData(PtrMap<std::string, ecaldqm::MESet const> const&, MonRunIOV&);
  bool writeOccupancy(PtrMap<std::string, ecaldqm::MESet const> const&, MonRunIOV&);

  EcalCondDBInterface* db_;
  std::string tagName_;
  std::string location_;
  std::string runType_;
  std::vector<std::string> inputRootFiles_;
  std::string clientNames_[nClients];
  edm::ParameterSet meSetParams_[nClients];
  bool executed_;
};

#endif
