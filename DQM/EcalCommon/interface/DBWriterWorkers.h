#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/EcalCommon/interface/MESet.h"

#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include <map>

namespace ecaldqm {

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

  class DBWriterWorker {
  public:
    DBWriterWorker(std::string const&);
    virtual ~DBWriterWorker() {}

    virtual void setup(edm::ParameterSet const&, BinService const*);
    virtual void retrieveSource();
    virtual bool run(EcalCondDBInterface*, MonRunIOV&) = 0;

    bool runsOn(std::string const& _runType) const { return runTypes_.find(_runType) != runTypes_.end(); }

    void setVerbosity(int _v) { verbosity_ = _v; }

    std::string const& getName() const { return name_; }

  protected:
    std::string const name_;
    std::set<std::string> runTypes_;
    PtrMap<std::string, MESet const> source_;
    int verbosity_;
  };

  class IntegrityWriter : public DBWriterWorker {
  public:
    IntegrityWriter();
    ~IntegrityWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&);
  };

  class LaserWriter : public DBWriterWorker {
  public:
    LaserWriter();
    ~LaserWriter() {}

    void setup(edm::ParameterSet const&, BinService const*);
    bool run(EcalCondDBInterface*, MonRunIOV&);

  private:
    std::map<int, unsigned> wlToME_;
  };

  class PedestalWriter : public DBWriterWorker {
  public:
    PedestalWriter();
    ~PedestalWriter() {}

    void setup(edm::ParameterSet const&, BinService const*);
    bool run(EcalCondDBInterface*, MonRunIOV&);

  private:
    std::map<int, unsigned> gainToME_;
    std::map<int, unsigned> pnGainToME_;
  };

  class PresampleWriter : public DBWriterWorker {
  public:
    PresampleWriter();
    ~PresampleWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&);
  };

  class TestPulseWriter : public DBWriterWorker {
  public:
    TestPulseWriter();
    ~TestPulseWriter() {}

    void setup(edm::ParameterSet const&, BinService const*);
    bool run(EcalCondDBInterface*, MonRunIOV&);

  private:
    std::map<int, unsigned> gainToME_;
    std::map<int, unsigned> pnGainToME_;
  };

  class TimingWriter : public DBWriterWorker {
  public:
    TimingWriter();
    ~TimingWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&);
  };

  class LedWriter : public DBWriterWorker {
  public:
    LedWriter();
    ~LedWriter() {}

    void setup(edm::ParameterSet const&, BinService const*);
    bool run(EcalCondDBInterface*, MonRunIOV&);

  private:
    std::map<int, unsigned> wlToME_;
  };

  class RawDataWriter : public DBWriterWorker {
  public:
    RawDataWriter();
    ~RawDataWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&);
  };

  class OccupancyWriter : public DBWriterWorker {
  public:
    OccupancyWriter();
    ~OccupancyWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&);
  };
}
