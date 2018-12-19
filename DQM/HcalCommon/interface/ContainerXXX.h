#ifndef ContainerXXX_h
#define ContainerXXX_h

/*
 *      file:           ContainerXXX.h
 *      Author:         Viktor Khristenko
 *
 *      Description:
 */

#include "DQM/HcalCommon/interface/Container1D.h"
#include <cmath>

namespace hcaldqm
{
  typedef boost::unordered_map<uint32_t, double> doubleCompactMap;
  typedef boost::unordered_map<uint32_t, int> intCompactMap;
  typedef boost::unordered_map<uint32_t, uint32_t> uintCompactMap;

  template<typename STDTYPE>
  class ContainerXXX
  {
  public:
    ContainerXXX() {}
    ContainerXXX(hashfunctions::HashType ht) : _hashmap(ht)
    {}
    ContainerXXX(ContainerXXX const& x);
    virtual ~ContainerXXX() {_cmap.clear();}

    //  initialize, booking. booking is done from Electronicsmap.
    virtual void initialize(hashfunctions::HashType,int debug=0);
    virtual void book(HcalElectronicsMap const*);
    virtual void book(HcalElectronicsMap const*,
                      filter::HashFilter const&);

    //  setters
    virtual void set(HcalDetId const&, STDTYPE);
    virtual void set(HcalElectronicsId const&, STDTYPE);
    virtual void set(HcalTrigTowerDetId const&, STDTYPE);

    //  getters
    virtual STDTYPE& get(HcalDetId const&);
    virtual STDTYPE& get(HcalElectronicsId const&);
    virtual STDTYPE& get(HcalTrigTowerDetId const&);

    //  pushers/adders - not a push_back.
    //  ignored if already is present
    virtual void push(HcalDetId const&, STDTYPE);
    virtual void push(HcalElectronicsId const&, STDTYPE);
    virtual void push(HcalTrigTowerDetId const&, STDTYPE);

    //  finders, note true/false - no pointer to the actual guy...
    //  I know, I know not the best coding...
    virtual bool exists(HcalDetId const&);
    virtual bool exists(HcalElectronicsId const&);
    virtual bool exists(HcalTrigTowerDetId const&);

    virtual void dump(Container1D*);
    virtual void dump(std::vector<Container1D*> const&);

    virtual void load(Container1D*);
    virtual void reset();
    virtual uint32_t size();
    virtual void print();

  protected:
    typedef boost::unordered_map<uint32_t, STDTYPE> CompactMap;
    CompactMap              _cmap;
    mapper::HashMapper      _hashmap;
    Logger                  _logger;

  public:
    virtual typename CompactMap::const_iterator begin()
    {return _cmap.begin();}
    virtual typename CompactMap::const_iterator end()
    {return _cmap.end();}
  };

  template<typename STDTYPE>
  ContainerXXX<STDTYPE>::ContainerXXX(ContainerXXX const& x)
  {
    for(auto& p : _cmap)
      {
        _cmap.insert(std::make_pair(p.first, p.second));
      }
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::initialize(hashfunctions::HashType ht,
                                         int debug)
  {
    _hashmap.initialize(ht);
    _logger.set("XXX", debug);
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::book(HcalElectronicsMap const* emap)
  {
    if (_hashmap.isDHash())
      {
        std::vector<HcalGenericDetId> dids =
          emap->allPrecisionId();
        for (std::vector<HcalGenericDetId>::const_iterator it=
               dids.begin(); it!=dids.end(); ++it)
          {
            if (!it->isHcalDetId())
              continue;

            HcalDetId did = HcalDetId(it->rawId());
            uint32_t hash = _hashmap.getHash(did);
            _logger.debug(_hashmap.getName(did));
            typename CompactMap::iterator mit = _cmap.find(hash);
            if (mit!=_cmap.end())
              continue;

            _cmap.insert(
                         std::make_pair(hash, STDTYPE(0)));
          }
      }
    else if (_hashmap.isEHash())
      {
        std::vector<HcalElectronicsId> eids =
          emap->allElectronicsIdPrecision();
        for (std::vector<HcalElectronicsId>::const_iterator it=
               eids.begin(); it!=eids.end(); ++it)
          {
            HcalElectronicsId eid = HcalElectronicsId(it->rawId());
            uint32_t hash = _hashmap.getHash(eid);
            _logger.debug(_hashmap.getName(eid));
            typename CompactMap::iterator mit = _cmap.find(hash);
            if (mit!=_cmap.end())
              continue;

            _cmap.insert(
                         std::make_pair(hash, STDTYPE(0)));
          }
      }
    else if (_hashmap.isTHash())
      {
        std::vector<HcalTrigTowerDetId> tids = emap->allTriggerId();
        for (std::vector<HcalTrigTowerDetId>::const_iterator it=
               tids.begin(); it!=tids.end(); ++it)
          {
            HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
            uint32_t hash = _hashmap.getHash(tid);
            _logger.debug(_hashmap.getName(tid));
            typename CompactMap::iterator mit = _cmap.find(hash);
            if (mit!=_cmap.end())
              continue;

            _cmap.insert(
                         std::make_pair(hash, STDTYPE(0)));
          }
      }
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::book(HcalElectronicsMap const* emap,
                                   filter::HashFilter const& filter)
  {
    if (_hashmap.isDHash())
      {
        std::vector<HcalGenericDetId> dids =
          emap->allPrecisionId();
        for (std::vector<HcalGenericDetId>::const_iterator it=
               dids.begin(); it!=dids.end(); ++it)
          {
            if (!it->isHcalDetId())
              continue;

            HcalDetId did = HcalDetId(it->rawId());
            uint32_t hash = _hashmap.getHash(did);
            typename CompactMap::iterator mit = _cmap.find(hash);
            if (mit!=_cmap.end())
              continue;
            if (filter.filter(did))
              continue;

            _logger.debug(_hashmap.getName(did));

            _cmap.insert(
                         std::make_pair(hash, STDTYPE(0)));
          }
      }
    else if (_hashmap.isEHash())
      {
        std::vector<HcalElectronicsId> eids =
          emap->allElectronicsIdPrecision();
        for (std::vector<HcalElectronicsId>::const_iterator it=
               eids.begin(); it!=eids.end(); ++it)
          {
            HcalElectronicsId eid = HcalElectronicsId(it->rawId());
            uint32_t hash = _hashmap.getHash(eid);
            typename CompactMap::iterator mit = _cmap.find(hash);
            if (filter.filter(eid))
              continue;
            if (mit!=_cmap.end())
              continue;
            _logger.debug(eid);

            _cmap.insert(
                         std::make_pair(hash, STDTYPE(0)));
          }
      }
    else if (_hashmap.isTHash())
      {
        std::vector<HcalTrigTowerDetId> tids = emap->allTriggerId();
        for (std::vector<HcalTrigTowerDetId>::const_iterator it=
               tids.begin(); it!=tids.end(); ++it)
          {
            HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
            uint32_t hash = _hashmap.getHash(tid);
            typename CompactMap::iterator mit = _cmap.find(hash);
            if (mit!=_cmap.end())
              continue;
            if (filter.filter(tid))
              continue;
            _logger.debug(_hashmap.getName(tid));

            _cmap.insert(
                         std::make_pair(hash, STDTYPE(0)));
          }
      }
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::set(HcalDetId const& did, STDTYPE x)
  {
    _cmap[_hashmap.getHash(did)] = x;
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::set(HcalElectronicsId const& did,
                                  STDTYPE x)
  {
    _cmap[_hashmap.getHash(did)] = x;
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::set(HcalTrigTowerDetId const& did,
                                  STDTYPE x)
  {
    _cmap[_hashmap.getHash(did)] = x;
  }

  template<typename STDTYPE>
  STDTYPE& ContainerXXX<STDTYPE>::get(HcalDetId const& did)
  {
    return _cmap[_hashmap.getHash(did)];
  }

  template<typename STDTYPE>
  STDTYPE& ContainerXXX<STDTYPE>::get(HcalElectronicsId const& eid)
  {
    return _cmap[_hashmap.getHash(eid)];
  }

  template<typename STDTYPE>
  STDTYPE& ContainerXXX<STDTYPE>::get(HcalTrigTowerDetId const& tid)
  {
    return _cmap[_hashmap.getHash(tid)];
  }

  template<typename STDTYPE>
  bool ContainerXXX<STDTYPE>::exists(HcalDetId const& id)
  {
    return _cmap.find(id.rawId())!=_cmap.end();
  }

  template<typename STDTYPE>
  bool ContainerXXX<STDTYPE>::exists(HcalElectronicsId const& id)
  {
    return _cmap.find(id.rawId())!=_cmap.end();
  }

  template<typename STDTYPE>
  bool ContainerXXX<STDTYPE>::exists(HcalTrigTowerDetId const& id)
  {
    return _cmap.find(id.rawId())!=_cmap.end();
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::push(HcalDetId const& did, STDTYPE x)
  {
    uint32_t hash = did.rawId();
    typename CompactMap::iterator mit=_cmap.find(hash);
    if (mit!=_cmap.end())
      return;
    _cmap.insert(
                 std::make_pair(hash, x));
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::push(HcalElectronicsId const& eid, STDTYPE x)
  {
    uint32_t hash = eid.rawId();
    typename CompactMap::iterator mit=_cmap.find(hash);
    if (mit!=_cmap.end())
      return;
    _cmap.insert(
                 std::make_pair(hash, x));
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::push(HcalTrigTowerDetId const& tid, STDTYPE x)
  {
    uint32_t hash = tid.rawId();
    typename CompactMap::iterator mit=_cmap.find(hash);
    if (mit!=_cmap.end())
      return;
    _cmap.insert(
                 std::make_pair(hash, x));
  }

  template<typename STDTYPE>
  uint32_t ContainerXXX<STDTYPE>::size()
  {
    return (uint32_t)(_cmap.size());
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::dump(Container1D* c)
  {
    for(auto& p : _cmap)
      {
        STDTYPE &x = p.second;
        uint32_t hash = p.first;
        c->fill(hash, (double)x);
      }
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::dump(std::vector<Container1D*> const &vc)
  {
    for(auto& p : _cmap)
      {
        STDTYPE &x = p.second;
        uint32_t hash = p.first;

        for (std::vector<Container1D*>::const_iterator it=vc.begin();
             it!=vc.end(); ++it)
          (*it)->fill(hash, (double)x);
      }
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::print()
  {
    std::cout << "Container by " << _hashmap.getHashTypeName() << std::endl;
    for(auto& p : _cmap)
      {
        if (_hashmap.isDHash())
          std::cout << HcalDetId(p.first) << p.second << std::endl;
        else if (_hashmap.isEHash())
          std::cout << HcalElectronicsId(p.first) << p.second
                    << std::endl;
        else if (_hashmap.isTHash())
          std::cout << HcalTrigTowerDetId(p.first) << p.second
                    << std::endl;
      }
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::reset()
  {
    for(auto& p : _cmap)
      {
        p.second = 0;
      }
  }

  template<typename STDTYPE>
  void ContainerXXX<STDTYPE>::load(Container1D* cont)
  {
    for(auto& p : _cmap)
      {
        STDTYPE &x = p.second;
        uint32_t hash = p.first;

        if (_hashmap.isDHash())
          x = cont->getBinContent(HcalDetId(hash));
        else if (_hashmap.isEHash())
          x = cont->getBinContent(HcalElectronicsId(hash));
        else if (_hashmap.isTHash())
          x = cont->getBinContent(HcalTrigTowerDetId(hash));
      }
  }
}

#endif
