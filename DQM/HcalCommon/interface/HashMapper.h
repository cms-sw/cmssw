#ifndef HashMapper_h
#define HashMapper_h

/**
 *	file:			HashMapper.h
 *	Author:			Viktor Khristenko
 *
 *	Description:
 *
 */

#include "DQM/HcalCommon/interface/HashFunctions.h"
#include "DQM/HcalCommon/interface/Mapper.h"

namespace hcaldqm {
  namespace mapper {
    class HashMapper : public Mapper {
    public:
      //	constructors/destructors
      HashMapper() {}
      HashMapper(hashfunctions::HashType htype) : Mapper(), _htype(htype) {}
      ~HashMapper() override {}

      //	initialize
      virtual void initialize(hashfunctions::HashType htype) { _htype = htype; }

      //	get hash
      using Mapper::getHash;
      uint32_t getHash(HcalDetId const &did) const override { return hashfunctions::hash_did[_htype](did); }
      uint32_t getHash(HcalElectronicsId const &eid) const override {
        return hashfunctions::hash_eid[_htype - hashfunctions::nHashType_did - 1](eid);
      }
      uint32_t getHash(HcalTrigTowerDetId const &tid) const override {
        return hashfunctions::hash_tid[_htype - hashfunctions::nHashType_eid - 1](tid);
      }
      uint32_t getHash(HcalTrigTowerDetId const &tid, HcalElectronicsId const &eid) const override {
        return hashfunctions::hash_mixid[_htype - hashfunctions::nHashType_tid - 1](tid, eid);
      }

      //	get name of the hashed element
      using Mapper::getName;
      std::string getName(HcalDetId const &did) const override { return hashfunctions::name_did[_htype](did); }
      std::string getName(HcalElectronicsId const &eid) const override {
        return hashfunctions::name_eid[_htype - hashfunctions::nHashType_did - 1](eid);
      }
      std::string getName(HcalTrigTowerDetId const &tid) const override {
        return hashfunctions::name_tid[_htype - hashfunctions::nHashType_eid - 1](tid);
      }
      std::string getName(HcalTrigTowerDetId const &tid, HcalElectronicsId const &eid) const override {
        return hashfunctions::name_mixid[_htype - hashfunctions::nHashType_tid - 1](tid, eid);
      }

      //	get the Hash Type Name
      virtual std::string getHashTypeName() const { return hashfunctions::hash_names[this->getLinearHashType(_htype)]; }
      virtual hashfunctions::HashType getHashType() const { return _htype; }

      //	determine the type of the hash
      virtual bool isDHash() const { return _htype < hashfunctions::nHashType_did ? true : false; }
      virtual bool isEHash() const {
        return (_htype > hashfunctions::nHashType_did && _htype < hashfunctions::nHashType_eid) ? true : false;
      }
      virtual bool isTHash() const {
        return (_htype > hashfunctions::nHashType_eid && _htype < hashfunctions::nHashType_tid) ? true : false;
      }
      virtual bool isMixHash() const {
        return (_htype > hashfunctions::nHashType_tid && _htype < hashfunctions::nHashType_mixid) ? true : false;
      }

      //	get the Linear Hash Type
      virtual int getLinearHashType(hashfunctions::HashType htype) const {
        int l = 0;
        if (htype < hashfunctions::nHashType_did)
          l = htype;
        else if (htype < hashfunctions::nHashType_eid)
          l = htype - 1;
        else if (htype < hashfunctions::nHashType_tid)
          l = htype - 2;
        else
          l = htype - 3;
        return l;
      }

    protected:
      hashfunctions::HashType _htype;
    };
  }  // namespace mapper
}  // namespace hcaldqm

#endif
