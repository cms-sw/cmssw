#ifndef JetMETCorrections_FFTJetObjects_FFTJetRcdMapper_h
#define JetMETCorrections_FFTJetObjects_FFTJetRcdMapper_h

//
// A factory to combat the proliferation of ES record types
// (multiple record types are necessary due to deficiencies
// in the record dependency tracking mechanism). Templated
// upon the data type which records hold.
//
// Igor Volobouev
// 08/03/2012

#include <map>
#include <string>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

template <class DataType>
struct AbsFFTJetRcdMapper {
  virtual ~AbsFFTJetRcdMapper() {}

  virtual void acquireToken(edm::ConsumesCollector iC) = 0;

  virtual void acquireToken(edm::ConsumesCollector iC, const std::string& label) = 0;

  virtual edm::ESHandle<DataType> load(const edm::EventSetup& iSetup) const = 0;
};

template <class DataType, class RecordType>
class ConcreteFFTJetRcdMapper : public AbsFFTJetRcdMapper<DataType> {
public:
  ~ConcreteFFTJetRcdMapper() override {}

  inline void acquireToken(edm::ConsumesCollector iC) override {
    if (tokenAcquired_)
      throw cms::Exception("ESGetTokenAlreadyAcquired");
    token_ = iC.esConsumes<DataType, RecordType>();
    tokenAcquired_ = true;
  }

  inline void acquireToken(edm::ConsumesCollector iC, const std::string& label) override {
    if (tokenAcquired_)
      throw cms::Exception("ESGetTokenAlreadyAcquired");
    token_ = iC.esConsumes<DataType, RecordType>(edm::ESInputTag("", label));
    tokenAcquired_ = true;
  }

  inline edm::ESHandle<DataType> load(const edm::EventSetup& iSetup) const override {
    if (!tokenAcquired_)
      throw cms::Exception("ESGetTokenNotAcquired");
    return iSetup.getHandle(token_);
  }

private:
  edm::ESGetToken<DataType, RecordType> token_;
  bool tokenAcquired_ = false;
};

template <class DataType>
struct DefaultFFTJetRcdMapper : public std::map<std::string, AbsFFTJetRcdMapper<DataType>*> {
  typedef DataType data_type;

  inline DefaultFFTJetRcdMapper() : std::map<std::string, AbsFFTJetRcdMapper<DataType>*>() {}

  virtual ~DefaultFFTJetRcdMapper() {
    for (typename std::map<std::string, AbsFFTJetRcdMapper<DataType>*>::iterator it = this->begin(); it != this->end();
         ++it)
      delete it->second;
  }

  inline void acquireToken(const std::string& record, edm::ConsumesCollector iC) {
    typename std::map<std::string, AbsFFTJetRcdMapper<DataType>*>::iterator it = this->find(record);
    if (it == this->end())
      throw cms::Exception("KeyNotFound") << "Record \"" << record << "\" is not registered\n";
    it->second->acquireToken(iC);
  }

  inline void acquireToken(const std::string& record, edm::ConsumesCollector iC, const std::string& label) {
    typename std::map<std::string, AbsFFTJetRcdMapper<DataType>*>::iterator it = this->find(record);
    if (it == this->end())
      throw cms::Exception("KeyNotFound") << "Record \"" << record << "\" is not registered\n";
    it->second->acquireToken(iC, label);
  }

  inline edm::ESHandle<DataType> load(const std::string& record, const edm::EventSetup& iSetup) const {
    typename std::map<std::string, AbsFFTJetRcdMapper<DataType>*>::const_iterator it = this->find(record);
    if (it == this->end())
      throw cms::Exception("KeyNotFound") << "Record \"" << record << "\" is not registered\n";
    return it->second->load(iSetup);
  }

  DefaultFFTJetRcdMapper(const DefaultFFTJetRcdMapper&) = delete;
  DefaultFFTJetRcdMapper& operator=(const DefaultFFTJetRcdMapper&) = delete;
};

#endif  // JetMETCorrections_FFTJetObjects_FFTJetRcdMapper_h
