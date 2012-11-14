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

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"

template<class DataType>
struct AbsFFTJetRcdMapper
{
    virtual ~AbsFFTJetRcdMapper() {}

    virtual void load(const edm::EventSetup& iSetup,
                      edm::ESHandle<DataType>& handle) const = 0;

    virtual void load(const edm::EventSetup& iSetup,
                      const std::string& label,
                      edm::ESHandle<DataType>& handle) const = 0;
};

template<class DataType, class RecordType>
struct ConcreteFFTJetRcdMapper : public AbsFFTJetRcdMapper<DataType>
{
    virtual ~ConcreteFFTJetRcdMapper() {}

    inline void load(const edm::EventSetup& iSetup,
                      edm::ESHandle<DataType>& handle) const
        {iSetup.get<RecordType>().get(handle);}

    inline void load(const edm::EventSetup& iSetup,
                     const std::string& label,
                     edm::ESHandle<DataType>& handle) const
        {iSetup.get<RecordType>().get(label, handle);}
};

template<class DataType>
struct DefaultFFTJetRcdMapper : 
    public std::map<std::string, AbsFFTJetRcdMapper<DataType>*>
{
    typedef DataType data_type;

    inline DefaultFFTJetRcdMapper()
        : std::map<std::string, AbsFFTJetRcdMapper<DataType>*>() {}
    
    virtual ~DefaultFFTJetRcdMapper()
    {
        for (typename std::map<std::string, AbsFFTJetRcdMapper<DataType>*>::
                 iterator it = this->begin(); it != this->end(); ++it)
            delete it->second;
    }

    inline void load(const edm::EventSetup& iSetup,
                     const std::string& record,
                     edm::ESHandle<DataType>& handle) const
    {
        typename std::map<std::string, AbsFFTJetRcdMapper<DataType>*>::
            const_iterator it = this->find(record);
        if (it == this->end())
            throw cms::Exception("KeyNotFound")
                << "Record \"" << record << "\" is not registered\n";
        it->second->load(iSetup, handle);
    }

    inline void load(const edm::EventSetup& iSetup,
                     const std::string& record,
                     const std::string& label,
                     edm::ESHandle<DataType>& handle) const
    {
        typename std::map<std::string, AbsFFTJetRcdMapper<DataType>*>::
            const_iterator it = this->find(record);
        if (it == this->end())
            throw cms::Exception("KeyNotFound")
                << "Record \"" << record << "\" is not registered\n";
        it->second->load(iSetup, label, handle);
    }

private:
    DefaultFFTJetRcdMapper(const DefaultFFTJetRcdMapper&);
    DefaultFFTJetRcdMapper& operator=(const DefaultFFTJetRcdMapper&);
};

//
// Singleton for the mapper
//
template <class Mapper>
class StaticFFTJetRcdMapper
{
public:
    typedef typename Mapper::Base::data_type data_type;

    static const Mapper& instance()
    {
        static Mapper obj;
        return obj;
    }

    template <class Record>
    static void registerRecord(const std::string& record)
    {
        Mapper& rd = const_cast<Mapper&>(instance());
        delete rd[record];
        rd[record] = new ConcreteFFTJetRcdMapper<data_type,Record>();
    }

private:
    StaticFFTJetRcdMapper();
};

#endif // JetMETCorrections_FFTJetObjects_FFTJetRcdMapper_h
