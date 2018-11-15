#ifndef CondFormats_HcalObjects_StorableDoubleMap_h
#define CondFormats_HcalObjects_StorableDoubleMap_h

#include <string>
#include <memory>
#include "FWCore/Utilities/interface/Exception.h"

#include "boost/serialization/map.hpp"

template<typename T>
class StorableDoubleMap
{
public:
    typedef T value_type;

    inline ~StorableDoubleMap() {clear();}

    inline void add(const std::string& name, const std::string& category,
                    std::unique_ptr<T> ptr)
        {delete data_[category][name]; data_[category][name] = ptr.release();}

    void clear();

    inline bool empty() const {return data_.empty();}

    const T* get(const std::string& name, const std::string& category) const;

    bool exists(const std::string& name, const std::string& category) const;

    bool operator==(const StorableDoubleMap& r) const;

    inline bool operator!=(const StorableDoubleMap& r) const
        {return !(*this == r);}

private:
    typedef std::map<std::string, T*> PtrMap;
    typedef std::map<std::string, PtrMap> DataMap;
    DataMap data_;

    friend class boost::serialization::access;

    template<class Archive>
    inline void serialize(Archive & ar, unsigned /* version */)
    {
        ar & data_;
    }
};


template<typename T>
void StorableDoubleMap<T>::clear()
{
    const typename DataMap::iterator end = data_.end();
    for (typename DataMap::iterator dit = data_.begin(); dit != end; ++dit)
    {
        const typename PtrMap::iterator pend = dit->second.end();
        for (typename PtrMap::iterator pit = dit->second.begin();
             pit != pend; ++pit)
            delete pit->second;
    }
    data_.clear();
}

template<typename T>
bool StorableDoubleMap<T>::exists(const std::string& name,
                                  const std::string& category) const
{
    typename DataMap::const_iterator dit = data_.find(category);
    if (dit == data_.end())
        return false;
    else
        return !(dit->second.find(name) == dit->second.end());
}

template<typename T>
const T* StorableDoubleMap<T>::get(const std::string& name,
                                   const std::string& category) const
{
    typename DataMap::const_iterator dit = data_.find(category);
    if (dit == data_.end()) throw cms::Exception(
        "In StorableDoubleMap::get: unknown category");
    typename PtrMap::const_iterator pit = dit->second.find(name);
    if (pit == dit->second.end()) throw cms::Exception(
        "In StorableDoubleMap::get: unknown name");
    return pit->second;
}

template<typename T>
bool StorableDoubleMap<T>::operator==(const StorableDoubleMap& r) const
{
    if (data_.size() != r.data_.size())
        return false;
    typename DataMap::const_iterator dit = data_.begin();
    const typename DataMap::const_iterator end = data_.end();
    typename DataMap::const_iterator rit = r.data_.begin();
    for (; dit != end; ++dit, ++rit)
    {
        if (dit->first != rit->first)
            return false;
        if (dit->second.size() != rit->second.size())
            return false;
        typename PtrMap::const_iterator pit = dit->second.begin();
        const typename PtrMap::const_iterator pend = dit->second.end();
        typename PtrMap::const_iterator rpit = rit->second.begin();
        for (; pit != pend; ++pit, ++rpit)
        {
            if (pit->first != rpit->first)
                return false;
            if (*(pit->second) != *(rpit->second))
                return false;
        }
    }
    return true;
}

#endif // CondFormats_HcalObjects_StorableDoubleMap_h
