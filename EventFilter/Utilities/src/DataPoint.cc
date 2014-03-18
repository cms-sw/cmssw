/*
 * DataPoint.cc
 *
 *  Created on: Sep 24, 2012
 *      Author: aspataru
 */

#include "EventFilter/Utilities/interface/DataPoint.h"

#include <tbb/concurrent_vector.h>

#include <algorithm>
#include <assert.h>

//max collected updates per lumi
#define MAXUPDATES 200
#define MAXBINS

using namespace jsoncollector;

template class HistoJ<unsigned int>;
template class HistoJ<double>;

const std::string DataPoint::SOURCE = "source";
const std::string DataPoint::DEFINITION = "definition";
const std::string DataPoint::DATA = "data";


DataPoint::~DataPoint()
{
  if (buf_) delete[] buf_;
}

/*
 *
 * Method implementation for simple DataPoint usage
 *
 */


void DataPoint::serialize(Json::Value& root) const
{
  if (source_.size()) {
    root[SOURCE] = source_;
  }
  if (definition_.size()) {
    root[DEFINITION] = definition_;
  }
  for (unsigned int i=0;i<data_.size();i++)
    root[DATA].append(data_[i]);
}


void DataPoint::deserialize(Json::Value& root)
{
  source_ = root.get(SOURCE, "").asString();
  definition_ = root.get(DEFINITION, "").asString();
  if (root.get(DATA, "").isArray()) {
    unsigned int size = root.get(DATA, "").size();
    for (unsigned int i = 0; i < size; i++) {
      data_.push_back(root.get(DATA, "")[i].asString());
    }
  }
}

/*
 *
 * Method implementation for the new multi-threaded model
 *
 * */

void DataPoint::trackMonitorable(JsonMonitorable *monitorable,bool NAifZeroUpdates)
{
  name_=monitorable->getName();
  tracked_ = (void*)monitorable;
  if (dynamic_cast<IntJ*>(monitorable)) monType_=TYPEINT;
  else if (dynamic_cast<DoubleJ*>(monitorable)) monType_=TYPEDOUBLE;
  else if (dynamic_cast<StringJ*>(monitorable)) monType_=TYPESTRING;
  else assert(0);
  NAifZeroUpdates_=NAifZeroUpdates;

}

void DataPoint::trackVectorUInt(std::string const& name, std::vector<unsigned int>  *monvec, bool NAifZeroUpdates)
{
  name_=name;
  tracked_ = (void*)monvec;
  isStream_=true;
  monType_=TYPEUINT;
  NAifZeroUpdates_=NAifZeroUpdates;
  makeStreamLumiMap(monvec->size());
}

void DataPoint::trackVectorUIntAtomic(std::string const& name, std::vector<AtomicMonUInt*>  *monvec, bool NAifZeroUpdates)
{
  name_=name;
  tracked_ = (void*)monvec;
  isStream_=true;
  isAtomic_=true;
  monType_=TYPEUINT;
  NAifZeroUpdates_=NAifZeroUpdates;
  makeStreamLumiMap(monvec->size());
}

void DataPoint::makeStreamLumiMap(unsigned int size)
{
  for (unsigned int i=0;i<size;i++) {
    streamDataMaps_.push_back(MonPtrMap());
  }
}

void DataPoint::serialize(Json::Value& root, bool rootInit, std::string const&input) const
{
  if (rootInit) {
    if (source_.size())
      root[SOURCE] = source_;
    if (definition_.size())
      root[DEFINITION] = definition_;
  }
  root[DATA].append(input);
}

void DataPoint::snap(unsigned int lumi)
{
  isCached_=false;
  if (isStream_) {
    if (monType_==TYPEUINT)
    {
      for (unsigned int i=0; i<streamDataMaps_.size();i++) {
        unsigned int streamLumi_=streamLumisPtr_->at(i);//get currently processed stream lumi
        unsigned int monVal;

#if ATOMIC_LEVEL>0
        if (isAtomic_) monVal = (static_cast<std::vector<AtomicMonUInt*>*>(tracked_))->at(i)->load(std::memory_order_relaxed);
#else 
        if (isAtomic_) monVal = *((static_cast<std::vector<AtomicMonUInt*>*>(tracked_))->at(i));
#endif
	else monVal = (static_cast<std::vector<unsigned int>*>(tracked_))->at(i);

	auto itr =  streamDataMaps_[i].find(streamLumi_);
	if (itr==streamDataMaps_[i].end())
	{
	  if (opType_==OPHISTO) {
            if (*nBinsPtr_) {
              HistoJ<unsigned int> *nh = new HistoJ<unsigned int>(1,MAXUPDATES);
              nh->update(monVal);
              streamDataMaps_[i][streamLumi_] = nh;
            }
          }
          else {//default to SUM
            IntJ *nj = new IntJ;
            nj->update(monVal);
            streamDataMaps_[i][streamLumi_]= nj;
          }
        }
        else { 
          if (opType_==OPHISTO) {
            if (*nBinsPtr_) {
              (static_cast<HistoJ<unsigned int> *>(itr->second.get()))->update(monVal);
            }
          }
          else {
            *(static_cast<IntJ*>(itr->second.get()))=monVal;
          }
        }
      }
    }
    else assert(monType_!=TYPEINT);//not yet implemented, application error
  }
  else snapGlobal(lumi);
}

void DataPoint::snapGlobal(unsigned int lumi)
{
  isCached_=false;
  if (isStream_) return;
  auto itr = globalDataMap_.find(lumi);
  if (itr==globalDataMap_.end()) {
    if (monType_==TYPEINT) {
      IntJ *ij = new IntJ;
      ij->update((static_cast<IntJ*>(tracked_))->value());
      globalDataMap_[lumi]=ij;
    }
    if (monType_==TYPEDOUBLE) {
      DoubleJ *dj = new DoubleJ;
      dj->update((static_cast<DoubleJ*>(tracked_))->value());
      globalDataMap_[lumi]=dj;
    }
    if (monType_==TYPESTRING) {
      StringJ *sj = new StringJ;
      sj->update((static_cast<StringJ*>(tracked_))->value());
      globalDataMap_[lumi]=sj;
    }
  } else { 
    if (monType_==TYPEINT)
      static_cast<IntJ*>(itr->second.get())->update((static_cast<IntJ*>(tracked_))->value());
    else if (monType_==TYPEDOUBLE)
      static_cast<DoubleJ*>(itr->second.get())->update((static_cast<DoubleJ*>(tracked_))->value());
    else if (monType_==TYPESTRING)
      static_cast<StringJ*>(itr->second.get())->concatenate((static_cast<StringJ*>(tracked_))->value());
  }
}

void DataPoint::snapStreamAtomic(unsigned int streamID, unsigned int lumi)
{
  if (!isStream_ || !isAtomic_) return;
  isCached_=false;
  if (monType_==TYPEUINT)
  {
    unsigned int monVal;
#if ATOMIC_LEVEL>0
    if (isAtomic_) monVal = (static_cast<std::vector<AtomicMonUInt*>*>(tracked_))->at(streamID)->load(std::memory_order_relaxed);
#else 
    if (isAtomic_) monVal = *((static_cast<std::vector<AtomicMonUInt*>*>(tracked_))->at(streamID));
#endif
    else monVal = (static_cast<std::vector<unsigned int>*>(tracked_))->at(streamID);

    auto itr =  streamDataMaps_[streamID].find(lumi);
    if (itr==streamDataMaps_[streamID].end()) //insert
    {
      if (opType_==OPHISTO) {
        if (*nBinsPtr_) {
          HistoJ<unsigned int> *h = new HistoJ<unsigned int>(1,MAXUPDATES);
          h->update(monVal);
          streamDataMaps_[streamID][lumi] = h;
        }
      }
      else {//default to SUM

        IntJ *h = new IntJ;
        h->update(monVal);
        streamDataMaps_[streamID][lumi] = h;
      }
    }
    else 
    { 
      if (opType_==OPHISTO) {
        if (*nBinsPtr_) {
          static_cast<HistoJ<unsigned int>*>(itr->second.get())->update(monVal);
        }
      }
      else
        *(static_cast<IntJ*>(itr->second.get()))=monVal;
    }
  }
  else assert(monType_!=TYPEINT);//not yet implemented
}

std::string DataPoint::fastOutCSV()
{
  if (tracked_) {
    if (isStream_) {
      std::stringstream ss;
      if (isAtomic_) { 
#if ATOMIC_LEVEL>0
        ss << (unsigned int) (static_cast<std::vector<AtomicMonUInt*>*>(tracked_))->at(fastIndex_)->load(std::memory_order_relaxed); 
#else
        ss << (unsigned int) *((static_cast<std::vector<AtomicMonUInt*>*>(tracked_))->at(fastIndex_));
#endif 
        fastIndex_ = (fastIndex_+1) % (static_cast<std::vector<AtomicMonUInt*>*>(tracked_))->size();
      }
      else {
        ss << (static_cast<std::vector<unsigned int>*>(tracked_))->at(fastIndex_);
        fastIndex_ = (fastIndex_+1) % (static_cast<std::vector<unsigned int>*>(tracked_))->size();
      }

      return ss.str();
    }
    return (static_cast<JsonMonitorable*>(tracked_))->toString();
  }
  return std::string("");
}

JsonMonitorable* DataPoint::mergeAndRetrieveValue(unsigned int lumi)
{
  assert(monType_==TYPEUINT && isStream_);//for now only support UINT and SUM for stream variables
  IntJ *newJ = new IntJ;
  for (unsigned int i=0;i<streamDataMaps_.size();i++) {
    auto itr = streamDataMaps_[i].find(lumi);
    if (itr!=streamDataMaps_[i].end()) {
      newJ->add(static_cast<IntJ*>(itr->second.get())->value());
    }
  }
  cacheI_=newJ->value();
  isCached_=true;
  return newJ;//assume the caller takes care of deleting the object
}

void DataPoint::mergeAndSerialize(Json::Value & root,unsigned int lumi,bool initJsonValue)
{
  if (initJsonValue) {
    root[SOURCE] = source_;
    root[DEFINITION] = definition_;
  }

  if (isDummy_) {
    root[DATA].append("N/A");
    return;
  }
  if (!isStream_) {
    auto itr = globalDataMap_.find(lumi);
    if (itr != globalDataMap_.end()) {
      root[DATA].append(itr->second.get()->toString());
    }
    else {
      if (NAifZeroUpdates_) root[DATA].append("N/A");
      else if (monType_==TYPESTRING)  root[DATA].append("");
      else  root[DATA].append("0");
    }
    return;
  }
  else {
    assert(monType_==TYPEUINT);
    if (isCached_) {
      std::stringstream ss;
      ss << cacheI_;
      root[DATA].append(ss.str());
      return;
    }
    if (opType_!=OPHISTO) {//sum is default
      std::stringstream ss;
      unsigned int updates=0;
      unsigned int sum=0;
      for (unsigned int i=0;i<streamDataMaps_.size();i++) {
        auto itr = streamDataMaps_[i].find(lumi);
        if (itr!=streamDataMaps_[i].end()) {
          sum+=static_cast<IntJ*>(itr->second.get())->value();
          updates++;
        }
      }
      if (!updates && NAifZeroUpdates_) ss << "N/A";
      ss << sum;
      root[DATA].append(ss.str());
      return;
    }
    if (opType_==OPHISTO) {
      if (nBinsPtr_==nullptr) {
        root[DATA].append("N/A");
        return;
      }
      if (*nBinsPtr_>bufLen_) {
        if (buf_) delete[] buf_;
        bufLen_=*nBinsPtr_;
        buf_= new uint32_t[bufLen_];
      }
      memset(buf_,0,bufLen_*sizeof(uint32_t));
      unsigned int updates=0;
      for (unsigned int i=0;i<streamDataMaps_.size();i++) {
        auto itr = streamDataMaps_[i].find(lumi);
        if (itr!=streamDataMaps_[i].end()) {
          HistoJ <unsigned int>* monObj = static_cast<HistoJ<unsigned int>*>(itr->second.get());
          updates+=monObj->getUpdates();
          auto &hvec = monObj->value();
          for (unsigned int i=0;i<hvec.size();i++) {
            unsigned int thisbin=(unsigned int) hvec[i];
            if (thisbin<*nBinsPtr_) {
              buf_[thisbin]++;
            }
          }
        }
      }
      std::stringstream ss;
      if (!*nBinsPtr_ || (!updates && NAifZeroUpdates_)) ss << "N/A";
      else {
        ss << "[";
        if (*nBinsPtr_) {
          for (unsigned int i=0;i<*nBinsPtr_-1;i++) {
            ss << buf_[i] << ",";
          }
          ss << buf_[*nBinsPtr_-1];
        }
        ss << "]";
      }
      root[DATA].append(ss.str());
      return;
    }
  }
}

//wipe out data that will no longer be used
void DataPoint::discardCollected(unsigned int lumi)
{
  for (unsigned int i=0;i<streamDataMaps_.size();i++)
  {
    auto itr = streamDataMaps_[i].find(lumi);
    if (itr!=streamDataMaps_[i].end()) streamDataMaps_[i].erase(lumi);
  }

  auto itr = globalDataMap_.find(lumi);
  if (itr!=globalDataMap_.end())
    globalDataMap_.erase(lumi);
}

