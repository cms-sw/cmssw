/*
 * DataPoint.cc
 *
 *  Created on: Sep 24, 2012
 *      Author: aspataru
 */

#include "EventFilter/Utilities/interface/DataPoint.h"
#include "EventFilter/Utilities/interface/Operations.h"

#include <tbb/concurrent_vector.h>

#include <algorithm>
#include <assert.h>

//max updates per lumi
#define MAXUPDATES 200
#define MAXBINS

using namespace jsoncollector;

//template class HistoJ<int>;
template class HistoJ<unsigned int>;
template class HistoJ<std::atomic<unsigned int>>;
template class HistoJ<double>;

const string DataPoint::SOURCE = "source";
const string DataPoint::DEFINITION = "definition";
const string DataPoint::DATA = "data";


/*
DataPoint::DataPoint(
		//unsigned int expectedUpdates, unsigned int maxUpdates):
		monitored_(&monVars) {
		monitored_ = &monVars;
		for (unsigned int i=0; i< monVars.size();i++) {
			std::unique_ptr<JsonMonitorable> jptr;
			if (monVars[i]) {
				const JsonMonitorable &  mon = *data[i];
				MonType=monConfig.getMonType();
				if (monConfig[i].getOperationType() == JsonMonConfig::HISTO) {
					//or use  std::bad_cast and a ref..
					if (MonType==JsonMonConfig::TYPEINT) {
						jptr.reset(new HistoJ<int>(expectedUpdates,maxUpdates));
					}
				}
				else if (monConfig[i].getOperationType() == JsonMonConfig::CAT) {

					if (MonType==JsonMonConfig::TYPEINT) {
						jptr.reset(new HistoJ<int>(expectedUpdates,maxUpdates));
					}
					if (MonType==JsonMonConfig::TYPEDOUBLE) {
						jptr.reset(new HistoJ<double>(expectedUpdates,maxUpdates));
					}
					if (MonType==JsonMonConfig::TYPESTRING) {
						jptr.reset(new HistoJ<double>(expectedUpdates,maxUpdates));
					}
				}
				else {
					if (MonType==JsonMonConfig::TYPEINT)
						jptr.reset(new IntJ());
					if (MonType==JsonMonConfig::TYPEDOUBLE)
						jptr.reset(new DoubleJ());
					if (MonType==JsonMonConfig::TYPESTRING)
						jptr.reset(new StringJ());
				}
			}
			//WARNING: can have non-initialized unique_ptr (this needs to be checked later)
			dataNative_.push_back(std::move(jptr));
		}
}

DataPoint::~DataPoint() {
}
*/


/*
 *
 * Method implementation for simple DataPoint usage
 *
 */


void DataPoint::serialize(Json::Value& root) const {

	root[SOURCE] = getSource();
	root[DEFINITION] = getDefinition();
	for (unsigned int i=0;i<getData().size();i++)
		root[DATA].append(getData()[i]);
}


void DataPoint::deserialize(Json::Value& root) {
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
 * Method implementation for new monitoring
 *
 * */

//initialization
void DataPoint::trackMonitorable(JsonMonitorable *monitorable,bool NAifZeroUpdates)
{
    name_=name;
    tracked_ = (void*)monitorable;
    if (dynamic_cast<IntJ*>(monitorable)) monType=OPINT;
    if (dynamic_cast<DoubleJ*>(monitorable)) monType=OPDOUBLE;
    NAifZeroUpdates_=NAifZeroUpdates;

}
void DataPoint::trackMonitorableUInt(std::string const& name, std::vector<unsigned int>  monvec, bool NAifZeroUpdates)
{
    name_=name;
    tracked_ = (void*)monvec;
    isStream_=true;
    monType_=OPINT;
    NAifZeroUpdates_=NAifZeroUpdates;
    makeVector(monvec->size());
}
void DataPoint::trackMonitorableUIntAtomic(std::string const& name, std::vector<std::atomic<unsigned int>>  *monvec, bool NAifZeroUpdates)
{
    name_=name;
    tracked_ = (void*)monvec;
    isStream_=true;
    isAtomic_=true;
    monType_=OPINT;
    NAifZeroUpdates_=NAifZeroUpdates;
    makeVector(monvec->size());
}

void trackDummy(std::string const& name)
{
    name_ = name;
}

//TODO:strings and double
void DataPoint::makeQueue(unsigned int size) {
    for (unsigned int i=0;i<size;i++) {
	    streamData_.push_back(std::map<unsigned int,JsonMonitorable> newMap);
//	    streamData_.push_back(std::queue<JsonMonitorable> q);
//	    queuedStreamLumi_.push_back(std::queue<unsigned int> q);
    }
}

void DataPoint::serialize(Json::Value& root, bool rootInit, std::string const&input) const {

	if (rootInit) {
	  root[SOURCE] = getSource();
	  root[DEFINITION] = getDefinition();
	}
	root[DATA].append(input);
}

void DataPoint::setOperation(OperationType op)
{
    opType_=op;
}

void DataPoint::setStreamLumiPtr(std::vector<std::atomic<unsigned int>> *streamLumisPtr)
{
  streamLumisPtr_=streamLumisPtr;
}

void DataPoint::snap(unsigned int lumi)
{
  isCached_=false;
  if (isStream_) {
    if (monType_==TYPEINT)
    {
      for (unsigned int i=0; i<streamDataMaps_.size();i++) {
	//TODO:stream lumis don't need to be atomic, protected by lock
	unsigned int streamLumi_=streamLumisPtr_[i];//get currently processed stream lumi
	unsigned int monVal;
	if (isAtomic_) monVal = ((std::vector<std::atomic<unsigned int>>*)tracked_)->at(i);
	else monVal = ((std::vector<unsigned int>*)tracked_)->at(i);

	auto itr =  streamDataMaps_[i].find(streamLumi_);
	if (itr==std::map::end) //insert
	{
	  if (opType_==OPHISTO) {
	    streamDataMaps_[i][streamLumi_] = HistoJ<unsigned int> hj(1,MAXUPDATES);
	    streamDataMaps_[i][streamLumi_].update(monVal)
	  }
	  else {//all other default to SUM
	    streamDataMaps_[i][streamLumi_]= IntJ ij;
	    streamDataMaps_[i][streamLumi_].update(monVal);
	  }
	}
	else { 
	  if (opType_==OPHISTO)
	    std::static_cast<HistoJ<unsigned int>>(*itr).update(monVal);
	  else
	    std::static_cast<IntJ>(*itr)=monVal;
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
  //atomic currently not implemented
  auto itr = globalDataMap_.find(lumi);
  if (itr==std::map::end) {
    if (monType==TYPEINT) {
      IntJ ij;
      ij.update(((IntJ*)tracked_)->value());
      globalDataMap_[lumi]=dj;
    }
    if (monType==TYPEDOUBLE) {
      DoubleJ dj;
      dj.update(((DoubleJ*)tracked_)->value());
      globalDataMap_[lumi]=dj;
    }
  } else { 
  if (monType==TYPEINT)
    ((IntJ)*itr).update(((IntJ*)tracked_)->value());
  else if (monType==TYPEDOUBLE)
    ((DoubleJ)*itr).update(((DoubleJ*)tracked_)->value());
  }
}

void DataPoint::snapStream(unsigned int streamID, unsigned int lumi)
{
  if (!isStream_) return;
  if (!isAtomic_) return;
  isCached_=false;
  if (monType_==TYPEINT)
  {
      unsigned int monVal;
      if (isAtomic_) monVal = ((std::vector<std::atomic<unsigned int>>*)tracked_)->at(i);
      else monVal = ((std::vector<unsigned int>*)tracked_)->at(i);

	auto itr =  streamDataMaps_[streamID].find(lumi);
	if (itr==std::map::end) //insert
	{
	  if (opType_==OPHISTO) {
	    streamDataMaps_[streamID][lumi] = HistoJ<unsigned int> hj(1,MAXUPDATES);
	    streamDataMaps_[streamID][lumi].update(monVal)
	  }
	  else {//all other default to SUM
	    streamDataMaps_[streamID][lumi]= IntJ ij;
	    streamDataMaps_[streamID][lumi].update(monVal);
	  }
	  else 
	  { 
	    if (opType_==OPHISTO)
	      std::static_cast<HistoJ<unsigned int>>(*itr).update(monVal);
	    else
	      std::static_cast<IntJ>(*itr)=monVal;
	  }
	}
      }
    }
    else assert(monType_!=TYPEINT);//not yet implemented, application error
}

std::string DataPoint::fastOutLatest()
{
  return std::string("Not implemented");//for now empty
}

//TODO: implement, cache result etc.
JsonMonitorable *mergeAndRetrieveValue(unsigned int lumi)
{
  assert(varType_==TYPEINT && isStream_);//for now only support stream ints, also always do sum
  IntJ newJ = new IntJ;
  //histogram expand
  //unsigned int haveForLumis_=0;
  for (int i=0;i<streamDataMaps_.size();i++) {
    //per lumi map find so not that critical
    auto itr = streamDataMaps[i].find(lumi);
    //haveForLumis_++;
    if (itr!=std::map:end) newJ->update(*itr);
  }
  cacheI_=newJ->value();
  isCached_=true;
  return newJ;//assume the caller takes care of deleting the object
}

void mergeAndSerialize(Json::Value & root,unsigned int lumi,bool initJsonValue)
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
		//just append latest value
		//TODO: implement "SAME"!
		auto itr = globalDataMap_.find(lumi);
	  if (itr != std::map:end) {
	    root[DATA].append(*itr.toString);
	  }
	  else {
	    if (NAifZeroUpdates_) root[DATA].append("N/A");
	    else if (monType==TYPESTRING)  root[DATA].append("");
	    else  root[DATA].append("0");
	  }
	  return;
	}
	else {
		assert(monType==OPINT);//for now only this is supported
		if (isCached_) {
			std::stringstream ss;
			ss << cacheI_;
			root[DATA].append(ss.str());
			return;
		}
		if (optype==OPSUM) {
			IntJ tmpJ;
			std::stringstream ss;
			unsigned int updates=0;
			unsigned int sum=0;
			for (unsigned int i=0;i<streamDataMaps_.size();i++) {
				//per lumi map find so not that critical
				auto itr = streamDataMaps[i].find(lumi);
				if (itr!=std::map:end) {sum+=*itr;updates++;}
			}
			if (!updates && NAifZeroUpdates_) ss << "N/A";
			ss << sum;
			root[DATA].append(ss.str());
			return;
		}
		if (opType==OPHISTO) {
			if (nBinsPtr_==nullptr) {
				root[DATA].append("N/A");
				return;
			}
			if (*nBinsPtr_>bufLen_) {
				if (buf_) delete buf_;
				bufLen_=*nBinsPtr_;
				buf_= new uint32_t[bufLen_];
				//memset(buf_,0,bufLen_*sizeof(uint32_t));
			}
			memset(buf_,0,bufLen_*sizeof(uint32_t));
			//histogram populate..(& bounds check)
			unsigned int updates=0;
			for (unsigned int i=0;i<streamDataMaps_.size();i++) {
				auto itr = streamDataMaps[i].find(lumi);
				if (itr!=std::map:end) {
					updates+=*itr.getUpdates();
					for (ith : *itr.value()) {
						unsigned int thisbin=(unsigned int) *ith;
						if (thisbin<bufLen_ && thisbin>0)
							buf_[thisbin]++;
					}
				}
			}
			std::stringstream ss;
			if (!updates && NAifZeroUpdates_) ss << "N/A";
			else {
				std::stringstream ss;
				ss << "[";
				if (bufLen_) {
					for (unsigned int i=0;i<bufLen_-1;i++) {
						ss << buf_[i] << ",";
					}
					ss << buf_[buflen-1];
				}
				ss << "]";
			}
			root[DATA].append(ss.str());
			return;
		}
	}
}
//TODO:caching
//  if (cached_==(int)lumi) {
//}
//else {

/TODO

		//root[DATA].append(DataPoint::mergeAndSerializeMonitorable(root,dataPoints,config,index);

//TODO: check if we should in some cases return [N/A], not "N/A"

//	if (!dataPoints.size()) return std::string("N/A");//no measurements - todo: [] if CAT/HISTO is requested ?
	if (operation==OPSUM || operation==OPAVG) {
		unsigned int totalUpdates=0;
		if (varType==TYPEINT) {
			unsigned int totalSum=0;
			for (unsigned int i=0;i< dataPoints.size();i++) {
				IntJ * collected = std::static_cast<IntJ*>(dataPoints[i]->monitorableAt(index));
				if (collected) {
					totalSum+=collected->value();
					totalUpdates+=collected->getUpdates();
				}
			}
			if (config[index].NAifZero() && !totalUpdates) return std::string("N/A");
			if (operation==OPAVG)
				totalSum = totalUpdates? totalSum/totalUpdates : 0;
			ss << totalSum;
		}
		else if (varType==TYPEDOUBLE) {
			double totalSum=0;
			for (unsigned int i=0;i< dataPoints.size();i++) {
				DoubleJ * collected = std::static_cast<DoubleJ*>(dataPoints[i]->monitorableAt(index));
				if (collected) {
					totalSum+=collected->value();
					totalUpdates+=collected->getUpdates();
				}
			}
			if (config[index].NAifZero() && !totalUpdates) return std::string("N/A");
			if (operation==OPAVG)
				totalSum = totalUpdates? totalSum/totalUpdates : 0;
			ss << totalSum;
		}
		else if (varType==TYPESTRING)
			return std::string("N/A");//wrong definition
		else if (varType==TYPEHISTOINT || varType==TYPEHISTODOUBLE) assert(0);//shouldn't be here
		else
			assert(0); //shouldn't be here
		return ss.str();
	}
	else if (operation==OPSAME) {
		unsigned int count=0;
		unsigned int totalUpdates=0;
		if (varType==TYPEINT) {
			unsigned int notSameValue=0;
			for (unsigned int i=0;i< dataPoints.size();i++) {
				IntJ * collected = std::static_cast<IntJ*>(dataPoints[i]->monitorableAt(index));
				if (collected) {
					int totalUpdatesOld=totalUpdates;
					totalUpdates+=collected->getUpdates();
					bool notSame = collected->notSame();
					if (notSame) return std::string("N/A");
					if (i==0) { 
						notSameValue = collected->value();
					}
					//this is N/A only in case previous thread has updated
					else if (notSameValue!=collected->value() && totalUpdatesOld) return std::string("N/A");
				}
			}
			if (config[index].NAifZero() && !totalUpdates) return std::string("N/A");
			ss << notSameValue;
		}
		else if (varType==TYPEDOUBLE) {
			double notSameValue=0;
			for (unsigned int i=0;i< dataPoints.size();i++) {
				DoubleJ * collected = std::static_cast<DoubleJ*>(dataPoints[i]->monitorableAt(index));
				if (collected) {
					int totalUpdatesOld=totalUpdates;
					totalUpdates+=collected->getUpdates();
					bool notSame = collected->notSame();
					if (notSame) return std::string("N/A");
					if (i==0) { 
						notSameValue = collected->value();
					}
					//this is N/A only in case previous thread has updated
					else if (notSameValue!=collected->value() && totalUpdatesOld) return std::string("N/A");
				}
			}
			if (config[index].NAifZero() && !totalUpdates) return std::string("N/A");
			ss << notSameValue;
		}
		else if (varType==TYPESTRING) {
		}
		else if (varType==TYPEHISTOINT || varType==TYPEHISTODOUBLE)
			return std::string("N/A");//wrong definition
		else
			assert(0);//shouldn't be here
		return ss.str();
	}
	else if (operation==OPHISTO) {
		if (varType==TYPEHISTOINT) {
			auto bufPtr = config[index].getBinBuffer();
			unsigned int bins = config[index].getNBins();
			memset(bufPtr.get(),0,bins*sizeof(unsigned int));
			for (unsigned int i=0;i< dataPoints.size();i++) {
				HistoJ<int> * collected = std::static_cast<HistoJ<int> *>(dataPoints[i]->monitorableAt(index));
				if (collected) {
					std::vector<int> & histo = collected->value();
					for (unsigned int j=0;j<histo.size();j++) {
						idx = histo[j];
						if (idx>=0 && idx<bins) bufPtr[idx]++; //only [0,bin) accepted
					}
				}
				//serialize
				ss << "[";
				if (bins) {
					for (unsigned int i=0;i< bins-1 ;i++) {
						ss << bufPtr[i] << ",";
					}
					ss << bufPtr[bins-1];
				}
				ss << "]";
			}
			else if (varType==TYPEHISTODOUBLE) {
				assert(0); //shouldn't be here
			}
			else if (varType==TYPEINT || varType==TYPEDOUBLE)
				assert(0);//shouldn't be here
			else if (varType==TYPESTRING)
				return std::string("N/A");//wrong definition //todo:[] ?
			else
			assert(0);//shouldn't be here
		return ss.str();
	}
	return std::string("N/A");//unknown operation
}
