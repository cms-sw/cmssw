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



using namespace jsoncollector;

template class HistoJ<int>;
template class HistoJ<double>;

const string DataPoint::SOURCE = "source";
const string DataPoint::DEFINITION = "definition";
const string DataPoint::DATA = "data";

DataPoint::DataPoint(
		std::vector<JsonMonitorable*> const& monVars, 
		std::vector<JsonMonConfigData> const& monConfig,
		unsigned int expectedUpdates, unsigned int maxUpdates):
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

//Serialized and deserializer to/from string array

void DataPoint::serialize(Json::Value& root) const {
	root[SOURCE] = getSource();
	root[DEFINITION] = getDefinition();
	else {
		for (unsigned int i = 0; i < getData().size(); i++)
			root[DATA].append(getData()[i]);
	}
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

//snap and in-place merge: new
//todo: maybe speed this up very slightly by not using dynamic_cast stuff
void DataPoint::snap() {
	assert(dataNative_.size()==monitored_->size());
	for (unsigned int i=0;i<dataNative_.size();i++) {
		//if variable is in JSON but not found in the FastMonitor (e.g. loading wrong file...)
		if (dataNative_[i].get()) {
			//histo and cat for int are same at this stage
			//histo is only possible for int
			JsonMonConfig::OperationType operation = = dpdPtr_->getLegendFor(i).getOperation();
			if (operation==JsonMonConfig::OPHISTO || operation==JsonMonConfig::OPCAT) {
				//histo is supported only for ints (they denote bin)
				auto inputInt = dynamic_cast<IntJ*>(monitored_->at(i));
				if (inputInt) {
					HistoJ<int>* toFill = static_cast<HistoJ<int>*>( dataNative_[i].get());
					if (toFill) {
						toFill->update(inputInt->value());
					}
				}
			}
			//cat for double uses histogram class
			else if (operation==JsonMonConfig::OPCAT) {
				auto inputDouble = dynamic_cast<DoubleJ*>(monitored_->at(i));
				if (inputDouble) {
					HistoJ<double>* toFill = static_cast<HistoJ<double>*>( dataNative_[i].get());
					if (toFill) {
						toFill->update(inputDouble->value());
					}
				}
				else {
					//in-place concatenation only for strings
					auto inputString = dynamic_cast<StringJ*>(monitored_->at(i));
					if (inputString) {
						static_cast<StringJ*>( dataNative_[i].get())->concatenate(inputString->value());
					}
				}
			}
			//applies to int,double, both operations same at this time
			else if (operation==JsonMonConfig::OPAVG || operation==JsonMonConfig::OPSUM) {
				auto inputInt = dynamic_cast<IntJ*>(monitored_->at(i));
				if (inputInt)
					static_cast<IntJ*>( dataNative_[i].get())->add(inputInt->value());
				else {
					auto inputDouble = dynamic_cast<DoubleJ*>(monitored_->at(i));
					if (inputDouble)
						static_cast<DoubleJ*>( dataNative_[i].get())->add(inputDouble->value());

				}
			}
			//applies to all types
			else if (operation==JsonMonConfig::OPSAME) {
				auto inputInt = dynamic_cast<IntJ*>(monitored_->at(i));
				if (inputInt)
					static_cast<IntJ*>( dataNative_[i].get())->compare(inputInt->value());
				else {
					auto inputDouble = dynamic_cast<DoubleJ*>(monitored_->at(i));
					if (inputDouble)
						static_cast<DoubleJ*>( dataNative_[i].get())->compare(inputDouble->value());
					else {
						auto inputString = dynamic_cast<StringJ*>(monitored_->at(i));
						if (inputString)
							static_cast<StringJ*>( dataNative_[i].get())->compare(inputString->value());
					}
				}
			}
		}
	}
	updates_++;
}

//static members (maybe move elsewhere)

void DataPoint::serialize( tbb::concurrent_vector<DataPoint*> & dataPoints, std::vector<JsonMonitorableConfig>& config) {

	Json::Value root;

	root[SOURCE] = *config[0].getSourceInfoPtr();
	root[DEFINITION] = *config[0].getDefinitionPtr();
	for (unsigned int index=0;index<config.size();index++)
		root[DATA].append(DataPoint::mergeAndSerializeMonitorable(root,dataPoints,config,index);

	Json::StyledWriter writer;
	output = writer.write(root);
}

//TODO: check if we should in some cases return [N/A], not "N/A"
std::string DataPoint::mergeAndSerializeMonitorable(
		tbb::concurrent_vector<DataPoint*> & dataPoints,
		std::vector<JsonMonitorableConfig>& config,
		unsigned int index) {

	if (!dataPoints.size()) return std::string("N/A");//no measurements - todo: [] if CAT/HISTO is requested ?
	MonType varType = config[index].getMonType();
	OperationType operation = config[index].getOperationType();
	std::stringstream ss;	
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
			std::string * notSameValue;//pointer to avoid a copy
			for (unsigned int i=0;i< dataPoints.size();i++) {
				StringJ * collected = std::static_cast<StringJ*>(dataPoints[i]->monitorableAt(index));
				if (collected) {
					int totalUpdatesOld=totalUpdates;
					totalUpdates+=collected->getUpdates();
					bool notSame = collected->notSame();
					if (notSame) return std::string("N/A");
					if (i==0) { 
						notSameValue = & collected->value();
					}
					//this is N/A only in case previous thread has updated
					else if (*notSameValue!=collected->value() && totalUpdatesOld) return std::string("N/A");
				}
			}
			if (config[index].NAifZero() && !totalUpdates) return std::string("N/A");
			ss << *notSameValue;
		}
		else if (varType==TYPEHISTOINT || varType==TYPEHISTODOUBLE)
			return std::string("N/A");//wrong definition
		else
			assert(0);//shouldn't be here
		return ss.str();
	}
	else if (operation==OPCAT) {
		if (varType==TYPESTRING) {
			//unsigned int totalUpdates=0;
			bool wasUpdated=false;
			ss << "[";
			for (unsigned int i=0;i< dataPoints.size();i++) {
				StringJ * collected = std::static_cast<StringJ*>(dataPoints[i]->monitorableAt(index));
				//totalUpdates+=collected->getUpdates();
				if (collected) {
					std::string& str = collected->value();
					if (str.size()) {
						if (wasUpdated) ss << ",";
						else wasUpdated=true;
						ss << str; //todo: what if some/all updates are empty ?
					}
				}
			}
			ss << "]";
			if (config[index].NAifZero() && !wasUpdated) return std::string("N/A"); //todo:[] ?
			return ss.str();
		}
		else if (varType==TYPEHISTOINT) {
			//unsigned int totalUpdates=0;
			bool wasUpdated=false;
			ss << "[";
			for (unsigned int i=0;i< dataPoints.size();i++) {
				HistoJ<int> * collected = std::static_cast<HistoJ<int> *>(dataPoints[i]->monitorableAt(index));
				if (collected) {
					std::vector<int> & histo = collected->value();
					if (histo.size()) {
						if (wasUpdated) ss << ",";
						else wasUpdated=true;
						for (unsigned int i=0;i<histo.size()-1;i++) {
							ss << histo[i] << ",";
						}
						ss << histo.size()-1;
					}
				}
			}
			ss << "]";
			if (config[index].NAifZero() && !wasUpdated) return std::string("N/A"); //todo:[] ?
			return ss.str();
		}
		else if (varType==TYPEHISTODOUBLE) {
			//unsigned int totalUpdates=0;
			bool wasUpdated=false;
			ss << "[";
			for (unsigned int i=0;i< dataPoints.size();i++) {
				HistoJ<double> * collected = std::static_cast<HistoJ<double> *>(dataPoints[i]->monitorableAt(index));
				if (collected) {
					std::vector<double> & histo = collected->value();
					if (histo.size()) {
						if (wasUpdated) ss << ",";
						else wasUpdated=true;
						for (unsigned int i=0;i<histo.size()-1;i++) {
							ss << histo[i] << ",";
						}
						ss << histo.size()-1;
					}
				}
			}
			ss << "]";
			if (config[index].NAifZero() && !wasUpdated) return std::string("N/A"); //todo:[] ?
			return ss.str();

		}
		else if (varType==TYPEINT || varType==TYPEDOUBLE)
			assert(0);//shouldn't be here
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
