#include <sstream>

#include <stdexcept>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Modules/src/EmptyESSource.h"

namespace edm {

EmptyESSource::EmptyESSource(const edm::ParameterSet & pset) :
   recordName_(pset.getParameter<std::string>("recordName")),
   iovIsTime_(!pset.getParameter<bool>("iovIsRunNotTime"))
{
   std::vector<unsigned int> temp(pset.getParameter< std::vector<unsigned int> >("firstValid"));
   for(std::vector<unsigned int>::iterator itValue = temp.begin(), itValueEnd = temp.end();
        itValue != itValueEnd;
        ++itValue) {
      if(iovIsTime_) {
         setOfIOV_.insert(IOVSyncValue(Timestamp(*itValue)));
      } else {
         setOfIOV_.insert(IOVSyncValue(EventID(*itValue, 0)));
      }
   }
   //std::copy(temp.begin(), temp.end(), inserter(setOfIOV_ , setOfIOV_.end()));
   
   
   eventsetup::EventSetupRecordKey recordKey = eventsetup::EventSetupRecordKey::TypeTag::findType(recordName_);
   if (recordKey == edm::eventsetup::EventSetupRecordKey()) {
      std::ostringstream errorMessage;
      errorMessage<<" The Record type named \""<<recordName_<<"\" could not be found. Please check the spelling. \n"
         <<"If the spelling is fine, try to move the declaration of the TestingIntevalFinder to the end of the configuration file.";
      throw std::runtime_error(errorMessage.str().c_str());
   }
   findingRecordWithKey(recordKey);
}
  
void 
EmptyESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                               const edm::IOVSyncValue& iTime, 
                               edm::ValidityInterval& oInterval) {
   oInterval = edm::ValidityInterval::invalidInterval();
   //if no intervals given, fail immediately
   if (setOfIOV_.size() == 0) {
      return;
   }
   
   std::pair< std::set<edm::IOVSyncValue>::iterator, 
      std::set<edm::IOVSyncValue>::iterator > itFound = setOfIOV_.equal_range(iTime);
   
   //we have overshot
   if(itFound.first == itFound.second){
      if(itFound.first == setOfIOV_.begin()){
         //request is before first valid interval, so fail
         return;
      }
      //go back one step
      --itFound.first;
   }
   if (itFound.first == setOfIOV_.end()) {
      return;
   }
   
   edm::IOVSyncValue endOfInterval = edm::IOVSyncValue::endOfTime();
   
   if(itFound.second != setOfIOV_.end()) {
      if(iovIsTime_) {
         endOfInterval = edm::IOVSyncValue(Timestamp(itFound.second->time().value()-1));
      } else {
         endOfInterval = edm::IOVSyncValue(itFound.second->eventID().previousRunLastEvent());
      }
   }
   oInterval = edm::ValidityInterval(*(itFound.first), endOfInterval);
}

}
