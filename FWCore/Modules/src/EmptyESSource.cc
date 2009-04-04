#include <set>
#include <sstream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/SourceFactory.h"

namespace edm {
   
class EmptyESSource : public  EventSetupRecordIntervalFinder {

   public:
      EmptyESSource(ParameterSet const&);
      //virtual ~EmptyESSource();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   void setIntervalFor(eventsetup::EventSetupRecordKey const&,
                        IOVSyncValue const& iTime, 
                        ValidityInterval& oInterval);
      
   private:
      EmptyESSource(EmptyESSource const&); // stop default

      EmptyESSource const& operator=(EmptyESSource const&); // stop default
      
      void delaySettingRecords();
      // ---------- member data --------------------------------
      std::string recordName_;
      std::set <IOVSyncValue> setOfIOV_;
      bool iovIsTime_;
};

EmptyESSource::EmptyESSource(ParameterSet const& pset) :
     recordName_(pset.getParameter<std::string>("recordName")),
     iovIsTime_(!pset.getParameter<bool>("iovIsRunNotTime")) {
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
   //copy_all(temp, inserter(setOfIOV_ , setOfIOV_.end()));
}
  
   
void 
EmptyESSource::delaySettingRecords() {
   eventsetup::EventSetupRecordKey recordKey = eventsetup::EventSetupRecordKey::TypeTag::findType(recordName_);
   if (recordKey == eventsetup::EventSetupRecordKey()) {
      throw edm::Exception(errors::Configuration) << " The Record type named \"" << recordName_
      << "\" could not be found. Please check the spelling. \n"
      << "If the spelling is fine, then no module in the job requires this Record and therefore EmptyESSource can not function.\n"
      "In such a case please either remove the EmptyESSource with label'"
      << descriptionForFinder().label_ << "' from your job or add a module which needs the Record to your job.";
   }
   findingRecordWithKey(recordKey);
}

void 
EmptyESSource::setIntervalFor(eventsetup::EventSetupRecordKey const&,
                               IOVSyncValue const& iTime, 
                               ValidityInterval& oInterval) {
   oInterval = ValidityInterval::invalidInterval();
   //if no intervals given, fail immediately
   if (setOfIOV_.size() == 0) {
      return;
   }
   
   std::pair< std::set<IOVSyncValue>::iterator, 
      std::set<IOVSyncValue>::iterator > itFound = setOfIOV_.equal_range(iTime);
   
   //we have overshot
   if(itFound.first == itFound.second) {
      if(itFound.first == setOfIOV_.begin()) {
         //request is before first valid interval, so fail
         return;
      }
      //go back one step
      --itFound.first;
   }
   if (itFound.first == setOfIOV_.end()) {
      return;
   }
   
   IOVSyncValue endOfInterval = IOVSyncValue::endOfTime();
   
   if(itFound.second != setOfIOV_.end()) {
      if(iovIsTime_) {
         endOfInterval = IOVSyncValue(Timestamp(itFound.second->time().value() - 1));
      } else {
         endOfInterval = IOVSyncValue(itFound.second->eventID().previousRunLastEvent());
      }
   }
   oInterval = ValidityInterval(*(itFound.first), endOfInterval);
}

}
using edm::EmptyESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(EmptyESSource);
