#include <set>
#include <sstream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/SourceFactory.h"

namespace edm {

  class EmptyESSource : public EventSetupRecordIntervalFinder {
  public:
    EmptyESSource(ParameterSet const&);
    EmptyESSource(EmptyESSource const&) = delete;                   // stop default
    EmptyESSource const& operator=(EmptyESSource const&) = delete;  // stop default

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void setIntervalFor(eventsetup::EventSetupRecordKey const&,
                        IOVSyncValue const& iTime,
                        ValidityInterval& oInterval) final;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    bool isConcurrentFinder() const final { return true; }
    void delaySettingRecords() final;

    static std::set<IOVSyncValue> makeSetOfIOV(std::vector<unsigned int> const& iValues, bool iovIsTime) {
      std::set<IOVSyncValue> returnValue;
      for (auto value : iValues) {
        if (iovIsTime) {
          returnValue.insert(IOVSyncValue(Timestamp(value)));
        } else {
          returnValue.insert(IOVSyncValue(EventID(value, 0, 0)));
        }
      }
      return returnValue;
    }
    // ---------- member data --------------------------------
    std::string const recordName_;
    bool const iovIsTime_;
    std::set<IOVSyncValue> const setOfIOV_;
  };

  EmptyESSource::EmptyESSource(ParameterSet const& pset)
      : recordName_(pset.getParameter<std::string>("recordName")),
        iovIsTime_(!pset.getParameter<bool>("iovIsRunNotTime")),
        setOfIOV_(makeSetOfIOV(pset.getParameter<std::vector<unsigned int>>("firstValid"), iovIsTime_)) {}

  void EmptyESSource::delaySettingRecords() {
    eventsetup::EventSetupRecordKey recordKey = eventsetup::EventSetupRecordKey::TypeTag::findType(recordName_);
    if (recordKey == eventsetup::EventSetupRecordKey()) {
      throw edm::Exception(errors::Configuration)
          << " The Record type named \"" << recordName_ << "\" could not be found. Please check the spelling. \n"
          << "If the spelling is fine, then no module in the job requires this Record and therefore EmptyESSource can "
             "not function.\n"
             "In such a case please either remove the EmptyESSource with label'"
          << descriptionForFinder().label_ << "' from your job or add a module which needs the Record to your job.";
    }
    findingRecordWithKey(recordKey);
  }

  void EmptyESSource::setIntervalFor(eventsetup::EventSetupRecordKey const&,
                                     IOVSyncValue const& iTime,
                                     ValidityInterval& oInterval) {
    oInterval = ValidityInterval::invalidInterval();
    //if no intervals given, fail immediately
    if (setOfIOV_.empty()) {
      return;
    }

    std::pair<std::set<IOVSyncValue>::iterator, std::set<IOVSyncValue>::iterator> itFound =
        setOfIOV_.equal_range(iTime);

    //we have overshot
    if (itFound.first == itFound.second) {
      if (itFound.first == setOfIOV_.begin()) {
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

    if (itFound.second != setOfIOV_.end()) {
      if (iovIsTime_) {
        endOfInterval = IOVSyncValue(Timestamp(itFound.second->time().value() - 1));
      } else {
        endOfInterval = IOVSyncValue(itFound.second->eventID().previousRunLastEvent(0));
      }
    }
    oInterval = ValidityInterval(*(itFound.first), endOfInterval);
  }

  void EmptyESSource::fillDescriptions(ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("recordName")
        ->setComment(
            "The name of the EventSetup Record for which intervals are to be generated.\n The record type must be used "
            "by some module in the job else an exception will occur.");
    desc.add<bool>("iovIsRunNotTime", true)->setComment("Sets how to interpret integer values used in 'firstValid'");
    desc.add<std::vector<unsigned int>>("firstValid")
        ->setComment(
            "Sets the beginning point of an IOV. The end point is assumed to be the next entry in the list. If there "
            "are no further entries than 'end of time' is used.");

    descriptions.addDefault(desc);
  }

}  // namespace edm
using edm::EmptyESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(EmptyESSource);
