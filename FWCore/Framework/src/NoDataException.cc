#include <FWCore/Framework/interface/NoDataException.h>

namespace edm {
  namespace eventsetup {

     NoDataExceptionBase::NoDataExceptionBase(const EventSetupRecordKey& iRecordKey,
                                              const DataKey& iDataKey,
                                              const char* category_name) :
     cms::Exception(category_name),
     record_(iRecordKey),
     dataKey_(iDataKey)
     {
     }
     
     NoDataExceptionBase::~NoDataExceptionBase() throw() {}
     
     const DataKey& NoDataExceptionBase::dataKey() const { return dataKey_; }
     
     std::string NoDataExceptionBase::providerButNoDataMessage(const EventSetupRecordKey& iKey) {
        return std::string(" A provider for this data exists, but it's unable to deliver the data for this \"")
        +iKey.name()
        +"\" record.\n Perhaps no valid data exists for this IOV? Please check the data's interval of validity.\n";
     }
     
     std::string NoDataExceptionBase::noProxyMessage() {
        return std::string("Please add an ESSource or ESProducer to your job which can deliver this data.\n");
     }
     
     void NoDataExceptionBase::beginDataTypeMessage(std::string& oString) const
     {
        oString+= std::string("No data of type \"");
     }
     
     void NoDataExceptionBase::endDataTypeMessage(std::string& oString) const
     {
        oString += "\" with label \"";
        oString += this->dataKey_.name().value();
        oString += "\" in record \"";
        oString += this->record_.name();
        oString += "\"";
     }
     
     void NoDataExceptionBase::constructMessage(const char* iClassName, const std::string& iExtraInfo)
     {
        std::string message;
        beginDataTypeMessage(message);
        message += iClassName;
        endDataTypeMessage(message);
        this->append(message+std::string("\n "));
        this->append(iExtraInfo);
     }
     
  }
}
