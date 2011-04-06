
#include "FWCore/Utilities/interface/Exception.h"

namespace cms {

  Exception::Exception(std::string const& aCategory) :
    std::exception(),
    ost_(),
    category_(aCategory),
    what_(),
    context_(),
    additionalInfo_()
  {
  }

  Exception::Exception(char const* aCategory)  :
    std::exception(),
    ost_(),
    category_(std::string(aCategory)),
    what_(),
    context_(),
    additionalInfo_()
  {
  }

  Exception::Exception(std::string const& aCategory,
		       std::string const& message) :
    std::exception(),
    ost_(),
    category_(aCategory),
    what_(),
    context_(),
    additionalInfo_()
  {
    init(message);
  }

  Exception::Exception(char const* aCategory,
	               std::string const& message) :
    std::exception(),
    ost_(),
    category_(std::string(aCategory)),
    what_(),
    context_(),
    additionalInfo_()
  {
    init(message);
  }


  Exception::Exception(std::string const& aCategory,
                       char const* message) :
    std::exception(),
    ost_(),
    category_(aCategory),
    what_(),
    context_(),
    additionalInfo_()
  {
    init(std::string(message));
  }


  Exception::Exception(char const* aCategory,
                       char const* message) :
    std::exception(),
    ost_(),
    category_(std::string(aCategory)),
    what_(),
    context_(),
    additionalInfo_()
  {
    init(std::string(message));
  }

  void Exception::init(std::string const& message) {
    ost_ << message;
    if(!message.empty()) {
	unsigned sz = message.size()-1;
	if(message[sz] != '\n' && message[sz] != ' ') ost_ << " ";
    }
  }

  Exception::Exception(std::string const& aCategory,
                       std::string const& message,
                       Exception const& another) :
    std::exception(),
    ost_(),
    category_(aCategory),
    what_(),
    context_(another.context()),
    additionalInfo_(another.additionalInfo())
  {
    ost_ << message;
    // check for newline at end of message first
    if(!message.empty() && message[message.size()-1]!='\n') {
      ost_ << "\n";
    }
    append(another);
  }

  Exception::Exception(Exception const& other):
    std::exception(),
    ost_(),
    category_(other.category_),
    what_(other.what_),
    context_(other.context_),
    additionalInfo_(other.additionalInfo_)
  {
    ost_ << other.ost_.str();
  }

  Exception::~Exception() throw() {
  }
  
  char const* Exception::what() const throw() {
    what_ = explainSelf();
    return what_.c_str();
  }

  std::string Exception::explainSelf() const {
    std::ostringstream ost;

    if (context_.empty()) {
      ost << "An exception of category '" << category_ << "' occurred.\n";
    }
    else {
      ost << "An exception of category '" << category_ << "' occurred while\n";
      int count = 0;
      for (std::list<std::string>::const_reverse_iterator i = context_.rbegin(),
	     iEnd = context_.rend();
           i != iEnd; ++i, ++count) {
        ost << "   [" << count << "] " << *i << "\n";
      }
    }

    std::string centralMessage(ost_.str());
    if (!centralMessage.empty()) {
      ost << "Exception Message:\n";
      ost << centralMessage;
      if (centralMessage[centralMessage.size() - 1] != '\n') {
        ost << "\n";
      }
    }

    if (!additionalInfo_.empty()) {
      ost << "   Additional Info:\n";
      char c = 'a';
      for (std::list<std::string>::const_reverse_iterator i = additionalInfo_.rbegin(),
	     iEnd = additionalInfo_.rend();
           i != iEnd; ++i, ++c) {
        ost << "      [" << c << "] " << *i << "\n";
      }
    }
    return ost.str();
  }

  std::string const& Exception::category() const {
    return category_;
  }
  
  std::string Exception::message() const {
    return ost_.str();
  }

  std::list<std::string> const& Exception::context() const {
    return context_;
  }

  std::list<std::string> const& Exception::additionalInfo() const {
    return additionalInfo_;
  }

  int  Exception::returnCode() const {
    return returnCode_();
  }

  void Exception::append(Exception const& another) {
    ost_ << another.message();
  }

  void Exception::append(std::string const& more_information) {
    ost_ << more_information;
  }

  void Exception::append(char const* more_information) {
    ost_ << more_information;
  }

  void Exception::clearMessage() {
    ost_.str("");
  }

  void Exception::clearContext() {
    context_.clear();
  }

  void Exception::clearAdditionalInfo() {
    additionalInfo_.clear();
  }

  void Exception::addContext(std::string const& context) {
    context_.push_back(context);
  }

  void Exception::addContext(char const* context) {
    context_.push_back(std::string(context));
  }

  void Exception::addAdditionalInfo(std::string const& info) {
    additionalInfo_.push_back(info);
  }

  void Exception::addAdditionalInfo(char const* info) {
    additionalInfo_.push_back(std::string(info));
  }

  void Exception::setContext(std::list<std::string> const& context) {
    context_ = context;
  }

  void Exception::setAdditionalInfo(std::list<std::string> const& info) {
    additionalInfo_ = info;
  }

  Exception* Exception::clone() const {
    return new Exception(*this);
  }

  void Exception::rethrow() {
    throw *this;
  }

  int  Exception::returnCode_() const {
    return 8001;
  }

  std::list<std::string> Exception::history() const {
    std::list<std::string> returnValue;
    returnValue.push_back(category_);
    return returnValue;
  }
}
