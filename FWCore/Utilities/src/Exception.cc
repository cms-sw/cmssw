
#include "FWCore/Utilities/interface/Exception.h"

namespace cms {

  Exception::Exception(std::string const& aCategory) :
    std::exception(),
    ost_(),
    category_(1, aCategory)
  {
  }

  Exception::Exception(std::string const& aCategory,
		       std::string const& message) :
    std::exception(),
    ost_(),
    category_(1, aCategory)
  {
    ost_ << message;
    if(!message.empty()) {
	unsigned sz = message.size()-1;
	if(message[sz]!='\n' && message[sz]!=' ') ost_ << " ";
    }
  }

  Exception::Exception(std::string const& aCategory,
		       std::string const& message,
		       Exception const& another) :
    std::exception(),
    ost_(),
    category_(1, aCategory)
  {
    ost_ << message;
    // check for newline at end of message first
    if(!message.empty() && message[message.size()-1]!='\n')
      ost_ << "\n";
    category_.push_back(another.category());
    append(another);
  }

  Exception::Exception(Exception const& other):
    std::exception(),
    ost_(),
    category_(other.category_)
  {
    ost_ << other.ost_.str();
  }

  Exception::~Exception() throw() {
  }
  
  std::string Exception::explainSelf() const {
    std::ostringstream ost;
    std::string part(ost_.str());
    // Remove any trailing newline

    ost << "---- " << category() << " BEGIN\n"
	<< part;

    if(!part.empty() && part[part.size()-1]!='\n') ost << "\n";
    ost << "---- " << category() << " END\n";

    return ost.str();
  }

  Exception::CategoryList const& Exception::history() const {
    return category_;
  }

  std::string Exception::category() const {
    return category_.front();
  }
  
  std::string Exception::rootCause() const {
    return category_.back();
  }
  
  void Exception::append(Exception const& another) {
    ost_ << another.explainSelf();
  }

  void Exception::append(std::string const& more_information) {
    ost_ << more_information;
  }

  void Exception::append(char const* more_information) {
    ost_ << more_information;
  }
  
  // --------------------------------
  // required for being a std::exception

  char const* Exception::what() const throw() {
    return explainSelf().c_str();;
  }

  std::exception* Exception::clone() const {
    return new Exception(*this);
  }

  void Exception::rethrow() {
    throw *this;
  }
  


}
