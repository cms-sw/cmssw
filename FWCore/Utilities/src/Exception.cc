
#include "FWCore/FWUtilities/interface/Exception.h"

namespace cms {

  Exception::Exception(const std::string& category):
    ost_(),
    category_(1,category)
  {
  }

  Exception::Exception(const std::string& category,
		       const std::string& message):
    ost_(),
    category_(1,category)
  {
    ost_ << message;
  }

  Exception::Exception(const std::string& category,
		       const std::string& message,
		       const Exception& another):
    ost_(),
    category_(1,category)
  {
    ost_ << message;
    // check for newline at end of message first
    if(!message.empty() && message[message.size()-1]!='\n')
      ost_ << "\n";
    category_.push_back(another.category());
    append(another);
  }

  Exception::Exception(const Exception& other):
    ost_(),
    category_(other.category_)
  {
    ost_ << other.ost_.str();
  }

  Exception::~Exception()
  {
  }
  
  std::string Exception::what() const
  {
    std::ostringstream ost;
    std::string part(ost_.str());
    // Remove any trailing newline

    ost << "---- " << category() << " BEGIN\n"
	<< part;

    if(!part.empty() && part[part.size()-1]!='\n') ost << "\n";
    ost << "---- " << category() << " END\n";

    return ost.str();
  }

  const Exception::CategoryList& Exception::history() const
  {
    return category_;
  }

  std::string Exception::category() const
  {
    return category_.front();
  }
  
  void Exception::append(const Exception& another)
  {
    ost_ << another.what();
  }

  void Exception::append(const std::string& more_information)
  {
    ost_ << more_information;
  }

  void Exception::append(const char* more_information)
  {
    ost_ << more_information;
  }
  
  // --------------------------------
  // required for being a seal::Error

  std::string Exception::explainSelf() const
  {
    return what();
  }

  seal::Error* Exception::clone() const
  {
    return new Exception(*this);
  }

  void Exception::rethrow()
  {
    throw *this;
  }
  


}
