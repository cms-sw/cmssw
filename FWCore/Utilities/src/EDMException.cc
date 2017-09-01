
#include "FWCore/Utilities/interface/EDMException.h"

#define FILLENTRY( name ) {name, #name }

namespace edm {
  namespace errors {
    static const std::map<ErrorCodes, std::string> codeMap =
    {
      FILLENTRY(CommandLineProcessing),
      FILLENTRY(ConfigFileNotFound),
      FILLENTRY(ConfigFileReadError),
      FILLENTRY(OtherCMS),
      FILLENTRY(StdException),
      FILLENTRY(Unknown),
      FILLENTRY(BadAlloc),
      FILLENTRY(BadExceptionType),
      FILLENTRY(ProductNotFound),
      FILLENTRY(DictionaryNotFound),
      FILLENTRY(NoProductSpecified),
      FILLENTRY(InsertFailure),
      FILLENTRY(Configuration),
      FILLENTRY(LogicError),
      FILLENTRY(UnimplementedFeature),
      FILLENTRY(InvalidReference),
      FILLENTRY(NullPointerError),
      FILLENTRY(EventTimeout),
      FILLENTRY(EventCorruption),
      FILLENTRY(ScheduleExecutionFailure),
      FILLENTRY(EventProcessorFailure),
      FILLENTRY(FileInPathError),
      FILLENTRY(FileOpenError),
      FILLENTRY(FileReadError),
      FILLENTRY(FatalRootError),
      FILLENTRY(MismatchedInputFiles),
      FILLENTRY(ProductDoesNotSupportViews),
      FILLENTRY(ProductDoesNotSupportPtr),
      FILLENTRY(NotFound),
      FILLENTRY(FormatIncompatibility),
      FILLENTRY(FallbackFileOpenError),
      FILLENTRY(ExceededResourceVSize),
      FILLENTRY(ExceededResourceRSS),
      FILLENTRY(ExceededResourceTime),
      FILLENTRY(CaughtSignal)
    };
    static const std::string kUnknownCode("unknownCode");
  }
  /// -------------- implementation details ------------------

  const std::string& Exception::codeToString(errors::ErrorCodes c) {
    auto i(errors::codeMap.find(c));
    return i!=errors::codeMap.end() ? i->second : errors::kUnknownCode;
  }

  Exception::Exception(errors::ErrorCodes aCategory):
    cms::Exception(codeToString(aCategory)),
    category_(aCategory) {
  }

  Exception::Exception(errors::ErrorCodes aCategory, std::string const& message):
    cms::Exception(codeToString(aCategory),message),
    category_(aCategory) {
  }

  Exception::Exception(errors::ErrorCodes aCategory, char const* message):
    cms::Exception(codeToString(aCategory), std::string(message)),
    category_(aCategory) {
  }

  Exception::Exception(errors::ErrorCodes aCategory, std::string const& message, cms::Exception const& another):
    cms::Exception(codeToString(aCategory),message,another),
    category_(aCategory) {
  }

  Exception::Exception(errors::ErrorCodes aCategory, char const* message, cms::Exception const& another):
    cms::Exception(codeToString(aCategory), std::string(message), another),
    category_(aCategory) {
  }

  Exception::Exception(Exception const& other):
    cms::Exception(other),
    category_(other.category_) {
  }

  Exception::~Exception() noexcept {
  }

  Exception&
  Exception::operator=(Exception const& other) {
    Exception temp(other);
    this->swap(temp);
    return *this;
  }
  
  int
  Exception::returnCode_() const {
    return static_cast<int>(category_);
  }

  void
  Exception::throwThis(errors::ErrorCodes aCategory,
                       char const* message0,
                       char const* message1,
                       char const* message2,
                       char const* message3,
                       char const* message4) {
    Exception e(aCategory, std::string(message0));
    e << message1 << message2 << message3 << message4;
    throw e;
  }

  void
  Exception::throwThis(errors::ErrorCodes aCategory, char const* message0, int intVal, char const* message1) {
    Exception e(aCategory, std::string(message0));
    e << intVal << message1;
    throw e;
  }

  Exception* Exception::clone() const {
    return new Exception(*this);
  }

  void
  Exception::rethrow() {
    throw *this;
  }
}
