
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  namespace errors {
    struct FilledMap {
      FilledMap();
      Exception::CodeMap trans_;
    };
    FilledMap::FilledMap() : trans_() {
      EDM_MAP_ENTRY_NONS(trans_, CommandLineProcessing);
      EDM_MAP_ENTRY_NONS(trans_, ConfigFileNotFound);
      EDM_MAP_ENTRY_NONS(trans_, ConfigFileReadError);
      EDM_MAP_ENTRY_NONS(trans_, OtherCMS);
      EDM_MAP_ENTRY_NONS(trans_, StdException);
      EDM_MAP_ENTRY_NONS(trans_, Unknown);
      EDM_MAP_ENTRY_NONS(trans_, BadAlloc);
      EDM_MAP_ENTRY_NONS(trans_, BadExceptionType);
      EDM_MAP_ENTRY_NONS(trans_, ProductNotFound);
      EDM_MAP_ENTRY_NONS(trans_, DictionaryNotFound);
      EDM_MAP_ENTRY_NONS(trans_, NoProductSpecified);
      EDM_MAP_ENTRY_NONS(trans_, InsertFailure);
      EDM_MAP_ENTRY_NONS(trans_, Configuration);
      EDM_MAP_ENTRY_NONS(trans_, LogicError);
      EDM_MAP_ENTRY_NONS(trans_, UnimplementedFeature);
      EDM_MAP_ENTRY_NONS(trans_, InvalidReference);
      EDM_MAP_ENTRY_NONS(trans_, NullPointerError);
      EDM_MAP_ENTRY_NONS(trans_, EventTimeout);
      EDM_MAP_ENTRY_NONS(trans_, EventCorruption);
      EDM_MAP_ENTRY_NONS(trans_, ScheduleExecutionFailure);
      EDM_MAP_ENTRY_NONS(trans_, EventProcessorFailure);
      EDM_MAP_ENTRY_NONS(trans_, FileInPathError);
      EDM_MAP_ENTRY_NONS(trans_, FileOpenError);
      EDM_MAP_ENTRY_NONS(trans_, FileReadError);
      EDM_MAP_ENTRY_NONS(trans_, FatalRootError);
      EDM_MAP_ENTRY_NONS(trans_, MismatchedInputFiles);
      EDM_MAP_ENTRY_NONS(trans_, ProductDoesNotSupportViews);
      EDM_MAP_ENTRY_NONS(trans_, ProductDoesNotSupportPtr);
      EDM_MAP_ENTRY_NONS(trans_, NotFound);
      EDM_MAP_ENTRY_NONS(trans_, FormatIncompatibility);
      EDM_MAP_ENTRY_NONS(trans_, FallbackFileOpenError);
    }
  }

  void getCodeTable(Exception::CodeMap*& setme) {
    static errors::FilledMap fm;
    setme = &fm.trans_;
  }

  /// -------------- implementation details ------------------

  std::string Exception::codeToString(errors::ErrorCodes c) {
    extern void getCodeTable(CodeMap*&);
    CodeMap* trans;
    getCodeTable(trans);
    CodeMap::const_iterator i(trans->find(c));
    return i!=trans->end() ? i->second : std::string("UnknownCode");
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

  Exception::~Exception() throw() {
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
