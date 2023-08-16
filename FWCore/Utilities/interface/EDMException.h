#ifndef FWCore_Utilities_EDMException_h
#define FWCore_Utilities_EDMException_h

/**
 This is the basic exception that is thrown by the framework code.
 It exists primarily to distinguish framework thrown exception types
 from developer thrown exception types. As such there is very little
 interface other than constructors specific to this derived type.
**/

#include "FWCore/Utilities/interface/Exception.h"
#include <map>
#include <string>

namespace edm {
  namespace errors {

    // If you add a new entry to the set of values, make sure to
    // update the translation map in EDMException.cc, and the configuration
    // fragment FWCore/Framework/python/test/cmsExceptionsFatalOption_cff.py.

    enum ErrorCodes {
      CommandLineProcessing = 7000,
      ConfigFileNotFound = 7001,
      ConfigFileReadError = 7002,

      OtherCMS = 8001,
      StdException = 8002,
      Unknown = 8003,
      BadAlloc = 8004,
      BadExceptionType = 8005,

      ProductNotFound = 8006,
      DictionaryNotFound = 8007,
      InsertFailure = 8008,
      Configuration = 8009,
      LogicError = 8010,
      UnimplementedFeature = 8011,
      InvalidReference = 8012,
      NullPointerError = 8013,
      NoProductSpecified = 8014,
      EventTimeout = 8015,
      EventCorruption = 8016,

      ScheduleExecutionFailure = 8017,
      EventProcessorFailure = 8018,

      FileInPathError = 8019,
      FileOpenError = 8020,
      FileReadError = 8021,
      FatalRootError = 8022,
      MismatchedInputFiles = 8023,

      ProductDoesNotSupportViews = 8024,
      ProductDoesNotSupportPtr = 8025,

      NotFound = 8026,
      FormatIncompatibility = 8027,
      FallbackFileOpenError = 8028,
      NoSecondaryFiles = 8029,

      ExceededResourceVSize = 8030,
      ExceededResourceRSS = 8031,
      ExceededResourceTime = 8032,

      FileWriteError = 8033,

      FileNameInconsistentWithGUID = 8034,

      UnavailableAccelerator = 8035,
      ExternalFailure = 8036,

      EventGenerationFailure = 8501,

      UnexpectedJobTermination = 8901,

      CaughtSignal = 9000
    };

  }  // namespace errors

  class dso_export Exception : public cms::Exception {
  public:
    typedef errors::ErrorCodes Code;

    explicit Exception(Code category);

    Exception(Code category, std::string const& message);
    Exception(Code category, char const* message);

    Exception(Code category, std::string const& message, cms::Exception const& another);
    Exception(Code category, char const* message, cms::Exception const& another);

    Exception(Exception const& other);

    ~Exception() noexcept override;

    void swap(Exception& other) { std::swap(category_, other.category_); }

    Exception& operator=(Exception const& other);

    Code categoryCode() const { return category_; }

    static const std::string& codeToString(Code);

    static void throwThis(Code category,
                          char const* message0 = "",
                          char const* message1 = "",
                          char const* message2 = "",
                          char const* message3 = "",
                          char const* message4 = "");
    static void throwThis(Code category, char const* message0, int intVal, char const* message2 = "");

    Exception* clone() const override;

  private:
    void rethrow() override;
    int returnCode_() const override;

    Code category_;
  };
}  // namespace edm

#endif
