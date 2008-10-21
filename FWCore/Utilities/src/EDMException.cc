
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  namespace errors {
    struct FilledMap {
      FilledMap();
      edm::Exception::CodeMap trans_;
    };
    FilledMap::FilledMap() : trans_() {
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
    }
  }

  void getCodeTable(edm::Exception::CodeMap*& setme) {
    static errors::FilledMap fm;
    setme = &fm.trans_;
  }
}
