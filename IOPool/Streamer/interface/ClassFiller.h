#ifndef Streamer_ClassFiller_h
#define Streamer_ClassFiller_h
#include "TClass.h"
#include <typeinfo>
namespace edm {
  void loadExtraClasses();
  TClass* getTClass(const std::type_info& ti);
}
#endif
