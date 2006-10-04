#include "IOPool/TFileAdaptor/interface/TStorageFactoryFile.h"
#include "IOPool/TFileAdaptor/interface/TStorageFactorySystem.h"
#include <boost/shared_ptr.hpp>
class TFileAdaptor;

class TFileAdaptorUI {
public:
  
  TFileAdaptorUI();
  ~TFileAdaptorUI();
  void stats() const;
  
private:
  boost::shared_ptr<TFileAdaptor> me;
};


#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class TFileAdaptorUI;

#pragma link C++ class TStorageFactoryFile;
#pragma link C++ class TStorageFactorySystem;

#endif // __CINT__

