#include <boost/shared_ptr.hpp>
class TFileAdaptorUI;

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

#endif

