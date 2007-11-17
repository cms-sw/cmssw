#ifndef TFILE_ADAPTOR_TFILE_ADAPTOR_UI_H
# define TFILE_ADAPTOR_TFILE_ADAPTOR_UI_H
# include "IOPool/TFileAdaptor/interface/TFileAdaptor.h"
# ifndef __MAKECINT__
#  include <boost/shared_ptr.hpp>
# else
   namespace boost { template <typename T> class shared_ptr; }
# endif
# include <iostream>

// Wrapper to bind TFileAdaptor to ROOT, Python etc.  Loading
// this library and instantiating a TFileAdaptorUI will make
// ROOT use StorageFactory for I/O, instead of the ROOT native
// plug-ins.
class TFileAdaptorUI
{
  boost::shared_ptr<TFileAdaptor> me;
public:
  TFileAdaptorUI (void);
  void stats (void) const;
  void statsXML (void) const;
};

#endif // TFILE_ADAPTOR_TFILE_ADAPTOR_UI_H
