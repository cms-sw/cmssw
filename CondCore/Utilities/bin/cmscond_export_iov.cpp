#include "CondCore/Utilities/interface/ExportIOVUtilities.h"

int main( int argc, char** argv ){
  cond::ExportIOVUtilities utilities("cmscond_export_iov");
  return utilities.run(argc,argv);
}

