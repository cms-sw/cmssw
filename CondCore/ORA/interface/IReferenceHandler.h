#ifndef INCLUDE_ORA_IREFERENCEHANDLER_H
#define INCLUDE_ORA_IREFERENCEHANDLER_H

namespace ora {

  class Reference;
  
  /// 
  class IReferenceHandler
  {
  public:
    /// destructor
    virtual ~IReferenceHandler() {}

    virtual bool onSave( ora::Reference& ref ) = 0;

    virtual bool onLoad( ora::Reference& ref ) = 0;
  };

}

#endif
