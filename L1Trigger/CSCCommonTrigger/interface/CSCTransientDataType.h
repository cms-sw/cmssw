#ifndef CSCCommonTrigger_CSCTransientDataType_h
#define CSCCommonTrigger_CSCTransientDataType_h

/**
 * \class CSCTransientDataType
 * \author L. Gray
 *
 * Interface class for transient trigger objects.
 */

class CSCTransientDataType
{
 public:

  /// Returns the endcap this object is in.
  virtual unsigned endcap() const = 0;

  /// Returns the station this object is in.
  virtual unsigned station() const = 0;
  
  /// Returns the sector this object is in.
  virtual unsigned sector() const = 0;

  /// Returns the trigger subsector this object is in.
  virtual unsigned subsector() const = 0;

  /// Returns the Trigger CscId this object is associated with.
  virtual unsigned cscid() const = 0;
  
  /// Returns the 25 ns bunch crossing this object is associated with.
  virtual int BX() const = 0;

};

#endif
