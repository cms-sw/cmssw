#ifndef GlobalTrackingGeometryBuilder_GlobalTrackingGeometryBuilder_h
#define GlobalTrackingGeometryBuilder_GlobalTrackingGeometryBuilder_h

/** \class GlobalTrackingGeometryBuilder
 *
 *  Build the GlobalTrackingGeometry.
 *
 *  $Date: 2011/08/16 14:54:34 $
 *  $Revision: 1.1 $
 *  \author Matteo Sani
 */

#include <string>

class GlobalTrackingGeometry;
class TrackerGeometry;
class DTGeometry;
class CSCGeometry;
class RPCGeometry;

class GlobalTrackingGeometryBuilder {
public:
  /// Constructor
  GlobalTrackingGeometryBuilder();

  /// Destructor
  virtual ~GlobalTrackingGeometryBuilder();

  /// Build the geometry
  GlobalTrackingGeometry* build(const TrackerGeometry* tk, const DTGeometry* dt, 
                                const CSCGeometry* csc, const RPCGeometry* rpc);

protected:

private:

  const std::string myName;

};
#endif
