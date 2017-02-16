#ifndef Alignment_CommonAlignment_AlignableObjectId_h
#define Alignment_CommonAlignment_AlignableObjectId_h

#include <string>
#include "Alignment/CommonAlignment/interface/StructureType.h"

class TrackerGeometry;
class DTGeometry;
class CSCGeometry;
class AlignableTracker;
class AlignableMuon;


/// Allows conversion between type and name, and vice-versa
class AlignableObjectId
{
public:
  struct entry;
  enum class Geometry { RunI, PhaseI, PhaseII, General, Unspecified };

  AlignableObjectId(Geometry);
  AlignableObjectId(const TrackerGeometry*, const DTGeometry*, const CSCGeometry*);
  AlignableObjectId(const AlignableObjectId&) = default;
  AlignableObjectId& operator=(const AlignableObjectId&) = default;
  AlignableObjectId(AlignableObjectId&&) = default;
  AlignableObjectId& operator=(AlignableObjectId&&) = default;
  virtual ~AlignableObjectId() = default;

  /// retrieve the geometry information
  Geometry geometry() const { return geometry_; }

  /// Convert name to type
  align::StructureType nameToType(const std::string& name) const;

  /// Convert type to name
  std::string typeToName( align::StructureType type ) const;
  const char *idToString(align::StructureType type) const;
  align::StructureType stringToId(const char*) const;
  align::StructureType stringToId(const std::string& s) const {
    return stringToId(s.c_str()); }

  static Geometry commonGeometry(Geometry, Geometry);
  static AlignableObjectId commonObjectIdProvider(const AlignableObjectId&,
                                                  const AlignableObjectId&);
  static AlignableObjectId commonObjectIdProvider(const AlignableTracker*,
                                                  const AlignableMuon*);

private:
  static Geometry trackerGeometry(const TrackerGeometry*);
  static Geometry muonGeometry(const DTGeometry*, const CSCGeometry*);

  const entry* entries_{nullptr};
  Geometry geometry_{Geometry::Unspecified};
};


#endif
