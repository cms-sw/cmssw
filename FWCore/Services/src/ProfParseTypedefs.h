#ifndef FWCore_Services_ProfParseTypedefs_h
#define FWCore_Services_ProfParseTypedefs_h

#include <map>
#include <set>
#include <vector>

struct VertexTracker;
struct PathTracker;

typedef std::vector<void*> VoidVec;
typedef std::vector<unsigned int> ULVec;
typedef std::map<unsigned int,int> EdgeMap;
typedef std::set<VertexTracker> VertexSet;
typedef std::set<PathTracker> PathSet;
typedef std::vector<VertexSet::const_iterator> Viter;
typedef std::vector<PathSet::const_iterator> Piter;

#endif
