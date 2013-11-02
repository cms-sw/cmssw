#ifndef FWCore_Services_Vertex_Tracker_h
#define FWCore_Services_Vertex_Tracker_h

#include <atomic>
#include <iosfwd>
#include <string>

#include "Sym.h"
#include "ProfParseTypedefs.h"

// ------------------- Vertex Tracker class ----------------
struct VertexTracker
{
  typedef void* address_type;
  VertexTracker():
    name_(),
    library_(),
    addr_(),
    id_(),
    total_as_leaf_(),
    total_seen_(),
    in_path_(),
    //size_(),
    percent_leaf_(0.0),
    percent_path_(0.0)
  { } 

  explicit VertexTracker(unsigned int id):
    name_(),
    library_(),
    addr_(),
    id_(id),
    total_as_leaf_(),
    total_seen_(),
    in_path_(),
    //size_(),
    percent_leaf_(0.0),
    percent_path_(0.0)
  { }

  VertexTracker(address_type addr, const std::string& name):
    name_(name),
    library_(),
    addr_(addr),
    id_(),
    total_as_leaf_(),
    total_seen_(),
    in_path_(),
    //size_(),
    percent_leaf_(0.0),
    percent_path_(0.0)
  { }

  explicit VertexTracker(const Sym& sym) :
    name_(sym.name_),
    library_(sym.library_),
    addr_(sym.addr_),
    id_(),
    total_as_leaf_(),
    total_seen_(),
    in_path_(),
    //size_(sym.size_),
    percent_leaf_(0.0),
    percent_path_(0.0)
  { }

  bool 
  operator<(const VertexTracker& a) const 
  { return addr_<a.addr_; }

  bool 
  operator<(unsigned int id) const 
  { return id_<id; }

  void 
  incLeaf() const 
  { ++total_as_leaf_; }

  void 
  incTotal() const 
  { ++total_seen_; }

  void 
  incPath(int by) const 
  { in_path_+=by; }

  void 
  setID() const 
  { id_=next_id_++; }

  std::string name_;
  std::string library_;
  address_type addr_;
  mutable unsigned int id_;
  mutable unsigned int total_as_leaf_;
  mutable unsigned int total_seen_;
  mutable unsigned int in_path_;
  mutable EdgeMap      edges_;
  //mutable int          size_;
  mutable float        percent_leaf_;
  mutable float        percent_path_;

  static std::atomic<unsigned int> next_id_;
};

std::ostream&
operator<< (std::ostream& os, VertexTracker const& vt);

#endif
