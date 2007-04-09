#ifndef FWCORE_SERVICES_VERTEX_TRACKER_H
#define FWCORE_SERVICES_VERTEX_TRACKER_H

#include <iosfwd>
#include <string>

#include "Sym.h"
#include "ProfParseTypedefs.h"

// ------------------- Vertex Tracker class ----------------
struct VertexTracker
{
  VertexTracker():
    name_(),
    addr_(),
    id_(),
    total_as_leaf_(),
    total_seen_(),
    in_path_(),
    size_(),
    percent_leaf_(0.0),
    percent_path_(0.0)
  { } 

  explicit VertexTracker(unsigned int id):
    name_(),
    addr_(),
    id_(id),
    total_as_leaf_(),
    total_seen_(),
    in_path_(),
    size_(),
    percent_leaf_(0.0),
    percent_path_(0.0)
  { }

  VertexTracker(unsigned int addr, const std::string& name):
    name_(name),
    addr_(addr),
    id_(),
    total_as_leaf_(),
    total_seen_(),
    in_path_(),
    size_(),
    percent_leaf_(0.0),
    percent_path_(0.0)
  { }

  explicit VertexTracker(const Sym& e) :
    name_(e.name_),
    addr_(e.addr_),
    id_(),
    total_as_leaf_(),
    total_seen_(),
    in_path_(),
    //size_(e.size_),
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
  unsigned int addr_;
  mutable unsigned int id_;
  mutable unsigned int total_as_leaf_;
  mutable unsigned int total_seen_;
  mutable unsigned int in_path_;
  mutable EdgeMap edges_;
  mutable int size_;
  mutable float percent_leaf_;
  mutable float percent_path_;

  static unsigned int next_id_;
};

std::ostream&
operator<< (std::ostream& os, VertexTracker const& vt);

#endif
