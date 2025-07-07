#ifndef CondFormats_MLObjects_Metadata_h
#define CondFormats_MLObjects_Metadata_h

// -*- C++ -*-
//
// Package:     CondFormats/MLObjects
// Class  :     Metadata
//
/**
  \class Metadata Metadata.h "CondFormats/MLObjects/interface/Metadata.h"

  Description: Persistent condition object containing core ML model identifiers.
               Stores the unique model name, versioning, and integrity hash 
               to ensure synchronization with the corresponding external metadata files.

  Author:      H. Kwon
  Created:     Feb 2026
*/

#include <string>
#include "CondFormats/Serialization/interface/Serializable.h"

class Metadata {
public:
  Metadata() {}
  Metadata(std::string model_name, int version, std::string hash)
      : model_name_(model_name), version_(version), hash_(hash) {}

  std::string model_name() const { return model_name_; }
  int version() const { return version_; }
  std::string hash() const { return hash_; }

  void set_model_name(const std::string& name) { model_name_ = name; }
  void set_version(int version) { version_ = version; }
  void set_hash(const std::string& hash) { hash_ = hash; }

private:
  std::string model_name_;
  int version_;
  std::string hash_;

  COND_SERIALIZABLE;
};

#endif
