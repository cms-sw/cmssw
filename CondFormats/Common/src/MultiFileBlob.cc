#include "CondFormats/Common/interface/MultiFileBlob.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <zlib.h>

MultiFileBlob::MultiFileBlob() : compressed(false), isize(0), expanded(false) {}

MultiFileBlob::~MultiFileBlob() {}

void MultiFileBlob::finalized(bool compress) {
  if (!compress)
    return;
  if (0 == isize)
    return;
  compressed = true;
  expanded = false;
  std::vector<unsigned char> out(isize);
  uLongf destLen = compressBound(isize);
  int zerr = compress2(&out.front(), &destLen, &blob.front(), isize, 9);
  if (zerr != 0)
    edm::LogError("MultiFileBlob") << "Compression error " << zerr;
  out.resize(destLen);
  blob.swap(out);
}

void MultiFileBlob::read(const std::string& name, std::istream& is) {
  Positions::const_iterator pos = positions.find(name);
  if (pos != positions.end()) {
    edm::LogError("MultiFileBlob:") << name << "already in this object";
    return;
  }
  positions[name] = isize;
  char c;
  while (is.get(c))
    blob.push_back((unsigned char)c);
  isize = blob.size();
}

void MultiFileBlob::write(const std::string& name, std::ostream& os) const {
  Range r = rawBlob(name);
  os.write((const char*)(r.first), r.second - r.first);
}

MultiFileBlob::Range MultiFileBlob::rawBlob(const std::string& name) const {
  const_cast<MultiFileBlob*>(this)->expand();
  Positions::const_iterator pos = positions.find(name);
  if (pos == positions.end()) {
    edm::LogError("MultiFileBlob:") << name << "not in this object";
    return Range(nullptr, nullptr);
  }
  unsigned long long b = (*pos).second;
  unsigned long long e = isize;
  pos++;
  if (pos != positions.end())
    e = (*pos).second;

  return Range(&blob[b], &blob[e]);
}

unsigned long long MultiFileBlob::size(const std::string& name) const {
  Range r = rawBlob(name);
  return r.second - r.first;
}

void MultiFileBlob::expand() {
  if (expanded)
    return;
  if (!compressed) {
    expanded = true;
    return;
  }
  std::vector<unsigned char> out(isize);
  uLongf destLen = out.size();
  int zerr = uncompress(&out.front(), &destLen, &blob.front(), blob.size());
  if (zerr != 0 || out.size() != destLen)
    edm::LogError("FileBlob") << "uncompressing error " << zerr << " original size was " << isize << " new size is "
                              << destLen;
  blob.swap(out);
  expanded = true;
}
