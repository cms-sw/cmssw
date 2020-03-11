#ifndef DQMSERVICES_CORE_DQM_NET_H
#define DQMSERVICES_CORE_DQM_NET_H

#include "classlib/iobase/Socket.h"
#include "classlib/iobase/IOSelector.h"
#include "classlib/iobase/Pipe.h"
#include "classlib/utils/Signal.h"
#include "classlib/utils/Error.h"
#include "classlib/utils/Time.h"
#include <pthread.h>
#include <cstdint>
#include <csignal>
#include <iostream>
#include <vector>
#include <string>
#include <list>
#include <map>
#include <set>
#include <ext/hash_set>

// for definition of QValue
#include "DataFormats/Histograms/interface/MonitorElementCollection.h"

//class DQMStore;

class DQMNet {
public:
  static const uint32_t DQM_PROP_TYPE_MASK = 0x000000ff;
  static const uint32_t DQM_PROP_TYPE_SCALAR = 0x0000000f;
  static const uint32_t DQM_PROP_TYPE_INVALID = 0x00000000;
  static const uint32_t DQM_PROP_TYPE_INT = 0x00000001;
  static const uint32_t DQM_PROP_TYPE_REAL = 0x00000002;
  static const uint32_t DQM_PROP_TYPE_STRING = 0x00000003;
  static const uint32_t DQM_PROP_TYPE_TH1F = 0x00000010;
  static const uint32_t DQM_PROP_TYPE_TH1S = 0x00000011;
  static const uint32_t DQM_PROP_TYPE_TH1D = 0x00000012;
  static const uint32_t DQM_PROP_TYPE_TH2F = 0x00000020;
  static const uint32_t DQM_PROP_TYPE_TH2S = 0x00000021;
  static const uint32_t DQM_PROP_TYPE_TH2D = 0x00000022;
  static const uint32_t DQM_PROP_TYPE_TH3F = 0x00000030;
  static const uint32_t DQM_PROP_TYPE_TH3S = 0x00000031;
  static const uint32_t DQM_PROP_TYPE_TH3D = 0x00000032;
  static const uint32_t DQM_PROP_TYPE_TPROF = 0x00000040;
  static const uint32_t DQM_PROP_TYPE_TPROF2D = 0x00000041;
  static const uint32_t DQM_PROP_TYPE_DATABLOB = 0x00000050;

  static const uint32_t DQM_PROP_REPORT_MASK = 0x00000f00;
  static const uint32_t DQM_PROP_REPORT_CLEAR = 0x00000000;
  static const uint32_t DQM_PROP_REPORT_ERROR = 0x00000100;
  static const uint32_t DQM_PROP_REPORT_WARN = 0x00000200;
  static const uint32_t DQM_PROP_REPORT_OTHER = 0x00000400;
  static const uint32_t DQM_PROP_REPORT_ALARM = (DQM_PROP_REPORT_ERROR | DQM_PROP_REPORT_WARN | DQM_PROP_REPORT_OTHER);

  static const uint32_t DQM_PROP_HAS_REFERENCE = 0x00001000;
  static const uint32_t DQM_PROP_TAGGED = 0x00002000;
  static const uint32_t DQM_PROP_ACCUMULATE = 0x00004000;
  static const uint32_t DQM_PROP_RESET = 0x00008000;

  static const uint32_t DQM_PROP_NEW = 0x00010000;
  static const uint32_t DQM_PROP_RECEIVED = 0x00020000;
  static const uint32_t DQM_PROP_LUMI = 0x00040000;
  static const uint32_t DQM_PROP_DEAD = 0x00080000;
  static const uint32_t DQM_PROP_STALE = 0x00100000;
  static const uint32_t DQM_PROP_EFFICIENCY_PLOT = 0x00200000;
  static const uint32_t DQM_PROP_MARKTODELETE = 0x01000000;

  static const uint32_t DQM_MSG_HELLO = 0;
  static const uint32_t DQM_MSG_UPDATE_ME = 1;
  static const uint32_t DQM_MSG_LIST_OBJECTS = 2;
  static const uint32_t DQM_MSG_GET_OBJECT = 3;

  static const uint32_t DQM_REPLY_LIST_BEGIN = 101;
  static const uint32_t DQM_REPLY_LIST_END = 102;
  static const uint32_t DQM_REPLY_NONE = 103;
  static const uint32_t DQM_REPLY_OBJECT = 104;

  static const uint32_t MAX_PEER_WAITREQS = 128;

  struct Peer;
  struct WaitObject;

  using QValue = MonitorElementData::QReport::QValue;
  using DataBlob = std::vector<unsigned char>;
  using QReports = std::vector<QValue>;
  using TagList = std::vector<uint32_t>;  // DEPRECATED
  using WaitList = std::list<WaitObject>;

  struct CoreObject {
    uint32_t flags;
    uint32_t tag;
    uint64_t version;
    uint32_t run;
    uint32_t lumi;
    uint32_t streamId;
    uint32_t moduleId;
    std::string dirname;
    std::string objname;
    QReports qreports;
  };

  struct Object : CoreObject {
    uint64_t hash;
    uint64_t lastreq;
    DataBlob rawdata;
    std::string scalar;
    std::string qdata;
  };

  struct Bucket {
    Bucket *next;
    DataBlob data;
  };

  struct WaitObject {
    lat::Time time;
    std::string name;
    std::string info;
    Peer *peer;
  };

  struct AutoPeer;
  struct Peer {
    std::string peeraddr;
    lat::Socket *socket;
    DataBlob incoming;
    Bucket *sendq;
    size_t sendpos;

    unsigned mask;
    bool source;
    bool update;
    bool updated;
    size_t updates;
    size_t waiting;
    AutoPeer *automatic;
  };

  struct AutoPeer {
    Peer *peer;
    lat::Time next;
    std::string host;
    int port;
    bool update;
  };

  DQMNet(const std::string &appname = "");
  virtual ~DQMNet();

  void debug(bool doit);
  void delay(int delay);
  void startLocalServer(int port);
  void startLocalServer(const char *path);
  void staleObjectWaitLimit(lat::TimeSpan time);
  void updateToCollector(const std::string &host, int port);
  void listenToCollector(const std::string &host, int port);
  void shutdown();
  void lock();
  void unlock();

  void start();
  void run();

  void sendLocalChanges();

  static bool setOrder(const CoreObject &a, const CoreObject &b) {
    if (a.run == b.run) {
      if (a.lumi == b.lumi) {
        if (a.streamId == b.streamId) {
          if (a.moduleId == b.moduleId) {
            if (a.dirname == b.dirname) {
              return a.objname < b.objname;
            }
            return a.dirname < b.dirname;
          }
          return a.moduleId < b.moduleId;
        }
        return a.streamId < b.streamId;
      }
      return a.lumi < b.lumi;
    }
    return a.run < b.run;
  }

  struct HashOp {
    uint32_t operator()(const Object &a) const { return a.hash; }
  };

  struct HashEqual {
    bool operator()(const Object &a, const Object &b) const {
      return a.hash == b.hash && a.dirname == b.dirname && a.objname == b.objname;
    }
  };

  static size_t dqmhash(const void *key, size_t keylen) {
    // Reduced version of Bob Jenkins' hash function at:
    //   http://www.burtleburtle.net/bob/c/lookup3.c
#define dqmhashrot(x, k) (((x) << (k)) | ((x) >> (32 - (k))))
#define dqmhashmix(a, b, c) \
  {                         \
    a -= c;                 \
    a ^= dqmhashrot(c, 4);  \
    c += b;                 \
    b -= a;                 \
    b ^= dqmhashrot(a, 6);  \
    a += c;                 \
    c -= b;                 \
    c ^= dqmhashrot(b, 8);  \
    b += a;                 \
    a -= c;                 \
    a ^= dqmhashrot(c, 16); \
    c += b;                 \
    b -= a;                 \
    b ^= dqmhashrot(a, 19); \
    a += c;                 \
    c -= b;                 \
    c ^= dqmhashrot(b, 4);  \
    b += a;                 \
  }
#define dqmhashfinal(a, b, c) \
  {                           \
    c ^= b;                   \
    c -= dqmhashrot(b, 14);   \
    a ^= c;                   \
    a -= dqmhashrot(c, 11);   \
    b ^= a;                   \
    b -= dqmhashrot(a, 25);   \
    c ^= b;                   \
    c -= dqmhashrot(b, 16);   \
    a ^= c;                   \
    a -= dqmhashrot(c, 4);    \
    b ^= a;                   \
    b -= dqmhashrot(a, 14);   \
    c ^= b;                   \
    c -= dqmhashrot(b, 24);   \
  }

    uint32_t a, b, c;
    a = b = c = 0xdeadbeef + (uint32_t)keylen;
    const auto *k = (const unsigned char *)key;

    // all but the last block: affect some bits of (a, b, c)
    while (keylen > 12) {
      a += k[0];
      a += ((uint32_t)k[1]) << 8;
      a += ((uint32_t)k[2]) << 16;
      a += ((uint32_t)k[3]) << 24;
      b += k[4];
      b += ((uint32_t)k[5]) << 8;
      b += ((uint32_t)k[6]) << 16;
      b += ((uint32_t)k[7]) << 24;
      c += k[8];
      c += ((uint32_t)k[9]) << 8;
      c += ((uint32_t)k[10]) << 16;
      c += ((uint32_t)k[11]) << 24;
      dqmhashmix(a, b, c);
      keylen -= 12;
      k += 12;
    }

    // last block: affect all 32 bits of (c); all case statements fall through
    switch (keylen) {
      case 12:
        c += ((uint32_t)k[11]) << 24;
        [[fallthrough]];
      case 11:
        c += ((uint32_t)k[10]) << 16;
        [[fallthrough]];
      case 10:
        c += ((uint32_t)k[9]) << 8;
        [[fallthrough]];
      case 9:
        c += k[8];
        [[fallthrough]];
      case 8:
        b += ((uint32_t)k[7]) << 24;
        [[fallthrough]];
      case 7:
        b += ((uint32_t)k[6]) << 16;
        [[fallthrough]];
      case 6:
        b += ((uint32_t)k[5]) << 8;
        [[fallthrough]];
      case 5:
        b += k[4];
        [[fallthrough]];
      case 4:
        a += ((uint32_t)k[3]) << 24;
        [[fallthrough]];
      case 3:
        a += ((uint32_t)k[2]) << 16;
        [[fallthrough]];
      case 2:
        a += ((uint32_t)k[1]) << 8;
        [[fallthrough]];
      case 1:
        a += k[0];
        break;
      case 0:
        return c;
    }

    dqmhashfinal(a, b, c);
    return c;
#undef dqmhashrot
#undef dqmhashmix
#undef dqmhashfinal
  }

  static void packQualityData(std::string &into, const QReports &qr);
  static void unpackQualityData(QReports &qr, uint32_t &flags, const char *from);

protected:
  std::ostream &logme();
  static void copydata(Bucket *b, const void *data, size_t len);
  virtual void sendObjectToPeer(Bucket *msg, Object &o, bool data);

  virtual bool shouldStop();
  void waitForData(Peer *p, const std::string &name, const std::string &info, Peer *owner);
  virtual void releaseFromWait(Bucket *msg, WaitObject &w, Object *o);
  virtual bool onMessage(Bucket *msg, Peer *p, unsigned char *data, size_t len);

  // bool			reconstructObject(Object &o);
  // bool			reinstateObject(DQMStore *store, Object &o);
  virtual Object *findObject(Peer *p, const std::string &name, Peer **owner = nullptr) = 0;
  virtual Object *makeObject(Peer *p, const std::string &name) = 0;
  virtual void markObjectsDead(Peer *p) = 0;
  virtual void purgeDeadObjects(Peer *p) = 0;

  virtual Peer *getPeer(lat::Socket *s) = 0;
  virtual Peer *createPeer(lat::Socket *s) = 0;
  virtual void removePeer(Peer *p, lat::Socket *s) = 0;
  virtual void sendObjectListToPeer(Bucket *msg, bool all, bool clear) = 0;
  virtual void sendObjectListToPeers(bool all) = 0;

  void updateMask(Peer *p);
  virtual void updatePeerMasks() = 0;
  static void discard(Bucket *&b);

  bool debug_;
  pthread_mutex_t lock_;

private:
  void losePeer(const char *reason, Peer *peer, lat::IOSelectEvent *event, lat::Error *err = nullptr);
  void requestObjectData(Peer *p, const char *name, size_t len);
  void releaseFromWait(WaitList::iterator i, Object *o);
  void releaseWaiters(const std::string &name, Object *o);

  bool onPeerData(lat::IOSelectEvent *ev, Peer *p);
  bool onPeerConnect(lat::IOSelectEvent *ev);
  bool onLocalNotify(lat::IOSelectEvent *ev);

  std::string appname_;
  int pid_;

  lat::IOSelector sel_;
  lat::Socket *server_;
  lat::Pipe wakeup_;
  lat::Time version_;

  AutoPeer upstream_;
  AutoPeer downstream_;
  WaitList waiting_;

  pthread_t communicate_;
  sig_atomic_t shutdown_;

  int delay_;
  lat::TimeSpan waitStale_;
  lat::TimeSpan waitMax_;
  bool flush_;

public:
  // copying is not available
  DQMNet(const DQMNet &) = delete;
  DQMNet &operator=(const DQMNet &) = delete;
};

template <class ObjType>
class DQMImplNet : public DQMNet {
public:
  struct ImplPeer;

  using DirMap = std::set<std::string>;
  typedef __gnu_cxx::hash_set<ObjType, HashOp, HashEqual> ObjectMap;
  typedef std::map<lat::Socket *, ImplPeer> PeerMap;
  struct ImplPeer : Peer {
    ImplPeer() = default;
    ObjectMap objs;
    DirMap dirs;
  };

  DQMImplNet(const std::string &appname = "") : DQMNet(appname) {}

  ~DQMImplNet() override = default;

protected:
  Object *findObject(Peer *p, const std::string &name, Peer **owner = nullptr) override {
    size_t slash = name.rfind('/');
    size_t dirpos = (slash == std::string::npos ? 0 : slash);
    size_t namepos = (slash == std::string::npos ? 0 : slash + 1);
    std::string path(name, 0, dirpos);
    ObjType proto;
    proto.hash = dqmhash(name.c_str(), name.size());
    proto.dirname = path;
    proto.objname.append(name, namepos, std::string::npos);

    typename ObjectMap::iterator pos;
    typename PeerMap::iterator i, e;
    if (owner)
      *owner = nullptr;
    if (p) {
      auto *ip = static_cast<ImplPeer *>(p);
      pos = ip->objs.find(proto);
      if (pos == ip->objs.end())
        return nullptr;
      else {
        if (owner)
          *owner = ip;
        return const_cast<ObjType *>(&*pos);
      }
    } else {
      for (i = peers_.begin(), e = peers_.end(); i != e; ++i) {
        pos = i->second.objs.find(proto);
        if (pos != i->second.objs.end()) {
          if (owner)
            *owner = &i->second;
          return const_cast<ObjType *>(&*pos);
        }
      }
      return nullptr;
    }
  }

  Object *makeObject(Peer *p, const std::string &name) override {
    auto *ip = static_cast<ImplPeer *>(p);
    size_t slash = name.rfind('/');
    size_t dirpos = (slash == std::string::npos ? 0 : slash);
    size_t namepos = (slash == std::string::npos ? 0 : slash + 1);
    ObjType o;
    o.flags = 0;
    o.tag = 0;
    o.version = 0;
    o.lastreq = 0;
    o.dirname = *ip->dirs.insert(name.substr(0, dirpos)).first;
    o.objname.append(name, namepos, std::string::npos);
    o.hash = dqmhash(name.c_str(), name.size());
    return const_cast<ObjType *>(&*ip->objs.insert(o).first);
  }

  // Mark all the objects dead.  This is intended to be used when
  // starting to process a complete list of objects, in order to
  // flag the objects that need to be killed at the end.  After
  // call to this method, revive all live objects by removing the
  // DQM_PROP_DEAD flag, then call purgeDeadObjects() at the end
  // to remove the dead ones.  This also turns off object request
  // for objects we've lost interest in.
  void markObjectsDead(Peer *p) override {
    uint64_t minreq = (lat::Time::current() - lat::TimeSpan(0, 0, 5 /* minutes */, 0, 0)).ns();
    auto *ip = static_cast<ImplPeer *>(p);
    typename ObjectMap::iterator i, e;
    for (i = ip->objs.begin(), e = ip->objs.end(); i != e; ++i) {
      if (i->lastreq && i->lastreq < minreq)
        const_cast<ObjType &>(*i).lastreq = 0;
      const_cast<ObjType &>(*i).flags |= DQM_PROP_DEAD;
    }
  }

  // Mark remaining zombie objects as dead.  See markObjectsDead().
  void purgeDeadObjects(Peer *p) override {
    auto *ip = static_cast<ImplPeer *>(p);
    typename ObjectMap::iterator i, e;
    for (i = ip->objs.begin(), e = ip->objs.end(); i != e;) {
      if (i->flags & DQM_PROP_DEAD)
        ip->objs.erase(i++);
      else
        ++i;
    }
  }

  Peer *getPeer(lat::Socket *s) override {
    auto pos = peers_.find(s);
    auto end = peers_.end();
    return pos == end ? nullptr : &pos->second;
  }

  Peer *createPeer(lat::Socket *s) override {
    ImplPeer *ip = &peers_[s];
    ip->socket = nullptr;
    ip->sendq = nullptr;
    ip->sendpos = 0;
    ip->mask = 0;
    ip->source = false;
    ip->update = false;
    ip->updated = false;
    ip->updates = 0;
    ip->waiting = 0;
    ip->automatic = nullptr;
    return ip;
  }

  void removePeer(Peer *p, lat::Socket *s) override {
    auto *ip = static_cast<ImplPeer *>(p);
    bool needflush = !ip->objs.empty();

    typename ObjectMap::iterator i, e;
    for (i = ip->objs.begin(), e = ip->objs.end(); i != e;)
      ip->objs.erase(i++);

    peers_.erase(s);

    // If we removed a peer with objects, our list of objects
    // has changed and we need to update downstream peers.
    if (needflush)
      sendLocalChanges();
  }

  /// Send all objects to a peer and optionally mark sent objects old.
  void sendObjectListToPeer(Bucket *msg, bool all, bool clear) override {
    typename PeerMap::iterator pi, pe;
    typename ObjectMap::iterator oi, oe;
    size_t size = 0;
    size_t numobjs = 0;
    for (pi = peers_.begin(), pe = peers_.end(); pi != pe; ++pi)
      for (oi = pi->second.objs.begin(), oe = pi->second.objs.end(); oi != oe; ++oi, ++numobjs)
        if (all || (oi->flags & DQM_PROP_NEW))
          size += 9 * sizeof(uint32_t) + oi->dirname.size() + oi->objname.size() + 1 + oi->scalar.size() +
                  oi->qdata.size() + (oi->lastreq > 0 ? oi->rawdata.size() : 0);

    msg->data.reserve(msg->data.size() + size + 8 * sizeof(uint32_t));

    uint32_t nupdates = 0;
    uint32_t words[4];
    words[0] = sizeof(words);
    words[1] = DQM_REPLY_LIST_BEGIN;
    words[2] = numobjs;
    words[3] = all;
    copydata(msg, &words[0], sizeof(words));

    for (pi = peers_.begin(), pe = peers_.end(); pi != pe; ++pi)
      for (oi = pi->second.objs.begin(), oe = pi->second.objs.end(); oi != oe; ++oi)
        if (all || (oi->flags & DQM_PROP_NEW)) {
          sendObjectToPeer(msg, const_cast<ObjType &>(*oi), oi->lastreq > 0);
          if (clear)
            const_cast<ObjType &>(*oi).flags &= ~DQM_PROP_NEW;
          ++nupdates;
        }

    words[1] = DQM_REPLY_LIST_END;
    words[2] = nupdates;
    copydata(msg, &words[0], sizeof(words));
  }

  void sendObjectListToPeers(bool all) override {
    typename PeerMap::iterator i, e;
    typename ObjectMap::iterator oi, oe;
    for (i = peers_.begin(), e = peers_.end(); i != e; ++i) {
      ImplPeer &p = i->second;
      if (!p.update)
        continue;

      if (debug_)
        logme() << "DEBUG: notifying " << p.peeraddr << std::endl;

      Bucket msg;
      msg.next = nullptr;
      sendObjectListToPeer(&msg, !p.updated || all, true);

      if (!msg.data.empty()) {
        Bucket **prev = &p.sendq;
        while (*prev)
          prev = &(*prev)->next;

        *prev = new Bucket;
        (*prev)->next = nullptr;
        (*prev)->data.swap(msg.data);
      }
      p.updated = true;
    }
  }

  void updatePeerMasks() override {
    typename PeerMap::iterator i, e;
    for (i = peers_.begin(), e = peers_.end(); i != e;)
      updateMask(&(i++)->second);
  }

protected:
  PeerMap peers_;
};

class DQMBasicNet : public DQMImplNet<DQMNet::Object> {
public:
  DQMBasicNet(const std::string &appname = "");

  void reserveLocalSpace(uint32_t size);
  void updateLocalObject(Object &o);
  bool removeLocalExcept(const std::set<std::string> &known);

private:
  ImplPeer *local_;
};

#endif  // DQMSERVICES_CORE_DQM_NET_H
