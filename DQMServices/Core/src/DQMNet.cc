#include "DQMServices/Core/interface/DQMNet.h"
#include "classlib/iobase/InetServerSocket.h"
#include "classlib/iobase/LocalServerSocket.h"
#include "classlib/iobase/Filename.h"
#include "classlib/sysapi/InetSocket.h"  // for completing InetAddress
#include "classlib/utils/TimeInfo.h"
#include "classlib/utils/StringList.h"
#include "classlib/utils/StringFormat.h"
#include "classlib/utils/StringOps.h"
#include "classlib/utils/SystemError.h"
#include "classlib/utils/Regexp.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <cassert>
#include <cfloat>
#include <cinttypes>

#include "FWCore/Utilities/interface/EDMException.h"

#if __APPLE__
#define MESSAGE_SIZE_LIMIT (1 * 1024 * 1024)
#define SOCKET_BUF_SIZE (1 * 1024 * 1024)
#else
#define MESSAGE_SIZE_LIMIT (8 * 1024 * 1024)
#define SOCKET_BUF_SIZE (8 * 1024 * 1024)
#endif
#define SOCKET_READ_SIZE (SOCKET_BUF_SIZE / 8)
#define SOCKET_READ_GROWTH (SOCKET_BUF_SIZE)

using namespace lat;

static const Regexp s_rxmeval("<(.*)>(i|f|s|qr)=(.*)</\\1>");

// TODO: Can't include the header file since that leads to ambiguities.
namespace dqm {
  namespace qstatus {
    static const int STATUS_OK = 100;  //< Test was succesful.
    static const int WARNING = 200;    //< Test had some problems.
    static const int ERROR = 300;      //< Test has failed.
  }                                    // namespace qstatus
}  // namespace dqm

//////////////////////////////////////////////////////////////////////
// Generate log prefix.
std::ostream &DQMNet::logme() {
  Time now = Time::current();
  return std::cout << now.format(true, "%Y-%m-%d %H:%M:%S.") << now.nanoformat(3, 3) << " " << appname_ << "[" << pid_
                   << "]: ";
}

// Append data into a bucket.
void DQMNet::copydata(Bucket *b, const void *data, size_t len) {
  b->data.insert(b->data.end(), (const unsigned char *)data, (const unsigned char *)data + len);
}

// Discard a bucket chain.
void DQMNet::discard(Bucket *&b) {
  while (b) {
    Bucket *next = b->next;
    delete b;
    b = next;
  }
}

//////////////////////////////////////////////////////////////////////
/** Handle errors with a peer socket.  Zaps the socket send queue,
    the socket itself, detaches the socket from the selector, and
    purges any pending wait requests linked to the socket.  */
void DQMNet::losePeer(const char *reason, Peer *peer, IOSelectEvent *ev, Error *err) {
  if (reason)
    logme() << reason << peer->peeraddr << (err ? "; error was: " + err->explain() : std::string("")) << std::endl;

  Socket *s = peer->socket;

  for (auto i = waiting_.begin(), e = waiting_.end(); i != e;)
    if (i->peer == peer)
      waiting_.erase(i++);
    else
      ++i;

  if (ev)
    ev->source = nullptr;

  discard(peer->sendq);
  if (peer->automatic)
    peer->automatic->peer = nullptr;

  sel_.detach(s);
  s->close();
  removePeer(peer, s);
  delete s;
}

/// Queue an object request to the data server.
void DQMNet::requestObjectData(Peer *p, const char *name, size_t len) {
  // Issue request to peer.
  Bucket **msg = &p->sendq;
  while (*msg)
    msg = &(*msg)->next;
  *msg = new Bucket;
  (*msg)->next = nullptr;

  uint32_t words[3];
  words[0] = sizeof(words) + len;
  words[1] = DQM_MSG_GET_OBJECT;
  words[2] = len;
  copydata(*msg, words, sizeof(words));
  copydata(*msg, name, len);
}

/// Queue a request for an object and put a peer into the mode of
/// waiting for object data to appear.
void DQMNet::waitForData(Peer *p, const std::string &name, const std::string &info, Peer *owner) {
  // FIXME: Should we automatically record which exact peer the waiter
  // is expecting to deliver data so we know to release the waiter if
  // the other peer vanishes?  The current implementation stands a
  // chance for the waiter to wait indefinitely -- although we do
  // force terminate the wait after a while.
  requestObjectData(owner, !name.empty() ? &name[0] : nullptr, name.size());
  WaitObject wo = {Time::current(), name, info, p};
  waiting_.push_back(wo);
  p->waiting++;
}

// Once an object has been updated, this is invoked for all waiting
// peers.  Send the object back to the peer in suitable form.
void DQMNet::releaseFromWait(WaitList::iterator i, Object *o) {
  Bucket **msg = &i->peer->sendq;
  while (*msg)
    msg = &(*msg)->next;
  *msg = new Bucket;
  (*msg)->next = nullptr;

  releaseFromWait(*msg, *i, o);

  assert(i->peer->waiting > 0);
  i->peer->waiting--;
  waiting_.erase(i);
}

// Release everyone waiting for the object @a o.
void DQMNet::releaseWaiters(const std::string &name, Object *o) {
  for (auto i = waiting_.begin(), e = waiting_.end(); i != e;)
    if (i->name == name)
      releaseFromWait(i++, o);
    else
      ++i;
}

//////////////////////////////////////////////////////////////////////
/// Pack quality results in @a qr into a string @a into for
/// peristent storage, such as network transfer or archival.
void DQMNet::packQualityData(std::string &into, const QReports &qr) {
  char buf[64];
  std::ostringstream qrs;
  QReports::const_iterator qi, qe;
  for (qi = qr.begin(), qe = qr.end(); qi != qe; ++qi) {
    int pos = 0;
    sprintf(buf, "%d%c%n%.*g", qi->code, 0, &pos, DBL_DIG + 2, qi->qtresult);
    qrs << buf << '\0' << buf + pos << '\0' << qi->qtname << '\0' << qi->algorithm << '\0' << qi->message << '\0'
        << '\0';
  }
  into = qrs.str();
}

/// Unpack the quality results from string @a from into @a qr.
/// Assumes the data was saved with packQualityData().
void DQMNet::unpackQualityData(QReports &qr, uint32_t &flags, const char *from) {
  const char *qdata = from;

  // Count how many qresults there are.
  size_t nqv = 0;
  while (*qdata) {
    ++nqv;
    while (*qdata)
      ++qdata;
    ++qdata;
    while (*qdata)
      ++qdata;
    ++qdata;
    while (*qdata)
      ++qdata;
    ++qdata;
    while (*qdata)
      ++qdata;
    ++qdata;
    while (*qdata)
      ++qdata;
    ++qdata;
  }

  // Now extract the qreports.
  qdata = from;
  qr.reserve(nqv);
  while (*qdata) {
    qr.emplace_back();
    DQMNet::QValue &qv = qr.back();

    qv.code = atoi(qdata);
    while (*qdata)
      ++qdata;
    switch (qv.code) {
      case dqm::qstatus::STATUS_OK:
        break;
      case dqm::qstatus::WARNING:
        flags |= DQMNet::DQM_PROP_REPORT_WARN;
        break;
      case dqm::qstatus::ERROR:
        flags |= DQMNet::DQM_PROP_REPORT_ERROR;
        break;
      default:
        flags |= DQMNet::DQM_PROP_REPORT_OTHER;
        break;
    }

    qv.qtresult = atof(++qdata);
    while (*qdata)
      ++qdata;

    qv.qtname = ++qdata;
    while (*qdata)
      ++qdata;

    qv.algorithm = ++qdata;
    while (*qdata)
      ++qdata;

    qv.message = ++qdata;
    while (*qdata)
      ++qdata;
    ++qdata;
  }
}

#if 0
// Deserialise a ROOT object from a buffer at the current position.
static TObject *
extractNextObject(TBufferFile &buf)
{
  if (buf.Length() == buf.BufferSize())
    return 0;

  buf.InitMap();
  Int_t pos = buf.Length();
  TClass *c = buf.ReadClass();
  buf.SetBufferOffset(pos);
  buf.ResetMap();
  return c ? buf.ReadObject(c) : 0;
}

// Reconstruct an object from the raw data.
bool
DQMNet::reconstructObject(Object &o)
{
  TBufferFile buf(TBufferFile::kRead, o.rawdata.size(), &o.rawdata[0], kFALSE);
  buf.Reset();

  // Extract the main object.
  if (! (o.object = extractNextObject(buf)))
    return false;
  
  // Extract the reference object.
  o.reference = extractNextObject(buf);

  // Extract quality reports.
  unpackQualityData(o.qreports, o.flags, o.qdata.c_str());
  return true;
}
#endif

#if 0
bool
DQMNet::reinstateObject(DQMStore *store, Object &o)
{
  if (! reconstructObject (o))
    return false;

  // Reconstruct the main object
  MonitorElement *obj = 0;
  store->setCurrentFolder(o.dirname);
  switch (o.flags & DQM_PROP_TYPE_MASK)
  {
  case DQM_PROP_TYPE_INT:
    obj = store->bookInt(o.objname);
    obj->Fill(atoll(o.scalar.c_str()));
    break;

  case DQM_PROP_TYPE_REAL:
    obj = store->bookFloat(name);
    obj->Fill(atof(o.scalar.c_str()));
    break;

  case DQM_PROP_TYPE_STRING:
    obj = store->bookString(name, o.scalar);
    break;

  case DQM_PROP_TYPE_TH1F:
    obj = store->book1D(name, dynamic_cast<TH1F *>(o.object));
    break;

  case DQM_PROP_TYPE_TH1S:
    obj = store->book1S(name, dynamic_cast<TH1S *>(o.object));
    break;

  case DQM_PROP_TYPE_TH1D:
    obj = store->book1DD(name, dynamic_cast<TH1D *>(o.object));
    break;

  case DQM_PROP_TYPE_TH1I:
    obj = store->book1I(name, dynamic_cast<TH1I *>(o.object));
    break;

  case DQM_PROP_TYPE_TH2F:
    obj = store->book2D(name, dynamic_cast<TH2F *>(o.object));
    break;

  case DQM_PROP_TYPE_TH2S:
    obj = store->book2S(name, dynamic_cast<TH2S *>(o.object));
    break;

  case DQM_PROP_TYPE_TH2D:
    obj = store->book2DD(name, dynamic_cast<TH2D *>(o.object));
    break;

 case DQM_PROP_TYPE_TH2I:
    obj = store->book2I(name, dynamic_cast<TH2I *>(o.object));
    break;

  case DQM_PROP_TYPE_TH3F:
    obj = store->book3D(name, dynamic_cast<TH3F *>(o.object));
    break;

  case DQM_PROP_TYPE_TH3S:
    obj = store->book3S(name, dynamic_cast<TH3S *>(o.object));
    break;

  case DQM_PROP_TYPE_TH3D:
    obj = store->book3DD(name, dynamic_cast<TH3D *>(o.object));
    break;

  case DQM_PROP_TYPE_PROF:
    obj = store->bookProfile(name, dynamic_cast<TProfile *>(o.object));
    break;

  case DQM_PROP_TYPE_PROF2D:
    obj = store->bookProfile2D(name, dynamic_cast<TProfile2D *>(o.object));
    break;

  default:
    logme()
      << "ERROR: unexpected monitor element of type "
      << (o.flags & DQM_PROP_TYPE_MASK) << " called '"
      << o.dirname << '/' << o.objname << "'\n";
    return false;
  }

  // Reconstruct tag and qreports.
  if (obj)
  {
    obj->data_.tag = o.tag;
    obj->data_.qreports = o.qreports;
  }

  // Inidicate success.
  return true;
}
#endif

//////////////////////////////////////////////////////////////////////
// Check if the network layer should stop.
bool DQMNet::shouldStop() { return shutdown_; }

// Once an object has been updated, this is invoked for all waiting
// peers.  Send the requested object to the waiting peer.
void DQMNet::releaseFromWait(Bucket *msg, WaitObject &w, Object *o) {
  if (o)
    sendObjectToPeer(msg, *o, true);
  else {
    uint32_t words[3];
    words[0] = sizeof(words) + w.name.size();
    words[1] = DQM_REPLY_NONE;
    words[2] = w.name.size();

    msg->data.reserve(msg->data.size() + words[0]);
    copydata(msg, &words[0], sizeof(words));
    copydata(msg, &w.name[0], w.name.size());
  }
}

// Send an object to a peer.  If not @a data, only sends a summary
// without the object data, except the data is always sent for scalar
// objects.
void DQMNet::sendObjectToPeer(Bucket *msg, Object &o, bool data) {
  uint32_t flags = o.flags & ~DQM_PROP_DEAD;
  DataBlob objdata;

  if ((flags & DQM_PROP_TYPE_MASK) <= DQM_PROP_TYPE_SCALAR)
    objdata.insert(objdata.end(), &o.scalar[0], &o.scalar[0] + o.scalar.size());
  else if (data)
    objdata.insert(objdata.end(), &o.rawdata[0], &o.rawdata[0] + o.rawdata.size());

  uint32_t words[9];
  uint32_t namelen = o.dirname.size() + o.objname.size() + 1;
  uint32_t datalen = objdata.size();
  uint32_t qlen = o.qdata.size();

  if (o.dirname.empty())
    --namelen;

  words[0] = 9 * sizeof(uint32_t) + namelen + datalen + qlen;
  words[1] = DQM_REPLY_OBJECT;
  words[2] = flags;
  words[3] = (o.version >> 0) & 0xffffffff;
  words[4] = (o.version >> 32) & 0xffffffff;
  words[5] = o.tag;
  words[6] = namelen;
  words[7] = datalen;
  words[8] = qlen;

  msg->data.reserve(msg->data.size() + words[0]);
  copydata(msg, &words[0], 9 * sizeof(uint32_t));
  if (namelen) {
    copydata(msg, &(o.dirname)[0], o.dirname.size());
    if (!o.dirname.empty())
      copydata(msg, "/", 1);
    copydata(msg, &o.objname[0], o.objname.size());
  }
  if (datalen)
    copydata(msg, &objdata[0], datalen);
  if (qlen)
    copydata(msg, &o.qdata[0], qlen);
}

//////////////////////////////////////////////////////////////////////
// Handle peer messages.
bool DQMNet::onMessage(Bucket *msg, Peer *p, unsigned char *data, size_t len) {
  // Decode and process this message.
  uint32_t type;
  memcpy(&type, data + sizeof(uint32_t), sizeof(type));
  switch (type) {
    case DQM_MSG_UPDATE_ME: {
      if (len != 2 * sizeof(uint32_t)) {
        logme() << "ERROR: corrupt 'UPDATE_ME' message of length " << len << " from peer " << p->peeraddr << std::endl;
        return false;
      }

      if (debug_)
        logme() << "DEBUG: received message 'UPDATE ME' from peer " << p->peeraddr << ", size " << len << std::endl;

      p->update = true;
    }
      return true;

    case DQM_MSG_LIST_OBJECTS: {
      if (debug_)
        logme() << "DEBUG: received message 'LIST OBJECTS' from peer " << p->peeraddr << ", size " << len << std::endl;

      // Send over current status: list of known objects.
      sendObjectListToPeer(msg, true, false);
    }
      return true;

    case DQM_MSG_GET_OBJECT: {
      if (debug_)
        logme() << "DEBUG: received message 'GET OBJECT' from peer " << p->peeraddr << ", size " << len << std::endl;

      if (len < 3 * sizeof(uint32_t)) {
        logme() << "ERROR: corrupt 'GET IMAGE' message of length " << len << " from peer " << p->peeraddr << std::endl;
        return false;
      }

      uint32_t namelen;
      memcpy(&namelen, data + 2 * sizeof(uint32_t), sizeof(namelen));
      if (len != 3 * sizeof(uint32_t) + namelen) {
        logme() << "ERROR: corrupt 'GET OBJECT' message of length " << len << " from peer " << p->peeraddr
                << ", expected length " << (3 * sizeof(uint32_t)) << " + " << namelen << std::endl;
        return false;
      }

      std::string name((char *)data + 3 * sizeof(uint32_t), namelen);
      Peer *owner = nullptr;
      Object *o = findObject(nullptr, name, &owner);
      if (o) {
        o->lastreq = Time::current().ns();
        if ((o->rawdata.empty() || (o->flags & DQM_PROP_STALE)) &&
            (o->flags & DQM_PROP_TYPE_MASK) > DQM_PROP_TYPE_SCALAR)
          waitForData(p, name, "", owner);
        else
          sendObjectToPeer(msg, *o, true);
      } else {
        uint32_t words[3];
        words[0] = sizeof(words) + name.size();
        words[1] = DQM_REPLY_NONE;
        words[2] = name.size();

        msg->data.reserve(msg->data.size() + words[0]);
        copydata(msg, &words[0], sizeof(words));
        copydata(msg, &name[0], name.size());
      }
    }
      return true;

    case DQM_REPLY_LIST_BEGIN: {
      if (len != 4 * sizeof(uint32_t)) {
        logme() << "ERROR: corrupt 'LIST BEGIN' message of length " << len << " from peer " << p->peeraddr << std::endl;
        return false;
      }

      // Get the update status: whether this is a full update.
      uint32_t flags;
      memcpy(&flags, data + 3 * sizeof(uint32_t), sizeof(uint32_t));

      if (debug_)
        logme() << "DEBUG: received message 'LIST BEGIN " << (flags ? "FULL" : "INCREMENTAL") << "' from "
                << p->peeraddr << ", size " << len << std::endl;

      // If we are about to receive a full list of objects, flag all
      // objects as possibly dead.  Subsequent object notifications
      // will undo this for the live objects.  We cannot delete
      // objects quite yet, as we may get inquiry from another client
      // while we are processing the incoming list, so we keep the
      // objects tentatively alive as long as we've not seen the end.
      if (flags)
        markObjectsDead(p);
    }
      return true;

    case DQM_REPLY_LIST_END: {
      if (len != 4 * sizeof(uint32_t)) {
        logme() << "ERROR: corrupt 'LIST END' message of length " << len << " from peer " << p->peeraddr << std::endl;
        return false;
      }

      // Get the update status: whether this is a full update.
      uint32_t flags;
      memcpy(&flags, data + 3 * sizeof(uint32_t), sizeof(uint32_t));

      // If we received a full list of objects, now purge all dead
      // objects. We need to do this in two stages in case we receive
      // updates in many parts, and end up sending updates to others in
      // between; this avoids us lying live objects are dead.
      if (flags)
        purgeDeadObjects(p);

      if (debug_)
        logme() << "DEBUG: received message 'LIST END " << (flags ? "FULL" : "INCREMENTAL") << "' from " << p->peeraddr
                << ", size " << len << std::endl;

      // Indicate we have received another update from this peer.
      // Also indicate we should flush to our clients.
      flush_ = true;
      p->updates++;
    }
      return true;

    case DQM_REPLY_OBJECT: {
      uint32_t words[9];
      if (len < sizeof(words)) {
        logme() << "ERROR: corrupt 'OBJECT' message of length " << len << " from peer " << p->peeraddr << std::endl;
        return false;
      }

      memcpy(&words[0], data, sizeof(words));
      uint32_t &namelen = words[6];
      uint32_t &datalen = words[7];
      uint32_t &qlen = words[8];

      if (len != sizeof(words) + namelen + datalen + qlen) {
        logme() << "ERROR: corrupt 'OBJECT' message of length " << len << " from peer " << p->peeraddr
                << ", expected length " << sizeof(words) << " + " << namelen << " + " << datalen << " + " << qlen
                << std::endl;
        return false;
      }

      unsigned char *namedata = data + sizeof(words);
      unsigned char *objdata = namedata + namelen;
      unsigned char *qdata = objdata + datalen;
      unsigned char *enddata = qdata + qlen;
      std::string name((char *)namedata, namelen);
      assert(enddata == data + len);

      if (debug_)
        logme() << "DEBUG: received message 'OBJECT " << name << "' from " << p->peeraddr << ", size " << len
                << std::endl;

      // Mark the peer as a known object source.
      p->source = true;

      // Initialise or update an object entry.
      Object *o = findObject(p, name);
      if (!o)
        o = makeObject(p, name);

      o->flags = words[2] | DQM_PROP_NEW | DQM_PROP_RECEIVED;
      o->tag = words[5];
      o->version = ((uint64_t)words[4] << 32 | words[3]);
      o->scalar.clear();
      o->qdata.clear();
      if ((o->flags & DQM_PROP_TYPE_MASK) <= DQM_PROP_TYPE_SCALAR) {
        o->rawdata.clear();
        o->scalar.insert(o->scalar.end(), objdata, qdata);
      } else if (datalen) {
        o->rawdata.clear();
        o->rawdata.insert(o->rawdata.end(), objdata, qdata);
      } else if (!o->rawdata.empty())
        o->flags |= DQM_PROP_STALE;
      o->qdata.insert(o->qdata.end(), qdata, enddata);

      // If we had an object for this one already and this is a list
      // update without data, issue an immediate data get request.
      if (o->lastreq && !datalen && (o->flags & DQM_PROP_TYPE_MASK) > DQM_PROP_TYPE_SCALAR)
        requestObjectData(p, (namelen ? &name[0] : nullptr), namelen);

      // If we have the object data, release from wait.
      if (datalen)
        releaseWaiters(name, o);
    }
      return true;

    case DQM_REPLY_NONE: {
      uint32_t words[3];
      if (len < sizeof(words)) {
        logme() << "ERROR: corrupt 'NONE' message of length " << len << " from peer " << p->peeraddr << std::endl;
        return false;
      }

      memcpy(&words[0], data, sizeof(words));
      uint32_t &namelen = words[2];

      if (len != sizeof(words) + namelen) {
        logme() << "ERROR: corrupt 'NONE' message of length " << len << " from peer " << p->peeraddr
                << ", expected length " << sizeof(words) << " + " << namelen << std::endl;
        return false;
      }

      unsigned char *namedata = data + sizeof(words);
      std::string name((char *)namedata, namelen);

      if (debug_)
        logme() << "DEBUG: received message 'NONE " << name << "' from " << p->peeraddr << ", size " << len
                << std::endl;

      // Mark the peer as a known object source.
      p->source = true;

      // If this was a known object, kill it.
      if (Object *o = findObject(p, name)) {
        o->flags |= DQM_PROP_DEAD;
        purgeDeadObjects(p);
      }

      // If someone was waiting for this, let them go.
      releaseWaiters(name, nullptr);
    }
      return true;

    default:
      logme() << "ERROR: unrecognised message of length " << len << " and type " << type << " from peer " << p->peeraddr
              << std::endl;
      return false;
  }
}

//////////////////////////////////////////////////////////////////////
/// Handle communication to a particular client.
bool DQMNet::onPeerData(IOSelectEvent *ev, Peer *p) {
  lock();
  assert(getPeer(dynamic_cast<Socket *>(ev->source)) == p);

  // If there is a problem with the peer socket, discard the peer
  // and tell the selector to stop prcessing events for it.  If
  // this is a server connection, we will eventually recreate
  // everything if/when the data server comes back.
  if (ev->events & IOUrgent) {
    if (p->automatic) {
      logme() << "WARNING: connection to the DQM server at " << p->peeraddr
              << " lost (will attempt to reconnect in 15 seconds)\n";
      losePeer(nullptr, p, ev);
    } else
      losePeer("WARNING: lost peer connection ", p, ev);

    unlock();
    return true;
  }

  // If we can write to the peer socket, pump whatever we can into it.
  if (ev->events & IOWrite) {
    while (Bucket *b = p->sendq) {
      IOSize len = b->data.size() - p->sendpos;
      const void *data = (len ? (const void *)&b->data[p->sendpos] : (const void *)&data);
      IOSize done;

      try {
        done = (len ? ev->source->write(data, len) : 0);
        if (debug_ && len)
          logme() << "DEBUG: sent " << done << " bytes to peer " << p->peeraddr << std::endl;
      } catch (Error &e) {
        losePeer("WARNING: unable to write to peer ", p, ev, &e);
        unlock();
        return true;
      }

      p->sendpos += done;
      if (p->sendpos == b->data.size()) {
        Bucket *old = p->sendq;
        p->sendq = old->next;
        p->sendpos = 0;
        old->next = nullptr;
        discard(old);
      }

      if (!done && len)
        // Cannot write any more.
        break;
    }
  }

  // If there is data to be read from the peer, first receive what we
  // can get out the socket, the process all complete requests.
  if (ev->events & IORead) {
    // First build up the incoming buffer of data in the socket.
    // Remember the last size returned by the socket; we need
    // it to determine if the remote end closed the connection.
    IOSize sz;
    try {
      std::vector<unsigned char> buf(SOCKET_READ_SIZE);
      do
        if ((sz = ev->source->read(&buf[0], buf.size()))) {
          if (debug_)
            logme() << "DEBUG: received " << sz << " bytes from peer " << p->peeraddr << std::endl;
          DataBlob &data = p->incoming;
          if (data.capacity() < data.size() + sz)
            data.reserve(data.size() + SOCKET_READ_GROWTH);
          data.insert(data.end(), &buf[0], &buf[0] + sz);
        }
      while (sz == sizeof(buf));
    } catch (Error &e) {
      auto *next = dynamic_cast<SystemError *>(e.next());
      if (next && next->portable() == SysErr::ErrTryAgain)
        sz = 1;  // Ignore it, and fake no end of data.
      else {
        // Houston we have a problem.
        losePeer("WARNING: failed to read from peer ", p, ev, &e);
        unlock();
        return true;
      }
    }

    // Process fully received messages as long as we can.
    size_t consumed = 0;
    DataBlob &data = p->incoming;
    while (data.size() - consumed >= sizeof(uint32_t) && p->waiting < MAX_PEER_WAITREQS) {
      uint32_t msglen;
      memcpy(&msglen, &data[0] + consumed, sizeof(msglen));

      if (msglen >= MESSAGE_SIZE_LIMIT) {
        losePeer("WARNING: excessively large message from ", p, ev);
        unlock();
        return true;
      }

      if (data.size() - consumed >= msglen) {
        bool valid = true;
        if (msglen < 2 * sizeof(uint32_t)) {
          logme() << "ERROR: corrupt peer message of length " << msglen << " from peer " << p->peeraddr << std::endl;
          valid = false;
        } else {
          // Decode and process this message.
          Bucket msg;
          msg.next = nullptr;
          valid = onMessage(&msg, p, &data[0] + consumed, msglen);

          // If we created a response, chain it to the write queue.
          if (!msg.data.empty()) {
            Bucket **prev = &p->sendq;
            while (*prev)
              prev = &(*prev)->next;

            *prev = new Bucket;
            (*prev)->next = nullptr;
            (*prev)->data.swap(msg.data);
          }
        }

        if (!valid) {
          losePeer("WARNING: data stream error with ", p, ev);
          unlock();
          return true;
        }

        consumed += msglen;
      } else
        break;
    }

    data.erase(data.begin(), data.begin() + consumed);

    // If the client has closed the connection, shut down our end.  If
    // we have something to send back still, leave the write direction
    // open.  Otherwise close the shop for this client.
    if (sz == 0)
      sel_.setMask(p->socket, p->mask &= ~IORead);
  }

  // Yes, please keep processing events for this socket.
  unlock();
  return false;
}

/** Respond to new connections on the server socket.  Accepts the
    connection and creates a new socket for the peer, and sets it up
    for further communication.  Returns @c false always to tell the
    IOSelector to keep processing events for the server socket.  */
bool DQMNet::onPeerConnect(IOSelectEvent *ev) {
  // Recover the server socket.
  assert(ev->source == server_);

  // Accept the connection.
  Socket *s = server_->accept();
  assert(s);
  assert(!s->isBlocking());

  // Record it to our list of peers.
  lock();
  Peer *p = createPeer(s);
  std::string localaddr;
  if (auto *inet = dynamic_cast<InetSocket *>(s)) {
    InetAddress peeraddr = inet->peername();
    InetAddress myaddr = inet->sockname();
    p->peeraddr = StringFormat("%1:%2").arg(peeraddr.hostname()).arg(peeraddr.port()).value();
    localaddr = StringFormat("%1:%2").arg(myaddr.hostname()).arg(myaddr.port()).value();
  } else if (auto *local = dynamic_cast<LocalSocket *>(s)) {
    p->peeraddr = local->peername().path();
    localaddr = local->sockname().path();
  } else
    assert(false);

  p->mask = IORead | IOUrgent;
  p->socket = s;

  // Report the new connection.
  if (debug_)
    logme() << "INFO: new peer " << p->peeraddr << " is now connected to " << localaddr << std::endl;

  // Attach it to the listener.
  sel_.attach(s, p->mask, CreateHook(this, &DQMNet::onPeerData, p));
  unlock();

  // We are never done.
  return false;
}

/** React to notifications from the DQM thread.  This is a simple
    message to tell this thread to wake up and send unsollicited
    updates to the peers when new DQM data appears.  We don't send
    the updates here, but just set a flag to tell the main event
    pump to send a notification later.  This avoids sending
    unnecessarily frequent DQM object updates.  */
bool DQMNet::onLocalNotify(IOSelectEvent *ev) {
  // Discard the data in the pipe, we care only about the wakeup.
  try {
    IOSize sz;
    unsigned char buf[1024];
    while ((sz = ev->source->read(buf, sizeof(buf))))
      ;
  } catch (Error &e) {
    auto *next = dynamic_cast<SystemError *>(e.next());
    if (next && next->portable() == SysErr::ErrTryAgain)
      ;  // Ignore it
    else
      logme() << "WARNING: error reading from notification pipe: " << e.explain() << std::endl;
  }

  // Tell the main event pump to send an update in a little while.
  flush_ = true;

  // We are never done, always keep going.
  return false;
}

/// Update the selector mask for a peer based on data queues.  Close
/// the connection if there is no reason to maintain it open.
void DQMNet::updateMask(Peer *p) {
  if (!p->socket)
    return;

  // Listen to writes iff we have data to send.
  unsigned oldmask = p->mask;
  if (!p->sendq && (p->mask & IOWrite))
    sel_.setMask(p->socket, p->mask &= ~IOWrite);

  if (p->sendq && !(p->mask & IOWrite))
    sel_.setMask(p->socket, p->mask |= IOWrite);

  if (debug_ && oldmask != p->mask)
    logme() << "DEBUG: updating mask for " << p->peeraddr << " to " << p->mask << " from " << oldmask << std::endl;

  // If we have nothing more to send and are no longer listening
  // for reads, close up the shop for this peer.
  if (p->mask == IOUrgent && !p->waiting) {
    assert(!p->sendq);
    if (debug_)
      logme() << "INFO: connection closed to " << p->peeraddr << std::endl;
    losePeer(nullptr, p, nullptr);
  }
}

//////////////////////////////////////////////////////////////////////
DQMNet::DQMNet(const std::string &appname /* = "" */)
    : debug_(false),
      appname_(appname.empty() ? "DQMNet" : appname.c_str()),
      pid_(getpid()),
      server_(nullptr),
      version_(Time::current()),
      communicate_((pthread_t)-1),
      shutdown_(0),
      delay_(1000),
      waitStale_(0, 0, 0, 0, 500000000 /* 500 ms */),
      waitMax_(0, 0, 0, 5 /* seconds */, 0),
      flush_(false) {
  // Create a pipe for the local DQM to tell the communicator
  // thread that local DQM data has changed and that the peers
  // should be notified.
  fcntl(wakeup_.source()->fd(), F_SETFL, O_RDONLY | O_NONBLOCK);
  sel_.attach(wakeup_.source(), IORead, CreateHook(this, &DQMNet::onLocalNotify));

  // Initialise the upstream and downstream to empty.
  upstream_.peer = downstream_.peer = nullptr;
  upstream_.next = downstream_.next = 0;
  upstream_.port = downstream_.port = 0;
  upstream_.update = downstream_.update = false;
}

DQMNet::~DQMNet() {
  // FIXME
}

/// Enable or disable verbose debugging.  Must be called before
/// calling run() or start().
void DQMNet::debug(bool doit) { debug_ = doit; }

/// Set the I/O dispatching delay.  Must be called before calling
/// run() or start().
void DQMNet::delay(int delay) { delay_ = delay; }

/// Set the time limit for waiting updates to stale objects.
/// Once limit has been exhausted whatever data exists is returned.
/// Applies only when data has been received, another time limit is
/// applied when no data payload has been received at all.
void DQMNet::staleObjectWaitLimit(lat::TimeSpan time) { waitStale_ = time; }

/// Start a server socket for accessing this DQM node remotely.  Must
/// be called before calling run() or start().  May throw an Exception
/// if the server socket cannot be initialised.
void DQMNet::startLocalServer(int port) {
  if (server_) {
    logme() << "ERROR: DQM server was already started.\n";
    return;
  }

  try {
    InetAddress addr("0.0.0.0", port);
    auto *s = new InetSocket(SOCK_STREAM, 0, addr.family());
    s->bind(addr);
    s->listen(10);
    s->setopt(SO_SNDBUF, SOCKET_BUF_SIZE);
    s->setopt(SO_RCVBUF, SOCKET_BUF_SIZE);
    s->setBlocking(false);
    sel_.attach(server_ = s, IOAccept, CreateHook(this, &DQMNet::onPeerConnect));
  } catch (Error &e) {
    // FIXME: Do we need to do this when we throw an exception anyway?
    // FIXME: Abort instead?
    logme() << "ERROR: Failed to start server at port " << port << ": " << e.explain() << std::endl;

    throw cms::Exception("DQMNet::startLocalServer") << "Failed to start server at port " <<

        port << ": " << e.explain().c_str();
  }

  logme() << "INFO: DQM server started at port " << port << std::endl;
}

/// Start a server socket for accessing this DQM node over a file
/// system socket.  Must be called before calling run() or start().
/// May throw an Exception if the server socket cannot be initialised.
void DQMNet::startLocalServer(const char *path) {
  if (server_) {
    logme() << "ERROR: DQM server was already started.\n";
    return;
  }

  try {
    server_ = new LocalServerSocket(path, 10);
    server_->setopt(SO_SNDBUF, SOCKET_BUF_SIZE);
    server_->setopt(SO_RCVBUF, SOCKET_BUF_SIZE);
    server_->setBlocking(false);
    sel_.attach(server_, IOAccept, CreateHook(this, &DQMNet::onPeerConnect));
  } catch (Error &e) {
    // FIXME: Do we need to do this when we throw an exception anyway?
    // FIXME: Abort instead?
    logme() << "ERROR: Failed to start server at path " << path << ": " << e.explain() << std::endl;

    throw cms::Exception("DQMNet::startLocalServer")
        << "Failed to start server at path " << path << ": " << e.explain().c_str();
  }

  logme() << "INFO: DQM server started at path " << path << std::endl;
}

/// Tell the network layer to connect to @a host and @a port and
/// automatically send updates whenever local DQM data changes.  Must
/// be called before calling run() or start().
void DQMNet::updateToCollector(const std::string &host, int port) {
  if (!downstream_.host.empty()) {
    logme() << "ERROR: Already updating another collector at " << downstream_.host << ":" << downstream_.port
            << std::endl;
    return;
  }

  downstream_.update = true;
  downstream_.host = host;
  downstream_.port = port;
}

/// Tell the network layer to connect to @a host and @a port and
/// automatically receive updates from upstream DQM sources.  Must be
/// called before calling run() or start().
void DQMNet::listenToCollector(const std::string &host, int port) {
  if (!upstream_.host.empty()) {
    logme() << "ERROR: Already receiving data from another collector at " << upstream_.host << ":" << upstream_.port
            << std::endl;
    return;
  }

  upstream_.update = false;
  upstream_.host = host;
  upstream_.port = port;
}

/// Stop the network layer and wait it to finish.
void DQMNet::shutdown() {
  shutdown_ = 1;
  if (communicate_ != (pthread_t)-1)
    pthread_join(communicate_, nullptr);
}

/** A thread to communicate with the distributed memory cache peers.
    All this does is run the loop to respond to new connections.
    Much of the actual work is done when a new connection is
    received, and in pumping data around in response to actual
    requests.  */
static void *communicate(void *obj) {
  sigset_t sigs;
  sigfillset(&sigs);
  pthread_sigmask(SIG_BLOCK, &sigs, nullptr);
  ((DQMNet *)obj)->run();
  return nullptr;
}

/// Acquire a lock on the DQM net layer.
void DQMNet::lock() {
  if (communicate_ != (pthread_t)-1)
    pthread_mutex_lock(&lock_);
}

/// Release the lock on the DQM net layer.
void DQMNet::unlock() {
  if (communicate_ != (pthread_t)-1)
    pthread_mutex_unlock(&lock_);
}

/// Start running the network layer in a new thread.  This is an
/// exclusive alternative to the run() method, which runs the network
/// layer in the caller's thread.
void DQMNet::start() {
  if (communicate_ != (pthread_t)-1) {
    logme() << "ERROR: DQM networking thread has already been started\n";
    return;
  }

  pthread_mutex_init(&lock_, nullptr);
  pthread_create(&communicate_, nullptr, &communicate, this);
}

/** Run the actual I/O processing loop. */
void DQMNet::run() {
  Time now;
  Time nextFlush = 0;
  AutoPeer *automatic[2] = {&upstream_, &downstream_};

  // Perform I/O.  Every once in a while flush updates to peers.
  while (!shouldStop()) {
    for (auto ap : automatic) {
      // If we need a server connection and don't have one yet,
      // initiate asynchronous connection creation.  Swallow errors
      // in case the server won't talk to us.
      if (!ap->host.empty() && !ap->peer && (now = Time::current()) > ap->next) {
        ap->next = now + TimeSpan(0, 0, 0, 15 /* seconds */, 0);
        InetSocket *s = nullptr;
        try {
          InetAddress addr(ap->host.c_str(), ap->port);
          s = new InetSocket(SOCK_STREAM, 0, addr.family());
          s->setBlocking(false);
          s->connect(addr);
          s->setopt(SO_SNDBUF, SOCKET_BUF_SIZE);
          s->setopt(SO_RCVBUF, SOCKET_BUF_SIZE);
        } catch (Error &e) {
          auto *sys = dynamic_cast<SystemError *>(e.next());
          if (!sys || sys->portable() != SysErr::ErrOperationInProgress) {
            // "In progress" just means the connection is in progress.
            // The connection is ready when the socket is writeable.
            // Anything else is a real problem.
            if (s)
              s->abort();
            delete s;
            s = nullptr;
          }
        }

        // Set up with the selector if we were successful.  If this is
        // the upstream collector, queue a request for updates.
        if (s) {
          Peer *p = createPeer(s);
          ap->peer = p;

          InetAddress peeraddr = ((InetSocket *)s)->peername();
          InetAddress myaddr = ((InetSocket *)s)->sockname();
          p->peeraddr = StringFormat("%1:%2").arg(peeraddr.hostname()).arg(peeraddr.port()).value();
          p->mask = IORead | IOWrite | IOUrgent;
          p->update = ap->update;
          p->automatic = ap;
          p->socket = s;
          sel_.attach(s, p->mask, CreateHook(this, &DQMNet::onPeerData, p));
          if (ap == &upstream_) {
            uint32_t words[4] = {2 * sizeof(uint32_t), DQM_MSG_LIST_OBJECTS, 2 * sizeof(uint32_t), DQM_MSG_UPDATE_ME};
            p->sendq = new Bucket;
            p->sendq->next = nullptr;
            copydata(p->sendq, words, sizeof(words));
          }

          // Report the new connection.
          if (debug_)
            logme() << "INFO: now connected to " << p->peeraddr << " from " << myaddr.hostname() << ":" << myaddr.port()
                    << std::endl;
        }
      }
    }

    // Pump events for a while.
    sel_.dispatch(delay_);
    now = Time::current();
    lock();

    // Check if flush is required.  Flush only if one is needed.
    // Always sends the full object list, but only rarely.
    if (flush_ && now > nextFlush) {
      flush_ = false;
      nextFlush = now + TimeSpan(0, 0, 0, 15 /* seconds */, 0);
      sendObjectListToPeers(true);
    }

    // Update the data server and peer selection masks.  If we
    // have no more data to send and listening for writes, remove
    // the write mask.  If we have something to write and aren't
    // listening for writes, start listening so we can send off
    // the data.
    updatePeerMasks();

    // Release peers that have been waiting for data for too long.
    Time waitold = now - waitMax_;
    Time waitstale = now - waitStale_;
    for (auto i = waiting_.begin(), e = waiting_.end(); i != e;) {
      Object *o = findObject(nullptr, i->name);

      // If we have (stale) object data, wait only up to stale limit.
      // Otherwise if we have no data at all, wait up to the max limit.
      if (i->time < waitold) {
        logme() << "WARNING: source not responding in " << (waitMax_.ns() * 1e-9) << "s to retrieval, releasing '"
                << i->name << "' from wait, have " << (o ? o->rawdata.size() : 0) << " data available\n";
        releaseFromWait(i++, o);
      } else if (i->time < waitstale && o && (o->flags & DQM_PROP_STALE)) {
        logme() << "WARNING: source not responding in " << (waitStale_.ns() * 1e-9) << "s to update, releasing '"
                << i->name << "' from wait, have " << o->rawdata.size() << " data available\n";
        releaseFromWait(i++, o);
      }

      // Keep it for now.
      else
        ++i;
    }

    unlock();
  }
}

// Tell the network cache that there have been local changes that
// should be advertised to the downstream listeners.
void DQMNet::sendLocalChanges() {
  char byte = 0;
  wakeup_.sink()->write(&byte, 1);
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
DQMBasicNet::DQMBasicNet(const std::string &appname /* = "" */) : DQMImplNet<DQMNet::Object>(appname) {
  local_ = static_cast<ImplPeer *>(createPeer((Socket *)-1));
}

/// Give a hint of how much capacity to allocate for local objects.
void DQMBasicNet::reserveLocalSpace(uint32_t size) { local_->objs.resize(size); }

/// Update the network cache for an object.  The caller must call
/// sendLocalChanges() later to push out the changes.
void DQMBasicNet::updateLocalObject(Object &o) {
  o.dirname = *local_->dirs.insert(o.dirname).first;
  std::pair<ObjectMap::iterator, bool> info(local_->objs.insert(o));
  if (!info.second) {
    // Somewhat hackish. Sets are supposedly immutable, but we
    // need to change the non-key parts of the object. Erasing
    // and re-inserting would produce too much memory churn.
    auto &old = const_cast<Object &>(*info.first);
    std::swap(old.flags, o.flags);
    std::swap(old.tag, o.tag);
    std::swap(old.version, o.version);
    std::swap(old.qreports, o.qreports);
    std::swap(old.rawdata, o.rawdata);
    std::swap(old.scalar, o.scalar);
    std::swap(old.qdata, o.qdata);
  }
}

/// Delete all local objects not in @a known.  Returns true if
/// something was removed.  The caller must call sendLocalChanges()
/// later to push out the changes.
bool DQMBasicNet::removeLocalExcept(const std::set<std::string> &known) {
  size_t removed = 0;
  std::string path;
  ObjectMap::iterator i, e;
  for (i = local_->objs.begin(), e = local_->objs.end(); i != e;) {
    path.clear();
    path.reserve(i->dirname.size() + i->objname.size() + 2);
    path += i->dirname;
    if (!path.empty())
      path += '/';
    path += i->objname;

    if (!known.count(path))
      ++removed, local_->objs.erase(i++);
    else
      ++i;
  }

  return removed > 0;
}
