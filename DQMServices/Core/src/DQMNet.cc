#include "DQMServices/Core/interface/DQMNet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "classlib/sysapi/InetSocket.h" // for completing InetAddress
#include "classlib/iobase/Filename.h"
#include "classlib/utils/TimeInfo.h"
#include "classlib/utils/StringList.h"
#include "classlib/utils/StringFormat.h"
#include "classlib/utils/StringOps.h"
#include "classlib/utils/SystemError.h"
#include "classlib/utils/Regexp.h"
#include "TBuffer.h"
#include "TObjString.h"
#include "TObject.h"
#include "TProfile2D.h"
#include "TProfile.h"
#include "TH3F.h"
#include "TH2F.h"
#include "TH1F.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <cassert>

using namespace lat;

static const Regexp s_rxmeval ("<(.*)>(i|f|s|qr)=(.*)</\\1>");
static const Regexp s_rxmeqr  ("^st\\.(\\d+)\\.(.*)$");

//////////////////////////////////////////////////////////////////////
// Parse an integer parameter in an image spec.  If @ap starts with
// the parameter prefix @a name of length @a len, extracts the number
// that follows the prefix into @a value and advances @a p to the next
// character after the extracted value.  Returns true if the parameter
// was parsed, false otherwise.
static bool parseInt (const char *&p, const char *name, size_t len, int &value)
{
  if (! strncmp(p, name, len))
  {
    value = strtol(p+len, (char **) &p, 10);
    return true;
  }
  return false;
}

//////////////////////////////////////////////////////////////////////
// Generate log prefix.
std::ostream &
DQMNet::logme (void)
{
  return std::cerr
    << Time::current().format(true, "%Y-%m-%d %H:%M:%S")
    << " " << appname_ << "[" << pid_ << "]: ";
}

// Append data into a bucket.
void
DQMNet::copydata(Bucket *b, const void *data, size_t len)
{
  b->data.insert(b->data.end(),
		 (const unsigned char *)data,
		 (const unsigned char *)data + len);
}

// Discard a bucket chain.
void
DQMNet::discard (Bucket *&b)
{
  while (b)
  {
    Bucket *next = b->next;
    delete b;
    b = next;
  }
}

//////////////////////////////////////////////////////////////////////
/** Handle errors with a peer socket.  Zaps the socket send queue,
    the socket itself, detaches the socket from the selector, and
    purges any pending wait requests linked to the socket.  */
bool
DQMNet::losePeer(const char *reason,
		 Peer *peer,
		 IOSelectEvent *ev,
		 Error *err)
{
  if (reason)
    logme ()
      << reason << peer->peeraddr
      << (err ? "; error was: " + err->explain() : std::string(""))
      << std::endl;

  Socket *s = peer->socket;

  for (WaitList::iterator i = waiting_.begin(), e = waiting_.end(); i != e; )
    if (*i == peer)
      waiting_.erase(i++);
    else
      ++i;

  if (ev)
    ev->source = 0;

  discard(peer->sendq);
  if (peer->automatic)
    peer->automatic->peer = 0;

  sel_.detach (s);
  s->close();
  removePeer (peer, s);
  delete s;
  return true;
}

/// Queue an object request to the data server.
void
DQMNet::requestObject(const char *name, size_t len)
{
  Bucket **msg = &upstream_.peer->sendq;
  while (*msg)
    msg = &(*msg)->next;
  *msg = new Bucket;
  (*msg)->next = 0;

  uint32_t words[3];
  words[0] = sizeof(words) + len;
  words[1] = DQM_MSG_GET_OBJECT;
  words[2] = len;
  copydata(*msg, words, sizeof(words));
  copydata(*msg, name, len);
}

/// Queue a request for an object and put a peer into the mode of
/// waiting for object data to appear.
void
DQMNet::waitForData(Peer *p, const std::string &name, const std::string &info)
{
  assert (p->waitobj.empty());
  assert (p->waitinfo.empty());
  assert (! p->waitreq);
  assert (std::find(waiting_.begin(), waiting_.end(), p) == waiting_.end());

  p->waitobj = name;
  p->waitinfo = info;
  p->waitreq = Time::current();

  requestObject(name.size() ? &name[0] : 0, name.size());
  waiting_.push_back(p);
}

// Once an object has been updated, this is invoked for all waiting
// peers.  Send the object back to the peer in suitable form.
void
DQMNet::releaseFromWait(WaitList::iterator i, Object *o)
{
  Peer *p = *i;
  Bucket **msg = &p->sendq;
  while (*msg)
    msg = &(*msg)->next;
  *msg = new Bucket;
  (*msg)->next = 0;

  releaseFromWait(*msg, *p, o);

  waiting_.erase(i);
  p->waitobj.clear();
  p->waitinfo.clear();
  p->waitreq = 0;
}

// Release everyone waiting for the object @a o.
void
DQMNet::releaseWaiters(Object *o)
{
  for (WaitList::iterator i = waiting_.begin(), e = waiting_.end(); i != e; )
    if ((*i)->waitobj == o->name)
      releaseFromWait(i++, o);
    else
      ++i;
}

//////////////////////////////////////////////////////////////////////
// Deserialise a ROOT object from a buffer at the current position.
static TObject *
extractNextObject(TBuffer &buf)
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

// Abort the construction of an object.
static bool
abortReconstructObject(DQMNet::Object &o)
{
  o.qreports.clear();
  delete o.object;
  o.object = 0;
  return false;
}

// Reconstruct an object from the raw data.
bool
DQMNet::reconstructObject(Object &o)
{
  TBuffer buf(TBuffer::kRead, o.rawdata.size(), &o.rawdata[0], kFALSE);
  buf.Reset();

  // Extract the main object.
  if (! (o.object = extractNextObject(buf)))
    return false;
  
  // Extract the reference object.
  o.reference = extractNextObject(buf);

  // Calculate quality report base name.
  int slash = StringOps::rfind(o.name, '/');
  std::string qrbase;
  qrbase.reserve(o.name.size()+2);
  qrbase = (slash == -1 ? o.name : o.name.substr(slash+1, std::string::npos));
  qrbase += ".";

  // Extract quality reports.
  while (TObjString *qrstr = dynamic_cast<TObjString *>(extractNextObject(buf)))
  {
    RegexpMatch m;
    if (! s_rxmeval.match(qrstr->GetName(), 0, 0, &m))
    {
      logme()
	<< "ERROR: unexpected quality report string '"
	<< qrstr->GetName() << "' for object '"
	<< o.name << "'\n";
      return abortReconstructObject(o);
    }

    std::string label = m.matchString(qrstr->GetName(), 1);
    std::string type = m.matchString(qrstr->GetName(), 2);
    std::string value = m.matchString(qrstr->GetName(), 3);

    if (type != "qr")
    {
      logme()
	<< "ERROR: expected a 'qr' for a quality report '"
	<< qrstr->GetName() << "' but found '" << type
	<< "' instead\n";
      return abortReconstructObject(o);
    }

    std::string qrname = label;
    qrname.replace(0, qrbase.size(), "");
    if (qrname == label)
    {
      logme()
	<< "ERROR: quality report label in '"
	<< qrstr->GetName()
	<< "' does not match object name '"
	<< o.name << "'\n";
      return abortReconstructObject(o);
    }

    m.reset();
    if (! s_rxmeqr.match(value, 0, 0, &m))
    {
      logme()
	<< "ERROR: quality test value '"
	<< value << "' is incorrectly formatted\n";
      return abortReconstructObject(o);
    }

    QReports::value_type qval(qrname, QValue());
    qval.second.code = 0;
    qval.second.message = m.matchString(value, 2);
    std::string strcode = m.matchString(value, 1);
    const char *p = strcode.c_str();
    if (! parseInt(p, "", 0, qval.second.code) || *p)
    {
      logme()
	<< "ERROR: failed to determine quality test code from '"
	<< value << "'\n";
      return abortReconstructObject(o);
    }

    o.qreports.insert (qval);
  }

  return true;
}

bool
DQMNet::reinstateObject(DaqMonitorBEInterface *bei, Object &o)
{
  if (! reconstructObject (o))
    return false;

  // Reconstruct the main object
  std::string folder = o.name;
  std::string name = o.name;
  folder.erase(folder.rfind('/'), std::string::npos);
  name.erase(0, name.rfind('/')+1);
  bei->setCurrentFolder(folder);
  if (TProfile2D *t = dynamic_cast<TProfile2D *>(o.object))
    bei->cloneProfile2D(name, t);
  else if (TProfile *t = dynamic_cast<TProfile *>(o.object))
    bei->cloneProfile(name, t);
  else if (TH3F *t = dynamic_cast<TH3F *>(o.object))
    bei->clone3D(name, t);
  else if (TH2F *t = dynamic_cast<TH2F *>(o.object))
    bei->clone2D(name, t);
  else if (TH1F *t = dynamic_cast<TH1F *>(o.object))
    bei->clone1D(name, t);
  else if (TObjString *t = dynamic_cast<TObjString *>(o.object))
  {
    RegexpMatch m;
    if (! s_rxmeval.match(t->GetName(), 0, 0, &m))
    {
      logme()
	<< "ERROR: unexpected monitor element string '"
	<< t->GetName() << "' for object '"
	<< o.name << "'\n";
      return false;
    }

    // std::string label = m.matchString(t->GetName(), 1);
    std::string type = m.matchString(t->GetName(), 2);
    std::string value = m.matchString(t->GetName(), 3);

    if (type == "i")
      bei->bookInt(name)->Fill(atoi(value.c_str()));
    else if (type == "f")
      bei->bookFloat(name)->Fill(atof(value.c_str()));
    else if (type == "s")
      bei->bookString(name, value);
    else
    {
      logme()
	<< "ERROR: unexpected string monitor element of type '"
	<< type << "' (from '" << t->GetName() << "') for object '"
	<< o.name << "'\n";
      return false;
    }
  }

  // Reconstruct tags.  (FIXME: untag old tags first?)
  for (size_t i = 0, e = o.tags.size(); i < e; ++i)
    bei->tag(o.name, o.tags[i]);

  // FIXME: Reference and quality reports?

  // Inidicate success.
  return true;
}

//////////////////////////////////////////////////////////////////////
// Check if the network layer should stop.
bool
DQMNet::shouldStop(void)
{
  return shutdown_;
}

// Once an object has been updated, this is invoked for all waiting
// peers.  Send the requested object back to the peer.
void
DQMNet::releaseFromWait(Bucket *msg, Peer &p, Object *o)
{
  if (o)
    sendObjectToPeer (msg, *o, true, sendScalarAsText_);
  else
  {
    uint32_t words [3];
    words[0] = sizeof(words) + p.waitobj.size();
    words[1] = DQM_REPLY_NONE;
    words[2] = p.waitobj.size();

    msg->data.reserve(msg->data.size() + words[0]);
    copydata(msg, &words[0], sizeof(words));
    copydata(msg, &p.waitobj[0], p.waitobj.size());
  }
}

// Send an object to a peer.  If not @a data, only sends a summary
// without the object data, except the data is always sent for scalar
// objects.  If @a text is @c true, sends an ASCII text version of the
// scalar value instead.
void
DQMNet::sendObjectToPeer(Bucket *msg, Object &o, bool data, bool text)
{
  uint32_t flags = o.flags;
  DataBlob objdata;

  if (text && (flags & DQM_FLAG_SCALAR))
  {
    TObject *obj = o.object;
    if (! obj && o.rawdata.size())
    {
      TBuffer buf(TBuffer::kRead, o.rawdata.size(), &o.rawdata[0], kFALSE);
      buf.InitMap();
      buf.Reset();
      obj = extractNextObject(buf);
    }

    if (TObjString *ostr = dynamic_cast<TObjString *>(obj))
    {
      const TString &s = ostr->String();
      objdata.insert(objdata.end(),
		     (unsigned char *) s.Data(),
		     (unsigned char *) s.Data() + s.Length());
      flags |= DQM_FLAG_TEXT;
    }
  }
  else if (data || (flags & DQM_FLAG_SCALAR))
    objdata.insert(objdata.end(),
		   &o.rawdata[0],
		   &o.rawdata[0] + o.rawdata.size());

  uint32_t words [8];
  uint32_t namelen = o.name.size();
  uint32_t taglen  = o.tags.size() * sizeof(uint32_t);
  uint32_t datalen = objdata.size();

  words[0] = 8*sizeof(uint32_t) + namelen + taglen + datalen;
  words[1] = DQM_REPLY_OBJECT;
  words[2] = flags;
  words[3] = (o.version >> 0 ) & 0xffffffff;
  words[4] = (o.version >> 32) & 0xffffffff;
  words[5] = namelen;
  words[6] = taglen / sizeof(uint32_t);
  words[7] = datalen;

  msg->data.reserve(msg->data.size() + words[0]);
  copydata(msg, &words[0], 8*sizeof(uint32_t));
  if (namelen)
    copydata(msg, &o.name[0], namelen);
  if (taglen)
    copydata(msg, &o.tags[0], taglen);
  if (datalen)
    copydata(msg, &objdata[0], datalen);
}

//////////////////////////////////////////////////////////////////////
// Handle peer messages.
bool
DQMNet::onMessage(Bucket *msg, Peer *p, unsigned char *data, size_t len)
{
  // Decode and process this message.
  uint32_t type;
  memcpy (&type, data + sizeof(uint32_t), sizeof (type));
  switch (type)
  {
  case DQM_MSG_UPDATE_ME:
    {
      if (len != 3*sizeof(uint32_t))
      {
	logme()
	  << "ERROR: corrupt 'UPDATE_ME' message of length " << len
	  << " from peer " << p->peeraddr << std::endl;
	return false;
      }

      // Get the update status: whether this is a full update.
      uint32_t full;
      memcpy(&full, data + 2*sizeof(uint32_t), sizeof(uint32_t));

      if (debug_)
	logme()
	  << "DEBUG: received message 'UPDATE ME' from peer "
	  << p->peeraddr << ", full = " << full << std::endl;

      p->update = true;
      p->updatefull = full;

      if (full && ! requestFullUpdates_)
      {
	if (debug_)
	  logme()
	    << "WARNING: forcing full update request mode on due to "
	    << "request from " << p->peeraddr << std::endl;
	requestFullUpdates_ = true;
	requestFullUpdatesFromPeers();
      }
    }
    return true;

  case DQM_MSG_LIST_OBJECTS:
    {
      if (debug_)
	logme()
	  << "DEBUG: received message 'LIST OBJECTS' from peer "
	  << p->peeraddr << std::endl;

      // Send over current status: list of known objects.
      lock();
      sendObjectListToPeer(msg, p->updatefull, true, false);
      unlock();
    }
    return true;

  case DQM_MSG_GET_OBJECT:
    {
      if (debug_)
	logme()
	  << "DEBUG: received message 'GET OBJECT' from peer "
	  << p->peeraddr << std::endl;

      if (len < 3*sizeof(uint32_t))
      {
	logme()
	  << "ERROR: corrupt 'GET IMAGE' message of length " << len
	  << " from peer " << p->peeraddr << std::endl;
	return false;
      }

      uint32_t namelen;
      memcpy (&namelen, data + 2*sizeof(uint32_t), sizeof(namelen));
      if (len != 3*sizeof(uint32_t) + namelen)
      {
	logme()
	  << "ERROR: corrupt 'GET OBJECT' message of length " << len
	  << " from peer " << p->peeraddr
	  << ", expected length " << (3*sizeof(uint32_t))
	  << " + " << namelen << std::endl;
	return false;
      }

      lock();
      std::string name ((char *) data + 3*sizeof(uint32_t), namelen);
      Object *o = findObject(0, name);
      if (o)
      {
	if (o->rawdata.empty())
	  waitForData(p, name, "");
	else
	  sendObjectToPeer(msg, *o, true, sendScalarAsText_);
      }
      else
      {
	uint32_t words [3];
	words[0] = sizeof(words) + name.size();
	words[1] = DQM_REPLY_NONE;
	words[2] = name.size();

	msg->data.reserve(msg->data.size() + words[0]);
	copydata(msg, &words[0], sizeof(words));
	copydata(msg, &name[0], name.size());
      }
      unlock();
    }
    return true;

  case DQM_REPLY_LIST_BEGIN:
    {
      if (len != 4*sizeof(uint32_t))
      {
	logme()
	  << "ERROR: corrupt 'LIST BEGIN' message of length " << len
	  << " from peer " << p->peeraddr << std::endl;
	return false;
      }

      if (debug_)
	logme()
	  << "DEBUG: received message 'LIST BEGIN' from "
	  << p->peeraddr << std::endl;

      // Get the update status: whether this is a full update.
      uint32_t flags;
      memcpy(&flags, data + 3*sizeof(uint32_t), sizeof(uint32_t));

      // If we are about to receive a full list of objects, flag all
      // objects dead.  Subsequent object notifications will undo this
      // for the live objects.  This tells us the object has been
      // removed, but we can keep making it available for a while if
      // there continues to be interest in it.
      if (flags)
      {
	lock();
	markAllObjectsDead(p);
	unlock();
      }
    }
    return true;

  case DQM_REPLY_LIST_END:
    {
      if (len != 4*sizeof(uint32_t))
      {
	logme()
	  << "ERROR: corrupt 'LIST END' message of length " << len
	  << " from peer " << p->peeraddr << std::endl;
	return false;
      }

      // Get the update status: whether this is a full update.
      uint32_t flags;
      memcpy(&flags, data + 3*sizeof(uint32_t), sizeof(uint32_t));

      if (debug_)
	logme()
	  << "DEBUG: received message 'LIST END' from "
	  << p->peeraddr << std::endl;

      // Send an object update to all peers that requested it
      // and reset the updated flag on the objects.
      lock();
      sendObjectListToPeers(flags != 0);
      unlock();

      // Indicate we have received another update from this peer.
      p->updates++;
    }
    return true;

  case DQM_REPLY_OBJECT:
    {
      uint32_t words[8];
      if (len < sizeof(words))
      {
	logme()
	  << "ERROR: corrupt 'OBJECT' message of length " << len
	  << " from peer " << p->peeraddr << std::endl;
	return false;
      }

      memcpy (&words[0], data, sizeof(words));
      uint32_t &namelen = words[5];
      uint32_t &taglen = words[6];
      uint32_t &datalen = words[7];

      if (len != sizeof(words) + namelen + taglen*sizeof(uint32_t) + datalen)
      {
	logme()
	  << "ERROR: corrupt 'OBJECT' message of length " << len
	  << " from peer " << p->peeraddr
	  << ", expected length " << sizeof(words)
	  << " + " << namelen
	  << " + " << (taglen*sizeof(uint32_t))
	  << " + " << datalen
	  << std::endl;
	return false;
      }

      unsigned char *namedata = data + sizeof(words);
      unsigned char *tagdata = namedata + namelen;
      unsigned char *objdata = tagdata + taglen*sizeof(uint32_t);
      unsigned char *enddata = objdata + datalen;
      std::string name ((char *) namedata, namelen);
      assert (enddata == data + len);

      if (debug_)
	logme()
	  << "DEBUG: received message 'OBJECT " << name
	  << "' from " << p->peeraddr << std::endl;

      // Mark the peer as a known object source.
      p->source = true;

      // Initialise or update an object entry.
      lock();
      Object *o = findObject(p, name);
      if (! o)
	o = makeObject(p, name);

      o->flags = words[2] | DQM_FLAG_NEW | DQM_FLAG_RECEIVED;
      o->version = ((uint64_t) words[4] << 32 | words[3]);
      o->tags.clear();
      o->tags.insert(o->tags.end(), (uint32_t *) tagdata, (uint32_t *) objdata);
      o->rawdata.clear();
      o->rawdata.insert (o->rawdata.end(), objdata, enddata);

      bool hadobject = (o->object != 0);
      delete o->object;
      o->object = 0;
      delete o->reference;
      o->reference = 0;

      // If we had an object for this one already and this is a list
      // update without data, issue an immediate data get request.
      if (hadobject && ! datalen)
	requestObject((namelen ? &name[0] : 0), namelen);

      // If we have the object data, release from wait.
      if (datalen)
	releaseWaiters(o);
      unlock();
    }
    return true;

  case DQM_REPLY_NONE:
    {
      uint32_t words[3];
      if (len < sizeof(words))
      {
	logme()
	  << "ERROR: corrupt 'NONE' message of length " << len
	  << " from peer " << p->peeraddr << std::endl;
	return false;
      }

      memcpy (&words[0], data, sizeof(words));
      uint32_t &namelen = words[2];

      if (len != sizeof(words) + namelen)
      {
	logme()
	  << "ERROR: corrupt 'NONE' message of length " << len
	  << " from peer " << p->peeraddr
	  << ", expected length " << sizeof(words)
	  << " + " << namelen << std::endl;
	return false;
      }

      unsigned char *namedata = data + sizeof(words);
      unsigned char *enddata = namedata + namelen;
      std::string name ((char *) namedata, namelen);
      assert (enddata == data + len);

      if (debug_)
	logme()
	  << "DEBUG: received message 'NONE " << name
	  << "' from " << p->peeraddr << std::endl;

      // Mark the peer as a known object source.
      p->source = true;

      // If this was a known object, update its entry.
      lock();
      Object *o = findObject(p, name);
      if (o)
	o->flags |= DQM_FLAG_DEAD;

      // If someone was waiting for this, let them go.
      releaseWaiters(o);
      unlock();
    }
    return true;

  default:
    logme()
      << "ERROR: unrecognised message of length " << len
      << " and type " << type << " from peer " << p->peeraddr
      << std::endl;
    return false;
  }
}

//////////////////////////////////////////////////////////////////////
/// Handle communication to a particular client.
bool
DQMNet::onPeerData(IOSelectEvent *ev, Peer *p)
{
  assert (getPeer(dynamic_cast<Socket *> (ev->source)) == p);

  // If there is a problem with the peer socket, discard the peer
  // and tell the selector to stop prcessing events for it.  If
  // this is a server connection, we will eventually recreate
  // everything if/when the data server comes back.
  if (ev->events & IOUrgent)
  {
    if (p->automatic)
    {
      logme()
	<< "WARNING: connection to the DQM server at " << p->peeraddr
	<< " lost (will attempt to reconnect in 15 seconds)\n";
      return losePeer(0, p, ev);
    }
    else
      return losePeer("WARNING: lost peer connection ", p, ev);
  }

  // If we can write to the peer socket, pump whatever we can into it.
  if (ev->events & IOWrite)
  {
    while (Bucket *b = p->sendq)
    {
      IOSize len = b->data.size() - p->sendpos;
      const void *data = (len ? (const void *)&b->data[p->sendpos]
			  : (const void *)&data);
      IOSize done;

      try
      {
	done = (len ? ev->source->write (data, len) : 0);
	if (debug_ && len)
	  logme()
	    << "DEBUG: sent " << done << " bytes to peer "
	    << p->peeraddr << std::endl;
      }
      catch (Error &e)
      {
	return losePeer("WARNING: unable to write to peer ",
			p, ev, &e);
      }

      p->sendpos += done;
      if (p->sendpos == b->data.size())
      {
	Bucket *old = p->sendq;
	p->sendq = old->next;
	p->sendpos = 0;
	old->next = 0;
	discard(old);
      }

      if (! done && len)
	// Cannot write any more.
	break;
    }
  }

  // If there is data to be read from the peer, first receive what we
  // can get out the socket, the process all complete requests.
  if (ev->events & IORead)
  {
    // First build up the incoming buffer of data in the socket.
    // Remember the last size returned by the socket; we need
    // it to determine if the remote end closed the connection.
    IOSize sz;
    try
    {
      unsigned char buf [1024];
      do
	if ((sz = ev->source->read(buf, sizeof(buf))))
	{
	  if (debug_)
	    logme()
	      << "DEBUG: received " << sz << " bytes from peer "
	      << p->peeraddr << std::endl;
	  DataBlob &data = p->incoming;
	  if (data.capacity () < data.size () + sz)
	    data.reserve (data.size() + 10240);
	  data.insert (data.end(), buf, buf + sz);
	}
      while (sz == sizeof (buf));
    }
    catch (Error &e)
    {
      SystemError *next = dynamic_cast<SystemError *>(e.next());
      if (next && next->portable() == SysErr::ErrTryAgain)
	sz = 1; // Ignore it, and fake no end of data.
      else
	// Houston we have a problem.
	return losePeer("WARNING: failed to read from peer ",
			p, ev, &e);
    }

    // Process fully received messages.
    size_t consumed = 0;
    DataBlob &data = p->incoming;
    while (data.size()-consumed >= sizeof(uint32_t))
    {
      uint32_t msglen;
      memcpy (&msglen, &data[0]+consumed, sizeof(msglen));

      if (msglen >= 512*1024)
	return losePeer("WARNING: excessively large message from ", p, ev);

      if (data.size()-consumed >= msglen)
      {
	bool valid = true;
	if (msglen < 2*sizeof(uint32_t))
	{
	  logme()
	    << "ERROR: corrupt peer message of length " << msglen
	    << " from peer " << p->peeraddr << std::endl;
	  valid = false;
	}
	else
	{
	  // Create response and chain it to the write queue.
	  Bucket **msg = &p->sendq;
	  while (*msg)
	    msg = &(*msg)->next;
	  *msg = new Bucket;
	  (*msg)->next = 0;

	  // Decode and process this message.
	  valid = onMessage(*msg, p, &data[0]+consumed, msglen);
	}

	if (! valid)
	  return losePeer("WARNING: data stream error with ", p, ev);

	consumed += msglen;
      }
      else
	break;
    }

    data.erase(data.begin(), data.begin()+consumed);

    // If the client has closed the connection, shut down our end.  If
    // we have something to send back still, leave the write direction
    // open.  Otherwise close the shop for this client.
    if (sz == 0)
      sel_.setMask(p->socket, p->mask &= ~IORead);
  }

  // Yes, please keep processing events for this socket.
  return false;
}

/** Respond to new connections on the server socket.  Accepts the
    connection and creates a new socket for the peer, and sets it up
    for further communication.  Returns @c false always to tell the
    IOSelector to keep processing events for the server socket.  */
bool
DQMNet::onPeerConnect(IOSelectEvent *ev)
{
  // Recover the server socket.
  assert (ev->source == server_);

  // Accept the connection.
  Socket *s = server_->accept();
  assert (s);
  assert (! s->isBlocking());

  // Record it to our list of peers.
  Peer *p = createPeer(s);
  InetAddress peeraddr = ((InetSocket *) s)->peername();
  InetAddress myaddr = ((InetSocket *) s)->sockname();
  p->peeraddr = StringFormat("%1:%2")
		.arg(peeraddr.hostname())
		.arg(peeraddr.port());
  p->mask = IORead|IOUrgent;
  p->socket = s;

  // Report the new connection.
  if (debug_)
    logme()
      << "INFO: new peer " << p->peeraddr << " is now connected to "
      << myaddr.hostname() << ":" << myaddr.port() << std::endl;

  // Attach it to the listener.
  sel_.attach(s, p->mask, CreateHook(this, &DQMNet::onPeerData, p));

  // We are never done.
  return false;
}

/** React to notifications from the DQM thread.  This is a simple
    message to tell this thread to wake up and send unsollicited
    updates to the peers when new DQM data appears.  We don't send
    the updates here, but just set a flag to tell the main event
    pump to send a notification later.  This avoids sending
    unnecessarily frequent DQM object updates.  */
bool
DQMNet::onLocalNotify(IOSelectEvent *ev)
{
  // Discard the data in the pipe, we care only about the wakeup.
  try
  {
    IOSize sz;
    unsigned char buf [1024];
    while ((sz = ev->source->read(buf, sizeof(buf))))
      ;
  }
  catch (Error &e)
  {
    SystemError *next = dynamic_cast<SystemError *>(e.next());
    if (next && next->portable() == SysErr::ErrTryAgain)
      ; // Ignore it
    else
      logme()
	<< "WARNING: error reading from notification pipe: "
	<< e.explain() << std::endl;
  }

  // Tell the main event pump to send an update in a little while.
  flush_ = true;

  // We are never done, always keep going.
  return false;
}

/// Update the selector mask for a peer based on data queues.  Close
/// the connection if there is no reason to maintain it open.
void
DQMNet::updateMask(Peer *p)
{
  if (! p->socket)
    return;

  // Listen to writes iff we have data to send.
  unsigned oldmask = p->mask;
  if (! p->sendq && (p->mask & IOWrite))
    sel_.setMask(p->socket, p->mask &= ~IOWrite);

  if (p->sendq && ! (p->mask & IOWrite))
    sel_.setMask(p->socket, p->mask |= IOWrite);

  if (debug_ && oldmask != p->mask)
    logme()
      << "DEBUG: updating mask for " << p->peeraddr << " to "
      << p->mask << " from " << oldmask << std::endl;

  // If we have nothing more to send and are no longer listening
  // for reads, close up the shop for this peer.
  if (p->mask == IOUrgent && ! p->waitreq)
  {
    assert (! p->sendq);
    if (debug_)
      logme() << "INFO: connection closed to " << p->peeraddr << std::endl;
    losePeer(0, p, 0);
  }
}

//////////////////////////////////////////////////////////////////////
DQMNet::DQMNet (const std::string &appname /* = "" */)
  : debug_ (false),
    sendScalarAsText_ (false),
    requestFullUpdates_ (false),
    appname_ (appname.empty() ? "DQMNet" : appname.c_str()),
    pid_ (getpid()),
    server_ (0),
    version_ (Time::current()),
    communicate_ ((pthread_t) -1),
    shutdown_ (0),
    delay_ (1000),
    flush_ (false)
{
  // Create a pipe for the local DQM to tell the communicator
  // thread that local DQM data has changed and that the peers
  // should be notified.
  fcntl(wakeup_.source()->fd(), F_SETFL, O_RDONLY | O_NONBLOCK);
  sel_.attach(wakeup_.source(), IORead, CreateHook(this, &DQMNet::onLocalNotify));

  // Initialise the upstream and downstream to empty.
  upstream_.peer   = downstream_.peer   = 0;
  upstream_.next   = downstream_.next   = 0;
  upstream_.port   = downstream_.port   = 0;
  upstream_.update = downstream_.update = false;
}

DQMNet::~DQMNet(void)
{
  // FIXME
}

/// Enable or disable verbose debugging.  Must be called before
/// calling run() or start().
void
DQMNet::debug(bool doit)
{
  debug_ = doit;
}

/// Set the I/O dispatching delay.  Must be called before calling
/// run() or start().
void
DQMNet::delay(int delay)
{
  delay_ = delay;
}

/// Enable or disable sending scalar monitoring values as text, rather
/// than their ROOT object values.  Must be called before run() or
/// start().
void
DQMNet::sendScalarAsText(bool doit)
{
  sendScalarAsText_ = doit;
}

/// Enable or disable requests for full updates.  Set this to get the
/// "old" DQM networking behaviour to automatically fetch all upstream
/// content when it changes, rather than fetching it lazily as needed.
/// You must call this method if you use receive(); any other use is
/// strongly discouraged.  Must be called before run() or start().
void
DQMNet::requestFullUpdates(bool doit)
{
  requestFullUpdates_ = doit;
}

/// Start a server socket for accessing this DQM node remotely.  Must
/// be called before calling run() or start().  May throw an Exception
/// if the server socket cannot be initialised.
void
DQMNet::startLocalServer(int port)
{
  if (server_)
  {
    logme() << "ERROR: DQM server was already started.\n";
    return;
  }

  try
  {
    server_ = new InetServerSocket(InetAddress (port), 10);
    server_->setBlocking(false);
    sel_.attach(server_, IOAccept, CreateHook(this, &DQMNet::onPeerConnect));
  }
  catch (Error &e)
  {
    // FIXME: Do we need to do this when we throw an exception anyway?
    // FIXME: Abort instead?
    logme()
      << "ERROR: Failed to start server at port " << port << ": "
      << e.explain() << std::endl;

    // FIXME: Throw something simpler that removes the dependency?
    throw cms::Exception("DQMNet::startLocalServer")
      << "Failed to start server at port " << port << ": "
      << e.explain();
  }
  
  logme() << "INFO: DQM server started at port " << port << std::endl;
}

/// Tell the network layer to connect to @a host and @a port and
/// automatically send updates whenever local DQM data changes.  Must
/// be called before calling run() or start().
void
DQMNet::updateToCollector(const std::string &host, int port)
{
  if (! downstream_.host.empty())
  {
    logme()
      << "ERROR: Already updating another collector at "
      << downstream_.host << ":" << downstream_.port << std::endl;
    return;
  }

  downstream_.update = true;
  downstream_.host = host;
  downstream_.port = port;
}

/// Tell the network layer to connect to @a host and @a port and
/// automatically receive updates from upstream DQM sources.  Must be
/// called before calling run() or start().
void
DQMNet::listenToCollector(const std::string &host, int port)
{
  if (! upstream_.host.empty())
  {
    logme()
      << "ERROR: Already receiving data from another collector at "
      << upstream_.host << ":" << upstream_.port << std::endl;
    return;
  }

  upstream_.update = false;
  upstream_.host = host;
  upstream_.port = port;
}

/// Stop the network layer and wait it to finish.
void
DQMNet::shutdown(void)
{
  shutdown_ = 1;
  if (communicate_ != (pthread_t) -1)
    pthread_join(communicate_, 0);
}

/** A thread to communicate with the distributed memory cache peers.
    All this does is run the loop to respond to new connections.
    Much of the actual work is done when a new connection is
    received, and in pumping data around in response to actual
    requests.  */
static void *communicate(void *obj)
{
  sigset_t sigs;
  sigfillset (&sigs);
  pthread_sigmask (SIG_BLOCK, &sigs, 0);
  ((DQMNet *)obj)->run();
  return 0;
}

/// Acquire a lock on the DQM net layer.
void
DQMNet::lock(void)
{
  if (communicate_ != (pthread_t) -1)
    pthread_mutex_lock(&lock_);
}

/// Release the lock on the DQM net layer.
void
DQMNet::unlock(void)
{
  if (communicate_ != (pthread_t) -1)
    pthread_mutex_unlock(&lock_);
}

/// Start running the network layer in a new thread.  This is an
/// exclusive alternative to the run() method, which runs the network
/// layer in the caller's thread.
void
DQMNet::start(void)
{
  if (communicate_ != (pthread_t) -1)
  {
    logme()
      << "ERROR: DQM networking thread has already been started\n";
    return;
  }

  pthread_mutex_init(&lock_, 0);
  pthread_create (&communicate_, 0, &communicate, this);
}

/** Run the actual I/O processing loop. */
void
DQMNet::run(void)
{
  Time now;
  Time nextFlush = 0;
  AutoPeer *automatic[2] = { &upstream_, &downstream_ };

  // Perform I/O.  Every once in a while flush updates to peers.
  while (! shouldStop())
  {
    for (int i = 0; i < 2; ++i)
    {
      AutoPeer *ap = automatic[i];

      // If we need a server connection and don't have one yet,
      // initiate asynchronous connection creation.  Swallow errors
      // in case the server won't talk to us.
      if (! ap->host.empty()
	  && ! ap->peer
	  && (now = Time::current()) > ap->next)
      {
	ap->next = now + TimeSpan(0, 0, 0, 15 /* seconds */, 0);
	InetSocket *s = 0;
	try
	{
	  s = new InetSocket (SocketConst::TypeStream);
	  s->setBlocking (false);
	  s->connect(InetAddress (ap->host.c_str(), ap->port));
	}
	catch (Error &e)
	{
	  SystemError *sys = dynamic_cast<SystemError *>(e.next());
	  if (! sys || sys->portable() != SysErr::ErrOperationInProgress)
	  {
	    // "In progress" just means the connection is in progress.
	    // The connection is ready when the socket is writeable.
	    // Anything else is a real problem.
	    logme()
	      << "WARNING: Unable to initiate connection to the DQM server at "
	      << ap->host << ":" << ap->port << ": " << e.explain()
	      << " (will attempt to reconnect in 15 seconds)\n";

	    if (s)
	      s->abort();
	    delete s;
	    s = 0;
	  }
	}

	// Set up with the selector if we were successful.  If this is
	// the upstream collector, queue a request for updates.
	if (s)
	{
	  lock();
	  Peer *p = createPeer(s);
	  ap->peer = p;
	  unlock();

	  InetAddress peeraddr = ((InetSocket *) s)->peername();
	  InetAddress myaddr = ((InetSocket *) s)->sockname();
	  p->peeraddr = StringFormat("%1:%2")
			.arg(peeraddr.hostname())
		       .arg(peeraddr.port());
	  p->mask = IORead|IOWrite|IOUrgent;
	  p->update = ap->update;
	  p->automatic = ap;
	  p->socket = s;
	  sel_.attach(s, p->mask, CreateHook(this, &DQMNet::onPeerData, p));
	  if (ap == &upstream_)
	  {
	    uint32_t words[5] = { 2*sizeof(uint32_t), DQM_MSG_LIST_OBJECTS,
				  3*sizeof(uint32_t), DQM_MSG_UPDATE_ME,
				  requestFullUpdates_ };
	    p->sendq = new Bucket;
	    p->sendq->next = 0;
	    copydata(p->sendq, words, sizeof(words));
	  }

	  // Report the new connection.
	  if (debug_)
	    logme()
	      << "INFO: now connected to " << p->peeraddr << " from "
	      << myaddr.hostname() << ":" << myaddr.port() << std::endl;
	}
      }
    }

    // Pump events for a while.
    sel_.dispatch(delay_);
    now = Time::current();

    // Check if flush is required.  Flush only if one is needed.
    // Even then prefer to make incremental flushes and send the
    // full list of objects only relatively rarely.
    if (flush_)
    {
      bool fullFlush = false;
      if (now > nextFlush)
      {
	nextFlush = now + TimeSpan(0, 0, 0, 30 /* seconds */, 0);
	fullFlush = true;
      }

      flush_ = false;

      // Queue updates for changed or all objects for peers that have
      // requested automatic notifications and mark updated objects
      // old.
      lock();
      sendObjectListToPeers(fullFlush);
      unlock();
    }

    // Update the data server and peer selection masks.  If we
    // have no more data to send and listening for writes, remove
    // the write mask.  If we have something to write and aren't
    // listening for writes, start listening so we can send off
    // the data.
    updatePeerMasks();

    // Release peers that have been waiting for data for too long.
    lock();
    Time waitold = now - TimeSpan(0, 0, 2 /* minutes */, 0, 0);
    for (WaitList::iterator i = waiting_.begin(), e = waiting_.end(); i != e; )
    {
      // If the peer has waited for too long, send something.
      if ((*i)->waitreq < waitold)
	releaseFromWait(i++, findObject(0, (*i)->waitobj));

      // Keep it for now.
      else
	++i;
    }

    // Compact objects no longer in active use.
    purgeDeadObjects(now - TimeSpan(0, 0, 2 /* minutes */, 0, 0),
		     now - TimeSpan(0, 0, 5 /* minutes */, 0, 0));
    unlock();
  }
}

int
DQMNet::receive(DaqMonitorBEInterface *)
{
  logme() << "ERROR: receive() method is not supported.\n";
  return 0;
}

void
DQMNet::updateLocalObject(Object &o)
{
  logme() << "ERROR: updateLocalObject() method is not supported.\n";
}

void
DQMNet::removeLocalObject(const std::string &name)
{
  logme() << "ERROR: removeLocalObject() method is not supported.\n";
}

// Tell the network cache that there have been local changes that
// should be advertised to the downstream listeners.
void
DQMNet::sendLocalChanges(void)
{
  char byte = 0;
  wakeup_.sink()->write(&byte, 1);
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
DQMBasicNet::DQMBasicNet(const std::string &appname /* = "" */)
  : DQMNet (appname)
{
  local_ = static_cast<BasicPeer *>(createPeer((Socket *) -1));
}

DQMBasicNet::~DQMBasicNet(void)
{
  PeerMap::iterator pi, pe;
  ObjectMap::iterator oi, oe;
  for (pi = peers_.begin(), pe = peers_.end(); pi != pe; ++pi)
    for (oi = pi->second.objs.begin(), oe = pi->second.objs.end(); oi != oe; ++oi)
    {
      Object &o = oi->second;
      delete o.object;
      delete o.reference;
      o.object = 0;
      o.reference = 0;
    }
}

int
DQMBasicNet::receive(DaqMonitorBEInterface *bei)
{
  int updates = 0;

  lock();
  PeerMap::iterator pi, pe;
  ObjectMap::iterator oi, oe;
  for (pi = peers_.begin(), pe = peers_.end(); pi != pe; ++pi)
  {
    BasicPeer &p = pi->second;
    if (&p == local_)
      continue;

    updates += p.updates;

    for (oi = p.objs.begin(), oe = p.objs.end(); oi != oe; )
    {
      Object &o = oi->second;
      if (o.flags & DQM_FLAG_DEAD)
      {
	std::string folder = o.name;
	std::string name = o.name;
	folder.erase(folder.rfind('/'), std::string::npos);
	name.erase(0, name.rfind('/')+1);
	bei->setCurrentFolder(folder);
	bei->removeElement(name);
	p.objs.erase(oi++);
      }
      else if ((o.flags & DQM_FLAG_RECEIVED) && reinstateObject(bei, o))
      {
	o.flags &= ~DQM_FLAG_RECEIVED;
	++oi;
      }
    }
  }
  unlock();

  return updates;
}

DQMNet::Object *
DQMBasicNet::findObject(Peer *p, const std::string &name)
{
  if (p)
  {
    BasicPeer *bp = static_cast<BasicPeer *>(p);
    ObjectMap::iterator pos = bp->objs.find(name);
    return pos == bp->objs.end() ? 0 : &pos->second;
  }
  else
  {
    for (PeerMap::iterator i = peers_.begin(), e = peers_.end(); i != e; ++i)
    {
      ObjectMap::iterator pos = i->second.objs.find(name);
      if (pos != i->second.objs.end())
	return &pos->second;
    }
    return 0;
  }
}

DQMNet::Object *
DQMBasicNet::makeObject(Peer *p, const std::string &name)
{
  BasicPeer *bp = static_cast<BasicPeer *>(p);
  Object *o = &bp->objs[name];
  o->version = 0;
  o->name = name;
  o->object = 0;
  o->reference = 0;
  o->flags = 0;
  o->lastreq = 0;
  return o;
}
 
// Mark all objects as dead.
void
DQMBasicNet::markAllObjectsDead(Peer *p)
{
  BasicPeer *bp = static_cast<BasicPeer *>(p);
  for (ObjectMap::iterator i = bp->objs.begin(), e = bp->objs.end(); i != e; ++i)
    i->second.flags |= DQM_FLAG_DEAD;
}

// Purge all old and dead objects.
void
DQMBasicNet::purgeDeadObjects(Time oldobj, Time deadobj)
{
  PeerMap::iterator pi, pe;
  ObjectMap::iterator oi, oe;
  for (pi = peers_.begin(), pe = peers_.end(); pi != pe; ++pi)
    for (oi = pi->second.objs.begin(), oe = pi->second.objs.end(); oi != oe; )
    {
      Object &o = oi->second;

      // Compact non-scalar objects that are unused.  We send scalar
      // objects to the web server so we keep them around.
      if (o.lastreq < oldobj && o.object && ! (o.flags & DQM_FLAG_SCALAR))
      {
	if (debug_)
	  logme()
	    << "DEBUG: compacting idle '" << o.name
	    << "' from " << pi->second.peeraddr << std::endl;

	delete o.object;
	delete o.reference;
	o.object = 0;
	o.reference = 0;
      }

      // Remove if dead and unused.
      if (o.lastreq < deadobj && (o.flags & DQM_FLAG_DEAD))
      {
	if (debug_)
	  logme()
	    << "DEBUG: removing dead '" << o.name
	    << "' from " << pi->second.peeraddr << std::endl;

	pi->second.objs.erase(oi++);
      }
      else
	++oi;
    }
}

/// Update the network cache for an object.  The caller must call
/// sendLocalChanges() later to push out the changes.
void
DQMBasicNet::updateLocalObject(Object &o)
{
  ObjectMap::iterator pos = local_->objs.find(o.name);
  if (pos == local_->objs.end())
    local_->objs.insert(ObjectMap::value_type(o.name, o));
  else
  {
    std::swap(pos->second.version,   o.version);
    std::swap(pos->second.tags,      o.tags);
    std::swap(pos->second.qreports,  o.qreports);
    std::swap(pos->second.flags,     o.flags);
    std::swap(pos->second.rawdata,   o.rawdata);

    delete pos->second.object;
    pos->second.object = 0;
    delete pos->second.reference;
    pos->second.reference = 0;
    pos->second.lastreq = 0;
  }
}

/// Delete the local object.  The caller must call sendLocalChanges()
/// later to push out the changes.
void
DQMBasicNet::removeLocalObject(const std::string &path)
{
  local_->objs.erase(path);
}

DQMNet::Peer *
DQMBasicNet::getPeer(lat::Socket *s)
{
  PeerMap::iterator pos = peers_.find(s);
  return pos == peers_.end() ? 0 : &pos->second;
}

DQMNet::Peer *
DQMBasicNet::createPeer(lat::Socket *s)
{
  BasicPeer *p = &peers_[s];
  p->socket = 0;
  p->sendq = 0;
  p->sendpos = 0;
  p->mask = 0;
  p->source = false;
  p->update = false;
  p->updated = false;
  p->updatefull = false;
  p->waitreq = 0;
  p->automatic = 0;
  return p;
}

void
DQMBasicNet::removePeer(Peer *p, lat::Socket *s)
{
  BasicPeer *bp = static_cast<BasicPeer *>(p);
  bool needflush = ! bp->objs.empty();

  for (ObjectMap::iterator i = bp->objs.begin(), e = bp->objs.end(); i != e; )
  {
    Object &o = i->second;
    delete o.object;
    delete o.reference;
    bp->objs.erase(i++);
  }

  peers_.erase(s);

  // If we removed a peer with objects, our list of objects
  // has changed and we need to update downstream peers.
  if (needflush)
    sendLocalChanges();
}

/// Send all objects to a peer and optionally mark sent objects old.
void
DQMBasicNet::sendObjectListToPeer(Bucket *msg, bool data, bool all, bool clear)
{
  PeerMap::iterator pi, pe;
  ObjectMap::iterator oi, oe;
  uint32_t numobjs = 0;
  for (pi = peers_.begin(), pe = peers_.end(); pi != pe; ++pi)
    numobjs += pi->second.objs.size();

  msg->data.reserve(msg->data.size() + 300*numobjs);

  uint32_t nupdates = 0;
  uint32_t words [4];
  words[0] = sizeof(words);
  words[1] = DQM_REPLY_LIST_BEGIN;
  words[2] = numobjs;
  words[3] = all;
  DQMNet::copydata(msg, &words[0], sizeof(words));

  for (pi = peers_.begin(), pe = peers_.end(); pi != pe; ++pi)
    for (oi = pi->second.objs.begin(), oe = pi->second.objs.end(); oi != oe; ++oi)
      if (all || (oi->second.flags & DQM_FLAG_NEW))
      {
	sendObjectToPeer(msg, oi->second, data, sendScalarAsText_);
	if (clear)
	  oi->second.flags &= ~DQM_FLAG_NEW;
	++nupdates;
      }

  words[1] = DQM_REPLY_LIST_END;
  words[2] = nupdates;
  copydata(msg, &words[0], sizeof(words));
}

void
DQMBasicNet::sendObjectListToPeers(bool all)
{
  for (PeerMap::iterator i = peers_.begin(), e = peers_.end(); i != e; ++i)
  {
    BasicPeer &p = i->second;
    if (! p.update)
      continue;

    if (debug_)
      logme()
	<< "DEBUG: notifying " << p.peeraddr
	<< ", full = " << p.updatefull << std::endl;

    Bucket **msg = &p.sendq;
    while (*msg)
      msg = &(*msg)->next;
    *msg = new Bucket;
    (*msg)->next = 0;

    sendObjectListToPeer(*msg, p.updatefull, !p.updated || all, true);
    p.updated = true;
  }
}

void
DQMBasicNet::requestFullUpdatesFromPeers(void)
{
  for (PeerMap::iterator i = peers_.begin(), e = peers_.end(); i != e; ++i)
  {
    BasicPeer &p = i->second;
    if (! p.source)
      continue;

    Bucket **msg = &p.sendq;
    while (*msg)
      msg = &(*msg)->next;
    *msg = new Bucket;
    (*msg)->next = 0;

    uint32_t words[3] = { 3*sizeof(uint32_t), DQM_MSG_UPDATE_ME, 1 };
    copydata(*msg, words, sizeof(words));
  }
}

void
DQMBasicNet::updatePeerMasks(void)
{
  for (PeerMap::iterator i = peers_.begin(), e = peers_.end(); i != e; )
    updateMask(&(i++)->second);
}
