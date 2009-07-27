#ifndef DQMSERVICES_CORE_DQM_NET_H
# define DQMSERVICES_CORE_DQM_NET_H

# include "classlib/iobase/InetServerSocket.h"
# include "classlib/iobase/IOSelector.h"
# include "classlib/iobase/Pipe.h"
# include "classlib/utils/Signal.h"
# include "classlib/utils/Error.h"
# include "classlib/utils/Time.h"
# include <pthread.h>
# include <stdint.h>
# include <iostream>
# include <vector>
# include <string>
# include <list>
# include <map>

class TObject;
class DQMStore;

class DQMNet
{
public:
  static const uint32_t DQM_MSG_HELLO		= 0;
  static const uint32_t DQM_MSG_UPDATE_ME	= 1;
  static const uint32_t DQM_MSG_LIST_OBJECTS	= 2;
  static const uint32_t DQM_MSG_GET_OBJECT	= 3;

  static const uint32_t DQM_REPLY_LIST_BEGIN	= 101;
  static const uint32_t DQM_REPLY_LIST_END	= 102;
  static const uint32_t DQM_REPLY_NONE		= 103;
  static const uint32_t DQM_REPLY_OBJECT	= 104;

  static const uint32_t DQM_FLAG_REPORT_ERROR	= 0x1;
  static const uint32_t DQM_FLAG_REPORT_WARNING	= 0x2;
  static const uint32_t DQM_FLAG_REPORT_OTHER	= 0x4;
  static const uint32_t DQM_FLAG_SCALAR		= 0x8;
  static const uint32_t DQM_FLAG_ZOMBIE		= 0x08000000;
  static const uint32_t DQM_FLAG_TEXT		= 0x10000000;
  static const uint32_t DQM_FLAG_RECEIVED	= 0x20000000;
  static const uint32_t DQM_FLAG_NEW		= 0x40000000;
  static const uint32_t DQM_FLAG_DEAD		= 0x80000000;

  static const uint32_t MAX_PEER_WAITREQS	= 128;

  struct Peer;
  struct QValue;
  struct WaitObject;

  typedef std::vector<unsigned char>    DataBlob;
  typedef std::vector<uint32_t>         TagList;
  typedef std::vector<QValue>		QReports;
  typedef std::list<WaitObject>		WaitList;

  struct QValue
  {
    int			code;
    float		qtresult;
    std::string		message;
    std::string		qtname;
    std::string		algorithm;
  };

  struct CoreObject
  {
    uint64_t		version;
    std::string		name;
    TagList 		tags;
    TObject		*object;
    TObject		*reference;
    QReports		qreports;
    uint32_t		flags;
  };
  
  struct Object : CoreObject
  {
    DataBlob		rawdata;
    lat::Time		lastreq;
  };

  struct Bucket
  {
    Bucket		*next;
    DataBlob		data;
  };

  struct WaitObject
  {
    lat::Time		time;
    std::string		name;
    std::string		info;
    Peer		*peer;
  };

  struct AutoPeer;
  struct Peer
  {
    std::string		peeraddr;
    lat::Socket		*socket;
    DataBlob		incoming;
    Bucket		*sendq;
    size_t		sendpos;

    unsigned		mask;
    bool		source;
    bool		update;
    bool		updated;
    bool		updatefull;
    size_t		updates;
    size_t		waiting;
    AutoPeer		*automatic;
  };

  struct AutoPeer
  {
    Peer		*peer;
    lat::Time		next;
    std::string		host;
    int			port;
    bool		update;
    bool		warned;
  };

  DQMNet(const std::string &appname = "");
  virtual ~DQMNet(void);

  void			debug(bool doit);
  void			delay(int delay);
  void			sendScalarAsText(bool doit);
  void			requestFullUpdates(bool doit);
  void			startLocalServer(int port);
  void			updateToCollector(const std::string &host, int port);
  void			listenToCollector(const std::string &host, int port);
  void			shutdown(void);
  void			lock(void);
  void			unlock(void);

  void			start(void);
  void			run(void);

  virtual int		receive(DQMStore *store);
  virtual void		updateLocalObject(Object &o);
  virtual void		removeLocalObject(const std::string &name);
  void			sendLocalChanges(void);

protected:
  std::ostream &	logme(void);
  static void		copydata(Bucket *b, const void *data, size_t len);
  bool			extractScalarData(DataBlob &objdata, Object &o);
  virtual void		sendObjectToPeer(Bucket *msg, Object &o, bool data, bool text);

  virtual bool		shouldStop(void);
  void			waitForData(Peer *p, const std::string &name, const std::string &info, Peer *owner);
  virtual void		releaseFromWait(Bucket *msg, WaitObject &w, Object *o);
  virtual bool		onMessage(Bucket *msg, Peer *p, unsigned char *data, size_t len);

  bool			reconstructObject(Object &o);
  bool			reinstateObject(DQMStore *store, Object &o);
  virtual Object *	findObject(Peer *p, const std::string &name, Peer **owner = 0) = 0;
  virtual Object *	makeObject(Peer *p, const std::string &name) = 0;
  virtual void		markObjectsZombies(Peer *p) = 0;
  virtual void		markObjectsDead(Peer *p) = 0;
  virtual void		purgeDeadObjects(lat::Time oldobj, lat::Time deadobj) = 0;

  virtual Peer *	getPeer(lat::Socket *s) = 0;
  virtual Peer *	createPeer(lat::Socket *s) = 0;
  virtual void		removePeer(Peer *p, lat::Socket *s) = 0;
  virtual void		sendObjectListToPeer(Bucket *msg, bool data, bool all, bool clear) = 0;
  virtual void		sendObjectListToPeers(bool all) = 0;
  virtual void		requestFullUpdatesFromPeers(void) = 0;

  void			updateMask(Peer *p);
  virtual void		updatePeerMasks(void) = 0;
  static void		discard(Bucket *&b);

  bool			debug_;
  bool			sendScalarAsText_;
  bool			requestFullUpdates_;
  pthread_mutex_t	lock_;

private:
  void			losePeer(const char *reason,
				 Peer *peer,
				 lat::IOSelectEvent *event,
				 lat::Error *err = 0);
  void			requestObject(Peer *p, const char *name, size_t len);
  void			releaseFromWait(WaitList::iterator i, Object *o);
  void			releaseWaiters(Object *o);

  bool			onPeerData(lat::IOSelectEvent *ev, Peer *p);
  bool			onPeerConnect(lat::IOSelectEvent *ev);
  bool			onLocalNotify(lat::IOSelectEvent *ev);

  std::string		appname_;
  int			pid_;

  lat::IOSelector	sel_;
  lat::InetServerSocket	*server_;
  lat::Pipe		wakeup_;
  lat::Time		version_;

  AutoPeer		upstream_;
  AutoPeer		downstream_;
  WaitList		waiting_;

  pthread_t		communicate_;
  sig_atomic_t		shutdown_;

  int			delay_;
  bool			flush_;

  // copying is not available
  DQMNet(const DQMNet &);
  DQMNet &operator=(const DQMNet &);
};

template <class ObjType>
class DQMImplNet : public DQMNet
{
public:
  struct ImplPeer;
  typedef std::map<std::string, ObjType> ObjectMap;
  typedef std::map<lat::Socket *, ImplPeer> PeerMap;
  struct ImplPeer : Peer
  {
    ObjectMap objs;
  };

  DQMImplNet(const std::string &appname = "")
    : DQMNet(appname)
    {}
  
  ~DQMImplNet(void)
    {
      typename PeerMap::iterator pi, pe;
      typename ObjectMap::iterator oi, oe;
      for (pi = peers_.begin(), pe = peers_.end(); pi != pe; ++pi)
	for (oi = pi->second.objs.begin(), oe = pi->second.objs.end(); oi != oe; ++oi)
	{
	  ObjType &o = oi->second;
	  delete o.object;
	  delete o.reference;
	  o.object = 0;
	  o.reference = 0;
	}
    }

protected:
  virtual Object *
  findObject(Peer *p, const std::string &name, Peer **owner = 0)
    {
      typename ObjectMap::iterator pos;
      typename PeerMap::iterator i, e;
      if (owner)
	*owner = 0;
      if (p)
      {
	ImplPeer *ip = static_cast<ImplPeer *>(p);
	pos = ip->objs.find(name);
	if (pos == ip->objs.end())
	  return 0;
	else
	{
	  if (owner) *owner = ip;
	  return &pos->second;
	}
      }
      else
      {
	for (i = peers_.begin(), e = peers_.end(); i != e; ++i)
	{
	  pos = i->second.objs.find(name);
	  if (pos != i->second.objs.end())
	  {
	    if (owner) *owner = &i->second;
	    return &pos->second;
	  }
	}
	return 0;
      }
    }

  virtual Object *
  makeObject(Peer *p, const std::string &name)
    {
      ImplPeer *ip = static_cast<ImplPeer *>(p);
      ObjType *o = &ip->objs[name];
      o->version = 0;
      o->name = name;
      o->object = 0;
      o->reference = 0;
      o->flags = 0;
      o->lastreq = 0;
      return o;
    }

  // Mark all the objects as zombies.  This is intended to be used
  // when starting to process a complete list of objects, in order
  // to flag the objects that need to be killed at the end.  After
  // call to this method, revive all live objects by removing the
  // DQM_FLAG_ZOMBIE flag, then call markObjectsDead() at the end
  // to flag dead as all remaining zombies.
  virtual void
  markObjectsZombies(Peer *p)
    {
      ImplPeer *ip = static_cast<ImplPeer *>(p);
      typename ObjectMap::iterator i, e;
      for (i = ip->objs.begin(), e = ip->objs.end(); i != e; ++i)
	i->second.flags |= DQM_FLAG_ZOMBIE;
    }

  // Mark remaining zombie objects as dead.  See markObjectsZombies().
  virtual void
  markObjectsDead(Peer *p)
    {
      ImplPeer *ip = static_cast<ImplPeer *>(p);
      typename ObjectMap::iterator i, e;
      for (i = ip->objs.begin(), e = ip->objs.end(); i != e; ++i)
	if (i->second.flags & DQM_FLAG_ZOMBIE)
	  i->second.flags = (i->second.flags & ~DQM_FLAG_ZOMBIE) | DQM_FLAG_DEAD;
    }

  // Purge all old and dead objects.
  virtual void
  purgeDeadObjects(lat::Time oldobj, lat::Time deadobj)
    {
      typename PeerMap::iterator pi, pe;
      typename ObjectMap::iterator oi, oe;
      for (pi = peers_.begin(), pe = peers_.end(); pi != pe; ++pi)
	for (oi = pi->second.objs.begin(), oe = pi->second.objs.end(); oi != oe; )
	{
	  ObjType &o = oi->second;

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
	    purgeDeadObject(o);
	  }

	  // Remove if dead, old and unused.
	  if (o.lastreq < deadobj
	      && o.version < deadobj
	      && (o.flags & DQM_FLAG_DEAD))
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

  virtual void
  purgeDeadObject(ObjType &o)
    {}

  virtual Peer *
  getPeer(lat::Socket *s)
    {
      typename PeerMap::iterator pos = peers_.find(s);
      typename PeerMap::iterator end = peers_.end();
      return pos == end ? 0 : &pos->second;
    }

  virtual Peer *
  createPeer(lat::Socket *s)
    {
      ImplPeer *ip = &peers_[s];
      ip->socket = 0;
      ip->sendq = 0;
      ip->sendpos = 0;
      ip->mask = 0;
      ip->source = false;
      ip->update = false;
      ip->updated = false;
      ip->updatefull = false;
      ip->updates = 0;
      ip->waiting = 0;
      ip->automatic = 0;
      return ip;
    }

  virtual void
  removePeer(Peer *p, lat::Socket *s)
    {
      ImplPeer *ip = static_cast<ImplPeer *>(p);
      bool needflush = ! ip->objs.empty();

      typename ObjectMap::iterator i, e;
      for (i = ip->objs.begin(), e = ip->objs.end(); i != e; )
      {
	ObjType &o = i->second;
	delete o.object;
	delete o.reference;
	ip->objs.erase(i++);
      }
    
      peers_.erase(s);

      // If we removed a peer with objects, our list of objects
      // has changed and we need to update downstream peers.
      if (needflush)
	sendLocalChanges();
    }

  /// Send all objects to a peer and optionally mark sent objects old.
  virtual void
  sendObjectListToPeer(Bucket *msg, bool data, bool all, bool clear)
    {
      typename PeerMap::iterator pi, pe;
      typename ObjectMap::iterator oi, oe;
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
      copydata(msg, &words[0], sizeof(words));

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

  virtual void
  sendObjectListToPeers(bool all)
    {
      typename PeerMap::iterator i, e;
      for (i = peers_.begin(), e = peers_.end(); i != e; ++i)
      {
	ImplPeer &p = i->second;
	if (! p.update)
	  continue;

	if (debug_)
	  logme()
	    << "DEBUG: notifying " << p.peeraddr
	    << ", full = " << p.updatefull << std::endl;

	Bucket msg;
        msg.next = 0;
	sendObjectListToPeer(&msg, p.updatefull, !p.updated || all, true);

	if (! msg.data.empty())
	{
	  Bucket **prev = &p.sendq;
	  while (*prev)
	     prev = &(*prev)->next;

	  *prev = new Bucket;
	  (*prev)->next = 0;
	  (*prev)->data.swap(msg.data);
	}
	p.updated = true;
      }
    }

  virtual void
  requestFullUpdatesFromPeers(void)
    {
      logme()
	<< "ERROR: invalid request for full updates from peers.\n";
    }

  virtual void
  updatePeerMasks(void)
    {
      typename PeerMap::iterator i, e;
      for (i = peers_.begin(), e = peers_.end(); i != e; )
	updateMask(&(i++)->second);
    }

protected:
  PeerMap		peers_;
};
  

class DQMBasicNet : public DQMImplNet<DQMNet::Object>
{
public:
  DQMBasicNet(const std::string &appname = "");

  virtual int		receive(DQMStore *store);

protected:
  virtual void		updateLocalObject(Object &o);
  virtual void		removeLocalObject(const std::string &name);
  virtual void		requestFullUpdatesFromPeers(void);

private:
  ImplPeer		*local_;
};


#endif // DQMSERVICES_CORE_DQM_NET_H
