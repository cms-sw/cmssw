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
# include <vector>
# include <string>
# include <list>
# include <map>

class TObject;
class DaqMonitorBEInterface;

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
  static const uint32_t DQM_FLAG_TEXT		= 0x1000000;
  static const uint32_t DQM_FLAG_RECEIVED	= 0x2000000;
  static const uint32_t DQM_FLAG_NEW		= 0x4000000;
  static const uint32_t DQM_FLAG_DEAD		= 0x8000000;

  struct Peer;
  struct QValue;

  typedef std::vector<unsigned char>    DataBlob;
  typedef std::vector<uint32_t>         TagList;
  typedef std::map<std::string, QValue>	QReports;
  typedef std::list<Peer *>		WaitList;

  struct QValue
  {
    int			code;
    std::string		message;
  };

  struct Object
  {
    uint64_t		version;
    std::string		name;
    TagList 		tags;
    TObject		*object;
    TObject		*reference;
    QReports		qreports;
    uint32_t		flags;
    DataBlob		rawdata;
    lat::Time		lastreq;
  };

  struct Bucket
  {
    Bucket		*next;
    DataBlob		data;
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

    lat::Time		waitreq;
    std::string		waitobj;
    std::string		waitinfo;

    AutoPeer		*automatic;
  };

  struct AutoPeer
  {
    Peer		*peer;
    lat::Time		next;
    std::string		host;
    int			port;
    bool		update;
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

  virtual int		receive(DaqMonitorBEInterface *bei) = 0;
  virtual void		updateLocalObject(Object &o) = 0;
  virtual void		removeLocalObject(const std::string &name) = 0;
  void			sendLocalChanges(void);

protected:
  std::ostream &	logme(void);
  static void		copydata(Bucket *b, const void *data, size_t len);
  void			sendObjectToPeer(Bucket *msg, Object &o, bool data, bool text);

  virtual bool		shouldStop(void);
  virtual void		releaseFromWait(Bucket *msg, Peer &p, Object *o);
  virtual bool		onMessage(Bucket *msg, Peer *p, unsigned char *data, size_t len);

  bool			reconstructObject(Object &o);
  bool			reinstateObject(DaqMonitorBEInterface *bei, Object &o);
  virtual Object *	findObject(Peer *p, const std::string &name) = 0;
  virtual Object *	makeObject(Peer *p, const std::string &name) = 0;
  virtual void		markAllObjectsDead(Peer *p) = 0;
  virtual void		purgeDeadObjects(lat::Time oldobj, lat::Time deadobj) = 0;

  virtual Peer *	getPeer(lat::Socket *s) = 0;
  virtual Peer *	createPeer(lat::Socket *s) = 0;
  virtual void		removePeer(Peer *p, lat::Socket *s) = 0;
  virtual void		sendObjectListToPeer(Bucket *msg, bool data, bool all, bool clear) = 0;
  virtual void		sendObjectListToPeers(bool all) = 0;
  virtual void		requestFullUpdatesFromPeers(void) = 0;

  void			updateMask(Peer *p);
  virtual void		updatePeerMasks(void) = 0;

  bool			debug_;
  bool			sendScalarAsText_;
  bool			requestFullUpdates_;

private:
  static void		discard(Bucket *&b);
  bool			losePeer(const char *reason,
				 Peer *peer,
				 lat::IOSelectEvent *event,
				 lat::Error *err = 0);
  void			requestObject(const char *name, size_t len);
  void			waitForData(Peer *p, const std::string &name, const std::string &info);
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

  pthread_mutex_t	lock_;
  pthread_t		communicate_;
  sig_atomic_t		shutdown_;

  int			delay_;
  bool			flush_;

  // copying is not available
  DQMNet(const DQMNet &);
  DQMNet &operator=(const DQMNet &);
};

class DQMBasicNet : public DQMNet
{
public:
  struct BasicPeer;
  typedef std::map<std::string, Object> ObjectMap;
  typedef std::map<lat::Socket *, BasicPeer> PeerMap;

  struct BasicPeer : Peer
  {
    ObjectMap		objs;
  };


  DQMBasicNet(const std::string &appname = "");
  ~DQMBasicNet(void);

  virtual int		receive(DaqMonitorBEInterface *bei);

protected:
  virtual void		updateLocalObject(Object &o);
  virtual void		removeLocalObject(const std::string &name);
  virtual Object *	findObject(Peer *p, const std::string &name);
  virtual Object *	makeObject(Peer *p, const std::string &name);
  virtual void		markAllObjectsDead(Peer *p);
  virtual void		purgeDeadObjects(lat::Time oldobj, lat::Time deadobj);

  virtual Peer *	getPeer(lat::Socket *s);
  virtual Peer *	createPeer(lat::Socket *s);
  virtual void		removePeer(Peer *p, lat::Socket *s);
  virtual void		sendObjectListToPeer(Bucket *msg, bool data, bool all, bool clear);
  virtual void		sendObjectListToPeers(bool all);
  virtual void		requestFullUpdatesFromPeers(void);
  virtual void		updatePeerMasks(void);

private:

  PeerMap		peers_;
  BasicPeer		*local_;
};


#endif // DQMSERVICES_CORE_DQM_NET_H
