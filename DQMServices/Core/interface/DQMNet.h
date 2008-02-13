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

  typedef std::vector<unsigned char>            DataBlob;
  typedef std::vector<uint32_t>                 TagList;
  typedef std::map<std::string, struct QValue>  QReports;
  typedef std::list<Peer *>			WaitList;
  typedef std::map<lat::Socket *, Peer>		PeerMap;

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
    bool		update;
    bool		updated;

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

  virtual bool		shouldStop(void);
  virtual void		releaseFromWait(Bucket *msg, Peer &p, Object *o);
  virtual void		sendObjectToPeer(Bucket *msg, Object &o, bool senddata, bool sendtext);
  virtual bool		onMessage(Bucket *msg, Peer *p, unsigned char *data, size_t len);

  bool			reconstructObject(Object &o);
  bool			reinstateObject(DaqMonitorBEInterface *bei, Object &o);
  virtual Object *	findObject(const std::string &name) = 0;
  virtual Object *	makeObject(const std::string &name) = 0;
  virtual void		sendObjectListToPeer(Bucket *msg, bool allObjects) = 0;
  virtual void		markAllObjectsOld(void) = 0;
  virtual void		markAllObjectsDead(void) = 0;
  virtual void		purgeDeadObjects(lat::Time oldobj, lat::Time deadobj) = 0;

  bool			debug_;
  bool			sendScalarAsText_;

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
  void			updateMask(Peer *p);

  std::string		appname_;
  int			pid_;

  lat::IOSelector	sel_;
  lat::InetServerSocket	*server_;
  lat::Pipe		wakeup_;
  bool			flush_;
  lat::Time		version_;

  PeerMap		peers_;
  AutoPeer		upstream_;
  AutoPeer		downstream_;
  WaitList		waiting_;

  pthread_mutex_t	lock_;
  pthread_t		communicate_;
  sig_atomic_t		shutdown_;

  int			delay_;

  // copying is not available
  DQMNet(const DQMNet &);
  DQMNet &operator=(const DQMNet &);
};

class DQMBasicNet : public DQMNet
{
public:
  DQMBasicNet(const std::string &appname = "");
  ~DQMBasicNet(void);

  virtual int		receive(DaqMonitorBEInterface *bei);

protected:
  virtual void		updateLocalObject(Object &o);
  virtual void		removeLocalObject(const std::string &name);
  virtual Object *	findObject(const std::string &name);
  virtual Object *	makeObject(const std::string &name);
  virtual void		sendObjectListToPeer(Bucket *msg, bool allObjects);
  virtual void		markAllObjectsOld(void);
  virtual void		markAllObjectsDead(void);
  virtual void		purgeDeadObjects(lat::Time oldobj, lat::Time deadobj);

private:
  typedef std::map<std::string, Object>  ObjectMap;
  ObjectMap		objs_;
};


#endif // DQMSERVICES_CORE_DQM_NET_H
