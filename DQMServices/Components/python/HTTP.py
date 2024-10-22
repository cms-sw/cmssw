from builtins import range
from io import StringIO
from io import BytesIO
from pycurl import *

class RequestManager:
  """Manager of multiple concurrent or overlapping HTTP requests.

This is a utility class acting as a pump of several overlapping
HTTP requests against any number of HTTP or HTTPS servers. It
uses a configurable number of simultaneous connections, ten by
default. The actual connection layer is handled using curl, and
the client classes need to aware of this to a limited degree.

The client supplies optional callback methods for initialising,
responding and handling errors on connections. At the very least
the request response callback should be defined.

This class is not designed for multi-threaded use. It employs
overlapping requests, but in a single thread. Only one thread
at a time should be calling `process()`; several threads may
call `.put()` provided the caller uses a mutex so that only one
thread calls into the method at a time."""

  def __init__(self, num_connections = 10, ssl_opts = None,
               user_agent = None, request_headers = None,
               request_init = None, request_respond = None,
               request_error = None, handle_init = None):
    """Initialise the request manager. The arguments are:

:arg num_connections: maximum number of simultaneous connections.
:arg ssl_opts: optional SSLOptions (Monitoring.Core.X509) for SSL
X509 parametre values, e.g. for X509 client authentication.
:arg user_agent: sets user agent identification string if defined.
:arg request_headers: if defined, specifies list of additional HTTP
request headers to be added to each request.
:arg request_init: optional callback to initialise requests; the
default assumes each task is a URL to access and sets the `URL`
property on the curl object to the task value.
:arg request_respond: callback for handling responses; at the very
minimum this should be defined as the default one does nothing.
:arg request_error: callback for handling connection errors; the
default one raises a RuntimeException.
:arg handle_init: callback for customising connection handles at
creation time; the callback will be invoked for each connection
object as it's created and queued to the idle connection list."""
    self.request_respond = request_respond or self._request_respond
    self.request_error = request_error or self._request_error
    self.request_init = request_init or self._request_init
    self.cm = CurlMulti()
    self.handles = [Curl() for i in range(0, num_connections)]
    self.free = [c for c in self.handles]
    self.queue = []

    for c in self.handles:
      c.buffer = None
      c.setopt(NOSIGNAL, 1)
      c.setopt(TIMEOUT, 300)
      c.setopt(CONNECTTIMEOUT, 30)
      c.setopt(FOLLOWLOCATION, 1)
      c.setopt(MAXREDIRS, 5)
      if user_agent:
        c.setopt(USERAGENT, user_agent)
      if ssl_opts:
        c.setopt(CAPATH, ssl_opts.ca_path)
        c.setopt(SSLCERT, ssl_opts.cert_file)
        c.setopt(SSLKEY, ssl_opts.key_file)
        if ssl_opts.key_pass:
          c.setopt(SSLKEYPASSWD, ssl_opts.key_pass)
      if request_headers:
        c.setopt(HTTPHEADER, request_headers)
      if handle_init:
        handle_init(c)

  def _request_init(self, c, url):
    """Default request initialisation callback."""
    c.setopt(URL, url)

  def _request_error(self, c, task, errmsg, errno):
    """Default request error callback."""
    raise RuntimeError((task, errmsg, errno))

  def _request_respond(self, *args):
    """Default request response callback."""
    pass

  def put(self, task):
    """Add a new task. The task object should be a tuple and is
passed to ``request_init`` callback passed to the constructor."""
    self.queue.append(task)

  def process(self):
    """Process pending requests until none are left.

This method processes all requests queued with `.put()` until they
have been fully processed. It calls the ``request_respond`` callback
for all successfully completed requests, and ``request_error`` for
all failed ones.

Any new requests added by callbacks by invoking ``put()`` are also
processed before returning."""
    npending = 0
    while self.queue or npending:
      while self.queue and self.free:
        c = self.free.pop()
        c.task = self.queue.pop(0)
        c.buffer = b = BytesIO()
        c.setopt(WRITEFUNCTION, b.write)
        self.request_init(c, *c.task)
        self.cm.add_handle(c)
        npending += 1

      while True:
        ret, nhandles = self.cm.perform()
        if ret != E_CALL_MULTI_PERFORM:
          break

      while True:
        numq, ok, err = self.cm.info_read()

        for c in ok:
          assert npending > 0
          self.cm.remove_handle(c)
          self.request_respond(c)
          c.buffer = None
          self.free.append(c)
          npending -= 1

        for c, errno, errmsg in err:
          assert npending > 0
          self.cm.remove_handle(c)
          self.free.append(c)
          npending -= 1
          self.request_error(c, c.task, errmsg, errno)

        if numq == 0:
          break

      self.cm.select(1.)

