import os.path, socket ; global CONFIGDIR
def reglob(pattern):
  """Extended version of glob that uses regular expressions."""
  from os import listdir
  import re
  cwd = pattern.rsplit('/',1)[0]
  f_pattern= pattern.rsplit('/',1)[-1]
  pat=re.compile(f_pattern)
  g = ["%s/%s" % (cwd,f) for f in listdir(cwd) if pat.match(f)]
  return g

CONFIGDIR = os.path.normcase(os.path.abspath(__file__)).rsplit('/', 1)[0]
BASEDIR   = CONFIGDIR.replace("/current/config/dqmgui", "")
STATEDIR  = "%s/state/dqmgui/online" % BASEDIR
LOGDIR    = "%s/logs/dqmgui/online" % BASEDIR

# Modifiable parameters.
LAYOUTS = reglob("%s/layouts/[^-_]*-layouts.py" % CONFIGDIR)
LAYOUTS += reglob("%s/layouts/shift_[^-_]*_layout.py" % CONFIGDIR)
LAYOUTS += reglob("%s/layouts/.*_overview_layouts.py" % CONFIGDIR)

# Do not modify configuration below this line.
DQMSERVERS  = ["dqm-prod-local", "dqm-prod-offsite", "dqm-integration", "dqm-test"]
HOST        = socket.gethostname().lower()
DOMAIN      = socket.getfqdn().split('.',1)[-1].lower()
HOSTADDR    = socket.getaddrinfo(HOST, None)[0][4][0]
BASEDIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOSTALIAS   = HOST
COLLHOST    = 'localhost'
COLLPORT    = DOMAIN == 'cms' and 9190 or 8061
#SERVICENAME = 'Online Development'
SERVICENAME = 'ECAL DQM'
#SERVERPORT  = 8070
SERVERPORT  = 8030
#BASEURL     = '/dqm/online-dev'
BASEURL     = '/dqm/ecal'
UPLOADDIR   = "%s/uploads" % STATEDIR
FILEREPO    = { "ROOT": "%s/data" % STATEDIR }

# Figure out a preferred alias for this out (if any)
for alias in DQMSERVERS:
  try:
    if len([x for x in socket.getaddrinfo(alias, None) if x[4][0] == HOSTADDR]):
      HOSTALIAS = alias
      break
  except: pass

# Figure out settings for DQM servers
if HOSTALIAS in DQMSERVERS:
  COLLPORT    = 9090
  SERVERPORT  = 8030
  if HOSTALIAS == 'dqm-integration':
    SERVICENAME = 'Online Playback'
    BASEURL     = '/dqm/online-playback'

  else:
    FILEREPO    = { "ROOT": "/dqmdata/dqm/repository/original/OnlineData",
                    "Original": "/dqmdata/dqm/repository/original/OnlineData",
                    "Merged": "/dqmdata/dqm/repository/merged/OnlineData" }
    COLLHOST    = HOSTALIAS != 'dqm-prod-local' and 'dqm-prod-local.cms' or 'localhost'
    if HOSTALIAS == 'dqm-test':
      SERVICENAME = 'Online Test'
      BASEURL     = '/dqm/online-test'

    else:
      SERVICENAME = 'Online'
      BASEURL     = '/dqm/online'
      UPLOADDIR   = "/dqmdata/dqm/uploads"

# Server configuration.
modules = ("Monitoring.DQM.GUI",)

#server.instrument  = 'valgrind --num-callers=999 `cmsvgsupp` --error-limit=no'
#server.instrument  = 'valgrind --tool=helgrind --num-callers=999 --error-limit=no'
#server.instrument  = 'igprof -d -t python -pp'
#server.instrument  = 'igprof -d -t python -mp'
server.localBase   = HOSTALIAS
server.serverDir   = STATEDIR
server.port        = SERVERPORT
server.logFile     = '%s/weblog-%%Y%%m%%d.log' % LOGDIR
server.baseUrl     = BASEURL
server.title       = 'CMS data quality'
server.serviceName = SERVICENAME

server.plugin('render', "%s/style/*.cc" % CONFIGDIR)
server.extend('DQMRenderLink', server.pathOfPlugin('render'))
server.extend('DQMToJSON')
server.extend('DQMFileAccess', None, UPLOADDIR,FILEREPO)
server.source('DQMUnknown')
server.source('DQMOverlay')
server.source('DQMStripChart')
server.source('DQMLive', "%s:%s" % (COLLHOST,COLLPORT))
server.source('DQMArchive', "%s/ix" % STATEDIR, '^/Global/')
server.source('DQMLayout')

execfile(CONFIGDIR + "/dqm-services.py")
execfile(CONFIGDIR + "/workspaces-online.py")
