import os.path, socket; global CONFIGDIR
CONFIGDIR = os.path.normcase(os.path.abspath(__file__)).rsplit('/', 1)[0]

# Modifiable parameters.
LAYOUTS = ["%s/%s-layouts.py" % (CONFIGDIR, x) for x in
	   ("castor","csc", "dt", "eb", "ee", "es","hcal", "hcalcalib", "hlt", "hlx", "l1t", "l1temulator", "rpc", "pixel", "sistrip")]
LAYOUTS += ["%s/%s_overview_layouts.py" % (CONFIGDIR, x) for x in
            ("sistrip","ecal","hcal","beammonitor","l1t","hlt")]
LAYOUTS += ["%s/shift_%s_layout.py" % (CONFIGDIR, x) for x in
            ("castor","csc", "dt", "eb", "ee", "es","hcal", "hcalcalib", "hlt", "hlx", "l1t", "l1temulator", "rpc", "pixel", "sistrip" , "fed" )]

# Do not modify configuration below this line.
HOST      = socket.gethostname().lower()
HOSTADDR  = socket.getaddrinfo(HOST, None)[0][4][0]
BASEDIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOSTALIAS = HOST

# Figure out a preferred alias for this out (if any)
for alias in ["dqm-prod-local", "dqm-prod-offsite", "dqm-integration", "dqm-test"]:
  if len([x for x in socket.getaddrinfo(alias, None) if x[4][0] == HOSTADDR]):
    HOSTALIAS = alias
    break

# Determine installation directories.
SRVDIR   = '/home/dqmlocal'
COLLHOST = 'localhost'

# Extension modules and environment to install.
modules = ("GuiDQM",)
envsetup = "export QUIET_ASSERT=a"

# Server configuration.
#server.instrument  = 'valgrind --num-callers=999 `cmsvgsupp` --error-limit=no'
#server.instrument  = 'valgrind --tool=helgrind --num-callers=999 --error-limit=no'
#server.instrument  = 'igprof -t python -pp'
server.localBase   = HOSTALIAS
server.serverDir   = '%s/gui' % SRVDIR
server.baseUrl     = '/dqm/milk'
server.title       = 'CMS data quality'
server.serviceName = 'Milk'

# Contents.
server.plugin('render', BASEDIR + "/style/*.cc")
server.extend('DQMRenderLink', server.pathOfPlugin('render'))
server.source('DQMUnknown')
server.source('DQMOverlay')
server.source('DQMStripChart')
server.source('DQMLive', '%s:9090' % COLLHOST)
server.source('DQMLayout', *LAYOUTS)

execfile(CONFIGDIR + "/dqm-services.py")
execfile(CONFIGDIR + "/workspaces-online.py")
