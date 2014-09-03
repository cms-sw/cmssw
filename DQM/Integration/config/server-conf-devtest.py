import os.path
global CONFIGDIR
BASEDIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGDIR = os.path.normcase(os.path.abspath(__file__)).rsplit('/', 1)[0]

LAYOUTS = ["%s/%s-layouts.py" % (CONFIGDIR, x) for x in
	   ("castor","csc", "dt", "eb", "ee", "es","hcal", "hcalcalib", "hlt", "hlx", "l1t", "l1temulator", "rpc", "pixel", "sistrip")]
LAYOUTS += ["%s/%s_overview_layouts.py" % (CONFIGDIR, x) for x in
            ("sistrip","ecal","hcal","beammonitor","l1t","hlt")]
LAYOUTS += ["%s/shift_%s_layout.py" % (CONFIGDIR, x) for x in
            ("castor","csc", "dt", "eb", "ee", "es","hcal", "hcalcalib", "hlt", "hlx", "l1t", "l1temulator", "rpc", "pixel", "sistrip" , "fed" )]

modules = ("GuiDQM",)
envsetup = "export QUIET_ASSERT=a"

#server.instrument  = 'valgrind --num-callers=999 `cmsvgsupp` --error-limit=no'
#server.instrument  = 'valgrind --tool=helgrind --num-callers=999 --error-limit=no'
#server.instrument  = 'igprof -t python -pp'
server.port        = 8888
server.serverDir   = BASEDIR + "/gui"
server.baseUrl     = '/dqm/devtest'
server.title       = 'CMS data quality'
server.serviceName = 'GUI test'

server.plugin('render', BASEDIR + "/style/*.cc")

#Extensions
server.extend('DQMRenderLink', server.pathOfPlugin('render'))
server.extend('DQMToJSON')
#server.extend('DQMFileAccess', None, "/home/dqmlocal/upload",
#              { "data": "/home/dqmlocal/data",
#                "upload": "/home/dqmlocal/upload" })

#Sources
server.source('DQMUnknown')
server.source('DQMOverlay')
server.source('DQMStripChart')
server.source('DQMLive', 'localhost:9190')
server.source('DQMArchive', BASEDIR + '/idx', '^/Global/')
server.source('DQMLayout', *LAYOUTS)

#Workspace and Services
execfile(CONFIGDIR + "/dqm-services.py")
execfile(CONFIGDIR + "/workspaces-online.py")
