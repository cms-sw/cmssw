import os.path, socket
CONFIGDIR = os.path.normcase(os.path.abspath(__file__)).rsplit('/', 1)[0]
PORT = 8060

envsetup = """
 source ~cmssw/cmsset_default.sh
 source /nfshome0/barvic/rpms/slc4_ia32_gcc345/cms/webtools/1.0.0/etc/profile.d/dependencies-setup.sh
"""

server = DQMServerSpec (
  port       = PORT,
  localBase  = '%s:%d' % (socket.gethostname(), PORT),
  serverDir  = '/nfshome0/barvic/gui',
  baseUrl    = '/dqm/online',
  title      = 'CMS &#8211; Online data quality',
  backends   = [
    DQMBackendSpec('dqm', 'Client', [ '--collector localhost:9090' ], waitFor = 'commands'),
#    DQMBackendSpec('csc', 'Layout', [ CONFIGDIR + '/csc-layouts.py' ])
  ])
