# Service example file (called by serviceTest.cfg)
# has to be used together with loggingServer.py

import libFWCorePython as edm
import xmlrpclib


class Service:
 
  def __init__(self):
    print "PythonService: constructor called"
    try:
      self.server =  xmlrpclib.Server('http://localhost:8000')
      self.server.logme("Job started")
    except:
      print "Can't connect to RPC server"


  def postBeginJob(self):
    print "PythonService: postBeginJob"

 
  def postEndJob(self):
    print "PythonService: postEndJob"

 
  def postProcessEvent(self, event):
    print "PythonService: postProcessEvent"
    handle = edm.Handle("edmtest::IntProduct")
    event.getByLabel("int",handle)
    print handle.get().value
    try: 
      print self.server.logme("executing PostProcessEvent")
    except:
      pass


service = Service()
