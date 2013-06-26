import libFWCorePython as edm


class Service:
 
  def __init__(self):
    print "PythonService: constructor called"


  def postBeginJob(self):
    print "PythonService: postBeginJob"

 
  def postEndJob(self):
    print "PythonService: postEndJob"

 
  def postProcessEvent(self, event):
    print "PythonService: postProcessEvent"
    handle = edm.Handle("edmtest::IntProduct")
    event.getByLabel("int",handle)
    print handle.get().value


##################
service = Service()
