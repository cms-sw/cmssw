#from CrabTask import CrabTask
from crabWrap import checkStatus,getOutput
import os,time
from threading import Thread,Lock,Event

class CrabWatch(Thread):
    #def __init__(self, task):
    #    Thread.__init__(self)
    #    self.task = task
    def __init__(self, project, action = getOutput):
        Thread.__init__(self)
        self.project = project
        self.action = action

        self.lock = Lock()
        self.finish = Event() 
  
    def run(self):
        exit = False
        while not exit:
            if checkStatus(self.project,80.0): break

            self.lock.acquire()
            if self.finish.isSet(): exit = True 
            self.lock.release()
 
            if not exit: time.sleep(180)
 
        print "Finished..."

        self.action(self.project)

if __name__ == '__main__':

    project = None
    import sys
    for opt in sys.argv:
        if opt[:8] == 'project=':
            project = opt[8:] 
 
    if not project: raise ValueError,'Need to set project' 

    crab = CrabWatch(project)
    crab.start()
