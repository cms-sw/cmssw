#from CrabTask import CrabTask
from crabWrap import crabStatus,convertStatus,getOutput
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
        self.threshold = 100.0

        self.lock = Lock()
        self.finish = Event() 
 
    def setThreshold(self,threshold):
        self.threshold = float(threshold)

    def run(self):
        exit = False
        failed=0
        while not exit:
            #if checkStatus(self.project,80.0): break
            try:
               status = crabStatus(self.project)
            except CrabException:
               failed+=1
               print "Crab retrieve status failed (",failed,")"
               if failed>10:
                  raise 
            else:
               statusNew = convertStatus(status)
               print "Relative percentage finished: %.0f%%" % statusNew['Finished']
               print "Relative percentage failed  : %.0f%%" % statusNew['Failed']
               print "Relative percentage running : %.0f%%" % statusNew['Running']
               if statusNew['Failed'] > 50.0: raise RuntimeError,'Too many jobs have failed (%.0f%%).' % statusNew['Failed']
               if statusNew['Finished'] >= self.threshold: break
   
               self.lock.acquire()
               if self.finish.isSet(): exit = True 
               self.lock.release()
 
            if not exit: time.sleep(180)
 
        print "Finished..."

        if self.action: self.action(self.project)

if __name__ == '__main__':

    project = None
    import sys
    for opt in sys.argv:
        if opt[:8] == 'project=':
            project = opt[8:] 
 
    if not project: raise ValueError,'Need to set project' 

    crab = CrabWatch(project)
    crab.start()
