
class CutFlowMaker():
    def __init__(self,counter,event,src1,src2 = None):
        self.counter=counter
        self.obj1=src1
        self.obj2=src2
        self.event=event
        
    def setSource1(self,src1):
        self.obj1=src1
        
    def setSource2(self,src2):
        self.obj2=src2
    
    def mergeResults(self,name=None):
        tmp = set(self.obj1)
        tmp.update(self.obj2)
        self.obj1=tmp
        self.obj2=None
        if name is not None:
            setattr(self.event,name,tmp)

    def applyCut(self,cut,text = '',minN = 1,name=None):
        selectedObjects=filter(cut,self.obj1)
        if self.counter is not None:
            if text not in self.counter.dico:
                self.counter.register(text)

        if len(selectedObjects)>=minN:
            if self.counter is not None: 
                self.counter.inc(text)
            self.obj1=selectedObjects

            if name is not None:
                 setattr(self.event,name,self.obj1)
	    return True
        else:
            self.obj1=selectedObjects
            if name is not None:
                setattr(self.event,name,self.obj1)
            return False

    def applyDoubleCut(self,cut1,cut2,text = '',minN = 1,name1 = None, name2 = None,minN1=-1,minN2=-1):
        selectedObjects1=filter(cut1,self.obj1)
        selectedObjects2=filter(cut2,self.obj2)
   
        merged=set(selectedObjects1)
        merged.update(selectedObjects2)

        isOK = (len(merged)>=minN) and \
               (len(selectedObjects1)>=minN1) and \
               (len(selectedObjects2)>=minN2)
        if self.counter is not None:
            if text not in self.counter.dico:
                self.counter.register(text)

        if isOK:
            if self.counter is not None:
                self.counter.inc(text)
            self.obj1=selectedObjects1
            self.obj2=selectedObjects2
            if name1 is not None:
                setattr(self.event,name1,selectedObjects1)
            if name2 is not None:
                setattr(self.event,name2,selectedObjects2)

            return True
        else:
            self.obj1=selectedObjects1
            self.obj2=selectedObjects2
            if name1 is not None:
                setattr(self.event,name1,selectedObjects1)
            if name2 is not None:
                setattr(self.event,name2,selectedObjects2)

        return False
   
