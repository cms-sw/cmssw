#
# iov info server backend
#
#  it assumes that all magic and incantations are done...
#
import time
import pluginCondDBPyInterface as CondDB

class WhatDescription :
       def __init__(self,w) :
              self.__me = w
              self.__ret ={}
       def describe(self) :
              _w = self.__me
              _atts = (att for att in dir(self.__me) if not (att[0]=='_' or att[0:4]=='set_' or att[0:6]=='descr_'))
              for _att in _atts:
                     exec('_a=_w.'+_att+'()')
                     if (_a.__class__==CondDB.VInt):
                            if(hasattr(self.__me,'descr_'+_att)) :
                                   self.multiple(_att)
                            else :
                                   self.commaSeparated(_att)
                     else :
                            self.single(_att,_a)
              return self.__ret

       def single(self,att,a) :
              self.__ret[att]=('single',[val for val in dir(a) if not (val[0]=='_' or val=='name'or val=='values')])

       def multiple(self,att) :
              _w = self.__me
              exec('_d=_w.descr_'+att+'()')
              self.__ret[att]=('multiple',[val for val in _d])

       def commaSeparated(self,att) :
              self.__ret[att]=('commaSeparated',[])
       
       
def extractorWhat(db, tag) :
       exec('import '+db.moduleName(tag)+' as Plug')
       ret ={}
       w =  WhatDescription(Plug.What())
       return w.describe()

def setWhat(w,ret) :
       for key in ret.keys():
              _val = ret[key]
              if (type(_val)==type([])) :
                     _vi = CondDB.VInt()
                     for i in _val :
                            _vi.append(i)
                     exec ('w.set_'+key+'(_vi)')
              else :
                     exec ('w.set_'+key+'(w.'+key+'().'+ret[key]+')')
       return w


class Iov :
       def __init__(self, db, tag, since=0, till=0, head=0, tail=0) :
           self.__db = db
           self.__tag = tag
           self.__db.startReadOnlyTransaction()
           try : 
               self.__modName = str(db.payloadModules(tag)[0])
               Plug = __import__(self.__modName)
           except RuntimeError :
               self.__modName = 0

           myDB =  db.iov(tag)
           self.__me = db.iov(tag)
           if (till) : self.__me = myDB.range(since,till)
           if (head) : self.__me = myDB.rangeHead(sinte, till, head)
           if (tail) : self.__me = myDB.rangeTail(since, till, tail)
           self.__db.commitTransaction()

       def list(self) :
           ret = []
           for elem in self.__me.elements :
               ret.append( (elem.payloadToken(), elem.since(), elem.till(),0))
           return ret
    
       def payloadSummaries(self):
           ret = []
           self.__db.startReadOnlyTransaction()
           Plug = __import__(self.__modName)
           payload = Plug.Object(self.__db)
           for elem in self.__me.elements:
              payloadtoken=elem.payloadToken()
              payload.load(elem)
              ret.append(payload.summary())
           self.__db.commitTransaction()
           return ret
           
       def summaries(self) :
           if (self.__modName==0) : return ["no plugin for "  + self.__tag+" no summaries"]
           self.__db.startReadOnlyTransaction()
           Plug = __import__(self.__modName)
           ret = []
           p = Plug.Object(self.__db)
           for elem in self.__me.elements :
               p.load(elem)
               ret.append( (elem.payloadToken(), elem.since(), elem.till(), p.summary()))
           self.__db.commitTransaction()
           return ret
           
       def trend(self, what) :
           if (self.__modName==0) : return ["no plugin for "  + self.__tag+" no trend"]
           self.__db.startReadOnlyTransaction()
           Plug = __import__(self.__modName)
           ret = []
           w = setWhat(Plug.What(),what)
           ex = Plug.Extractor(w)
           p = Plug.Object(self.__db)
           for elem in self.__me.elements :
               p.load(elem)
               p.extract(ex)
               v = [i for i in ex.values()]
               ret.append((elem.since(),elem.till(),v))
           self.__db.commitTransaction()
           return ret
    
       def trendinrange(self, what, head, tail) :
           '''extract trend in the given range. the input parameters are in 64bit integer format. Users should pack the timestamp or lumiid before calling this method
           '''
           if (self.__modName==0) : return ["no plugin for "  + self.__tag+" no trend"]
           self.__db.startReadOnlyTransaction()
           Plug = __import__(self.__modName)
           ret = []
           w = setWhat(Plug.What(),what)
           ex = Plug.Extractor(w)

           p = Plug.Object(self.__db)
           for elem in self.__me.elements :
                  since = elem.since()
                  till = elem.till()
                  if (head < since < tail) or (since < head < till) or (since < tail < till):
                         p.load(elem)
                         p.extract(ex)
                         v = [i for i in ex.values()]
                         ret.append((elem.since(),elem.till(),v))
           self.__db.commitTransaction()
           return ret
    
       def timetype(self):
           return  self.__me.timetype()
       def comment(self):
           return self.__me.comment()
       def revision(self):
           return self.__me.revision()
       def timestamp(self):
           return  CondDB.unpackTime(self.__me.timestamp())
       def payloadClasses(self):
           return list(self.__me.payloadClasses())
       def payLoad(self, since):
           listOfIovElem= [iovElem for iovElem in self.__me.elements if iovElem.since() == since]
           IOVElem = listOfIovElem[0]
           self.__db.startReadOnlyTransaction()
           Plug = __import__(self.__modName)
           payload = Plug.Object(self.__db)
           payload.load(IOVElem)
           self.__db.commitTransaction()
           #print payload
           return payload
              
              
    
class PayLoad :
    def __init__(self, db, tag, elem) :
        self.__db = db
        self.__tag = tag
        self.__elem = elem
        self.__db.startReadOnlyTransaction()
        self.__modName = str(db.payloadModules(tag)[0])
        Plug = __import__(self.__modName)
        self.__me = Plug.Object(db)
        self.__me.load(elem)
        self.__db.commitTransaction()

    def __str__(self) :
        return self.__me.dump()

    def object(self) :
        return self.__me

    def summary(self) :
        return self.__me.summary()
       
    def dump(self) :
        return self.__me.dump()

    def plot(self, fname, s, il, fl) :
        vi = CondDB.VInt()
        vf = CondDB.VFloat()
        for i in il:
            vi.append(int(i))
        for i in fl:
            vf.append(float(i))
        return self.__me.plot(fname,s,vi,vf)

    def trend_plot(self, fname, s, il, fl, sl) :
        vi = CondDB.VInt()
        vf = CondDB.VFloat()
        vs = CondDB.VString()
        for i in il:
            vi.append(int(i))
        for i in fl:
            vf.append(float(i))
        for i in sl:
            vs.append(str(i))
        return self.__me.trend_plot(fname,s,vi,vf,vs)

    def summary_adv(self, s, il, fl, sl):
        #i = int(i)
        vi = CondDB.VInt()
        vf = CondDB.VFloat()
        vs = CondDB.VString()
        for i in il:
            vi.append(int(i))
        for i in fl:
            vf.append(float(i))
        for i in sl:
            vs.append(str(i))
        return self.__me.summary_adv(s,vi,vf,vs)
    
    def dumpFile(self, fname, s, il, fl, sl):
        vi = CondDB.VInt()
        vf = CondDB.VFloat()
        vs = CondDB.VString()
        for i in il:
            vi.append(int(i))
        for i in fl:
            vf.append(float(i))
        for i in sl:
            vs.append(str(i))
        return self.__me.dumpFile(fname,s,vi,vf,vs)
