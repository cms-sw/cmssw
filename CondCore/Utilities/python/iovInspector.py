#
# iov info server backend
#
#  it assumes that all magic and incantations are done...
#

import pluginCondDBPyInterface as CondDB

class WhatDescription :
       def __init__(self,w) :
              self.__me = w
              self.__ret ={}
       def describe(self) :
              atts = (att for att in dir(self.__me) if not (att[0]=='_' or att[0:4]=='set_' or att[0:6]=='descr_'))
              for att in atts:
                     exec('a=w.'+att+'()')
                     if (a.__class__==CondDB.VInt):
                            if(hasattr(self.__me,'descr_'+att)) :
                                   self.multiple(att)
                            else :
                                   self.commaSeparated(att)
                     else :
                            self.single(att,a)
              return self.__ret

       def single(self,att) :
              w = self.__me
              self.__ret[att]=('single',[val for val in dir(a) if not (val[0]=='_' or val=='name'or val=='values')])

       def multiple(self,att) :
              w = self.__me
              exec('d=w.descr_'+att+'()')
              self.__ret[att]=('multiple',[val for val in d])

       def commaSeparated(self,att) :
              w = self.__me
              self.__ret[att]=('commaSeparated',[])
       
       
def extractorWhat(db, tag) :
       exec('import '+db.moduleName(tag)+' as Plug')
       ret ={}
       w = Plug.What()
       atts = (att for att in dir(w) if not (att[0]=='_' or att[0:4]=='set_' or att[0:6]=='descr_'))
       for att in atts:
              exec('a=w.'+att+'()')
              ret[att]=[val for val in dir(a) if not (val[0]=='_' or val=='name'or val=='values')]
       return ret

def setWhat(w,ret) :
       for key in ret.keys():
              exec ('w.set_'+key+'(w.'+key+'().'+ret[key]+')')
       return w


class Iov :
       def __init__(self, db, tag, since=0, till=0) :
           self.__db = db
           self.__tag = tag
           try : 
               self.__modName = db.moduleName(tag)
               exec('import '+self.__modName+' as Plug')
           except RuntimeError :
               self.__modName = 0
           self.__me = db.iov(tag)
           if (till) : self.__me.setRange(since,till)

       def list(self) :
           ret = []
           for elem in self.__me.elements :
               ret.append( (elem.payloadToken(), elem.since(), elem.till(),0))
           return ret

       def summaries(self) :
           if (self.__modName==0) : return ["no plugin for "  + self.__tag+" no summaries"]
           exec('import '+self.__modName+' as Plug')
           ret = []
           for elem in self.__me.elements :
               p = Plug.Object(elem)
               ret.append( (elem.payloadToken(), elem.since(), elem.till(), p.summary()))
           return ret
           
       def trend(self, what) :
           if (self.__modName==0) : return ["no plugin for "  + self.__tag+" no trend"]
           exec('import '+self.__modName+' as Plug')
           ret = []
           w = setWhat(Plug.What(),what)
           ex = Plug.Extractor(w)
           for elem in self.__me.elements :
               p = Plug.Object(elem)
               p.extract(ex)
               v = [i for i in ex.values()]
               ret.append((elem.since(),elem.till(),v))
           return ret  


class PayLoad :
    def __init__(self, db, token) :
        self.__db = db
        self.__token = token
        exec('import '+db.moduleName(token)+' as Plug')
        self.__me = Plug.Object(db.payLoad(token))

    def __str__(self) :
        return self.__me.dump()

    def object(self) :
        return self.__me

    def summary(self) :
        return self.__me.summary()

    def plot(self, fname, s, il, fl) :
        vi = CondDB.VInt()
        vf = CondDB.VFloat()
        for i in il:
            vi.append(int(i))
        for i in fl:
            vf.append(float(i))
        return self.__me.plot(fname,s,vi,vf)




