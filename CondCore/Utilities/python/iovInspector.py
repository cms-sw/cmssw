#
# iov info server backend
#
#  it assumes that all magic and incantations are done...
#

import pluginCondDBPyInterface as CondDB

class Iov :
       def __init__(self, db, tag) :
           self.__db = db
           self.__tag = tag
           exec('import '+db.moduleName(tag)+' as Plug')
           self.__me = db.iov(tag)

       def list(self) :
           for elem in self.__me.elements :
               p = Plug.Object(elem)
               print elem.since(), elem.till(),p.summary()
  
        


class PayLoad :
    def __init__(self, db, token) :
        self.__db = db
        self.__token = token
        exec('import '+db.moduleName(token)+' as Plug')
        self.__me = Plug.Object(db.payLoad(token))

    def __str__(self) :
        return self.__me.dump()


