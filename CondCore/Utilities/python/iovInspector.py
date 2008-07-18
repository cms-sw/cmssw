#
# iov info server backend
#
#  it assumes that all magic and incantations are done...
#

import pluginCondDBPyInterface as CondDB

class PayLoad :
    def __init__(self, db, token) :
        self.__db = db
        self.__token = token
        exec('import '+db.moduleName(token)+' as Plug')
        self.__me = Plug.Object(db.payLoad(token))

    def ___str__(self) :
        return self.__me.dump()


