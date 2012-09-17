from types import *
from re import *

class LayoutObj(object):
    """Base class for layout objects"""

    global LayoutObj

    replacements_ = {}

    def __init__(self, name):
        self.name_ = name
    
    def expand(self, fil):
        pass

    def isA(self):
        return 'LayoutObj'

    def clone(self, name=''):
        if name == '':
            name = self.name_
        lobj = LayoutObj(name)
        lobj.replacements_.update(self.replacements_)
        return lobj

    def _substitute(self, string):
        for ph in findall(r'\%\(([^\)]+)\)s', string):
            if ph not in self.replacements_:
                self.replacements_[ph] = '%(' + ph + ')s'
            
        return string % self.replacements_

class LayoutDir(LayoutObj):
    """Directory class"""

    global LayoutDir
    global LayoutObj

    pwd_ = ""

    def __init__(self, name, objs, addSerial = True):
        LayoutObj.__init__(self, name)
        self.addSerial_ = addSerial
        # no cloning
        self.objs_ = objs
        # labels to enumerate the objects in the directory; [dirLabel, elemLabel]
        self.labels_ = [0, 0]

    def expand(self, fil):
        pwd = LayoutDir.pwd_
        if len(pwd):
            LayoutDir.pwd_ += "/" + self._substitute(self.name_)
        else:
            LayoutDir.pwd_ = self._substitute(self.name_)
            fil.write("def ecallayout(i, p, *rows): i[p] = DQMItem(layout=rows)\n\n")
            
        self.labels_ = [0, 0]
        for x in self.objs_:
            name = x.name_
            if x.addSerial_:
                x.name_ = '%02d ' % x.incrementIndex(self.labels_) + name
            x.expand(fil)
            x.name_ = name
        LayoutDir.pwd_ = pwd

    def incrementIndex(self, labels):
        i = labels[0]
        labels[0] += 1
        return i

    def isA(self):
        return 'LayoutDir'

    def _copyObjs(self):
        newobjs = []
        for o in self.objs_:
            newobjs.append(o.clone())

        return newobjs

    def clone(self, name=''):
        if name == '':
            name = self.name_
        return LayoutDir(name, self._copyObjs(), self.addSerial_)

    def get(self, relPath):
        parts = relPath.partition('/')
        i = 0
        while i < len(self.objs_):
            x = self.objs_[i]
            if x.name_ == parts[0]:
                if parts[1] == '':
                    return x
                elif x.isA() == 'LayoutDir':
                    ret = x.get(parts[2])
                    if ret:
                        return ret

            i += 1

        return None
    
    def remove(self, relPath):
        parts = relPath.partition('/')
        i = 0
        while i < len(self.objs_):
            x = self.objs_[i]
            if x.name_ == parts[0]:
                if parts[1] == '':
                    self.objs_.remove(x)
                    i -= 1
                elif x.isA() == 'LayoutDir':
                    x.remove(parts[2])
                    if len(x.objs_) == 0:
                        self.objs_.remove(x)
                        i -= 1

            i += 1

    def append(self, obj):
        if type(obj) is ListType:
            self.objs_ = self.objs_ + obj
        else:
            self.objs_.append(obj)

class LayoutElem(LayoutObj):
    """Monitor element wrapper"""

    global LayoutElem
    global LayoutObj
    global DQMItem

    def __init__(self, name, layoutSpecs, addSerial = True):
        LayoutObj.__init__(self, name)
        # no copying
        self.layoutSpecs_ = layoutSpecs
        self.addSerial_ = addSerial

    def expand(self, fil):
        layoutList = []
        for row in self.layoutSpecs_:
            layoutRow = []
            for column in row:
                if len(column) == 0:
                    layoutRow.append(None)
                if len(column) == 1:
                    layoutRow.append({'path' : self._substitute(column[0])})
                elif len(column) == 2:
                    layoutRow.append({'path' : self._substitute(column[0]), 'description' : self._substitute(column[1])})
                else:
                    layoutRow.append({'path' : self._substitute(column[0]), 'description' : self._substitute(column[1]), 'draw' : column[2]})

            layoutList.append(layoutRow)

        
        fil.write("ecallayout(dqmitems, '" + LayoutDir.pwd_ + "/" + self._substitute(self.name_) + "'," + ','.join(map(str, layoutList)) + ")\n")

    def incrementIndex(self, labels):
        i = labels[1]
        labels[1] += 1
        return i

    def isA(self):
        return 'LayoutElem'

    def _copySpecs(self):
        newspecs = []
        for row in self.layoutSpecs_:
            newrow = []
            for column in row:
                newcol = []
                for item in column:
                    newcol.append(item)
                newrow.append(newcol)
            newspecs.append(newrow)

        return newspecs

    def clone(self, name=''):
        if name == '':
            name = self.name_
        return LayoutElem(name, self._copySpecs(), self.addSerial_)

class LayoutSet(LayoutObj):

    global LayoutSet
    global LayoutObj

    def __init__(self, name, repLists):
        LayoutObj.__init__(self, name)
        self.repLists_ = repLists

    def expand(self, fil):
        maxSize = 0
        paramSize = max(map(len, self.repLists_.values()))
        # aggregate the parameters for each entry
        for i in range(paramSize):
            for key, list in self.repLists_.items():
                replacement = ''
                if type(list) is StringType:
                    replacement = list
                elif i >= len(list):
                    replacement = list[len(list) - 1]
                else:
                    replacement = list[i]
                    
                LayoutObj.replacements_[key] = replacement
                
            template = self.generate()
            template.expand(fil)

        for key in self.repLists_.keys():
            LayoutObj.replacements_.pop(key)
            
    def generate(self):
        return LayoutObj(self.name_)

    def isA(self):
        return 'LayoutSet'

    def clone(self, name=''):
        if name == '':
            name = self.name_
        return LayoutSet(name, self.repLists_)

# LayoutSet must be the first inheritance
class LayoutDirSet(LayoutSet, LayoutDir):
    """Set of iteratively produced directories"""

    global LayoutDirSet
    global LayoutDir
    global LayoutSet

    # objs: template of objects to be placed under each generated directory
    def __init__(self, name, objs, repLists, addSerial = True):
        LayoutSet.__init__(self, name, repLists)
        LayoutDir.__init__(self, name, objs, addSerial)

    def generate(self):
        return LayoutDir(self.name_, self.objs_, self.addSerial_)

    def isA(self):
        return 'LayoutDirSet'

    def clone(self, name=''):
        if name == '':
            name = self.name_
        return LayoutDirSet(name, self._copyObjs(), self.repLists_, self.addSerial_)

# LayoutSet must be the first inheritance
class LayoutElemSet(LayoutSet, LayoutElem):
    """Set of iteratively produced elements"""

    global LayoutElemSet
    global LayoutElem
    global LayoutSet

    def __init__(self, name, layoutSpecs, repLists, addSerial = True):
        LayoutSet.__init__(self, name, repLists)
        LayoutElem.__init__(self, name, layoutSpecs, addSerial)

    def generate(self):
        return LayoutElem(self.name_, self.layoutSpecs_, self.addSerial_)

    def isA(self):
        return 'LayoutElemSet'

    def clone(self, name=''):
        if name == '':
            name = self.name_
        return LayoutElemSet(name, self._copySpecs(), self.repLists_, self.addSerial_)
